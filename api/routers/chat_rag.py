"""
Hybrid Chat + RAG endpoint combining:
- Advanced multi-search strategy from /api/search/ask
- Conversational memory from /api/chat/sessions
"""
import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from loguru import logger
from pydantic import BaseModel, Field

from ai_prompter import Prompter
from open_notebook.ai.models import Model, model_manager
from open_notebook.ai.provision import provision_langchain_model
from open_notebook.domain.notebook import ChatSession, vector_search
from open_notebook.exceptions import NotFoundError
from open_notebook.graphs.chat import graph as chat_graph
from open_notebook.utils import clean_thinking_content
from open_notebook.utils.text_utils import extract_text_content

# Reuse Strategy model from ask.py
from open_notebook.graphs.ask import Strategy

router = APIRouter()


# Request/Response models
class ChatRAGRequest(BaseModel):
    session_id: str = Field(..., description="Chat session ID")
    message: str = Field(..., description="User message")
    notebook_id: Optional[str] = Field(None, description="Notebook ID for search scope")
    strategy_model: Optional[str] = Field(None, description="Model for search strategy")
    model_override: Optional[str] = Field(None, description="Model for chat response")
    stream: bool = Field(default=True, description="Stream response")


class ChatMessage(BaseModel):
    id: str
    type: str
    content: str


class ChatRAGResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    chunks_used: int = Field(..., description="Number of RAG chunks retrieved")


async def generate_search_strategy(
    question: str, strategy_model_id: str
) -> Strategy:
    """Generate multi-search strategy using the ask graph logic."""
    parser = PydanticOutputParser(pydantic_object=Strategy)
    system_prompt = Prompter(
        prompt_template="ask/entry", parser=parser
    ).render(data={"question": question})

    model = await provision_langchain_model(
        system_prompt,
        strategy_model_id,
        "tools",
        max_tokens=2000,
        structured=dict(type="json"),
    )

    ai_message = await model.ainvoke(system_prompt)
    message_content = (
        ai_message.content
        if isinstance(ai_message.content, str)
        else str(ai_message.content)
    )
    cleaned_content = clean_thinking_content(message_content)
    strategy = parser.parse(cleaned_content)

    return strategy


async def condense_question(
    question: str,
    prior_messages: List[Any],
    strategy_model_id: str,
    max_recent: int = 6,
) -> str:
    """Rewrite a follow-up message as a standalone question using the last
    few turns of conversation. Keeps retrieval sane when the user says things
    like "tell me more about that" — the planner only sees the latest message
    otherwise and can't disambiguate references.

    Fails open: on any error or empty output, returns the original question.
    """
    if not prior_messages:
        return question

    recent = prior_messages[-max_recent:]
    log_lines = []
    for msg in recent:
        is_human = isinstance(msg, HumanMessage)
        role = "User" if is_human else "Assistant"
        raw = getattr(msg, "content", msg)
        content = extract_text_content(raw) if raw is not None else ""
        # Keep each line short — we don't need full prior answers to disambiguate.
        if content and len(content) > 600:
            content = content[:600] + "…"
        log_lines.append(f"{role}: {content}")
    history = "\n".join(log_lines)

    prompt = (
        "You rewrite follow-up chat messages into standalone search queries.\n"
        "Given the conversation and the user's latest message, output ONE "
        "standalone question that captures what the user is asking, resolving "
        "pronouns and implicit references using the prior turns.\n"
        "Rules:\n"
        "- Use the SAME language as the user's latest message.\n"
        "- If the message is already fully standalone, return it verbatim.\n"
        "- Output only the rewritten question. No quotes, prefix, or explanation.\n\n"
        f"# CONVERSATION\n{history}\n\n"
        f"# LATEST MESSAGE\n{question}\n\n"
        "# STANDALONE QUESTION\n"
    )

    try:
        model = await provision_langchain_model(
            prompt, strategy_model_id, "tools", max_tokens=150
        )
        ai_message = await model.ainvoke(prompt)
        rewritten = extract_text_content(ai_message.content)
        rewritten = clean_thinking_content(rewritten).strip()
        # Strip surrounding quotes if the model added them.
        if len(rewritten) >= 2 and rewritten[0] in ('"', "'", "«") and rewritten[-1] in ('"', "'", "»"):
            rewritten = rewritten[1:-1].strip()
        return rewritten or question
    except Exception as e:
        logger.warning(f"Query condensation failed; using original question: {e}")
        return question


async def execute_multi_search(
    strategy: Strategy,
    notebook_id: str | None = None,
    min_similarity: float = 0.5,
    per_search_results: int = 10,
) -> List[Dict[str, Any]]:
    """Execute all searches from the strategy, normalize hits, and deduplicate.

    vector_search returns rows of `{id, parent_id, title, similarity, matches}`
    where `matches` is a flattened array of content strings for that record.
    text_search returns `{id, parent_id, title, relevance}` with no content.

    Dedup key is `id` (not parent_id) so a source and its derived "Dense Summary"
    insight — which share parent_id but have different ids — both stay in play.
    Highest score wins on collision.
    """
    from open_notebook.domain.notebook import text_search

    use_vector_search = False
    try:
        embedding_model = await model_manager.get_embedding_model()
        use_vector_search = embedding_model is not None
    except Exception as e:
        logger.warning(f"Could not get embedding model: {e}, will use text search")

    results_list: List[List[Dict[str, Any]]] = []
    if use_vector_search:
        logger.info(
            f"Using vector search (min_similarity={min_similarity}, notebook={notebook_id})"
        )
        try:
            results_list = await asyncio.gather(
                *[
                    vector_search(
                        s.term,
                        results=per_search_results,
                        source=True,
                        note=True,
                        minimum_score=min_similarity,
                        notebook_id=notebook_id,
                    )
                    for s in strategy.searches
                ]
            )
        except Exception as e:
            logger.warning(f"Vector search failed: {e}; falling back to text search")
            use_vector_search = False

    if not use_vector_search:
        logger.info(f"Using text search fallback (notebook={notebook_id})")
        results_list = await asyncio.gather(
            *[
                text_search(
                    s.term,
                    results=per_search_results,
                    source=True,
                    note=True,
                    notebook_id=notebook_id,
                )
                for s in strategy.searches
            ]
        )

    # Normalize + dedup on id (keep highest score).
    best_by_id: Dict[str, Dict[str, Any]] = {}
    for results in results_list:
        if not results:
            continue
        for hit in results:
            hit_id = str(hit.get("id")) if hit.get("id") is not None else None
            if not hit_id:
                continue

            score = hit.get("similarity") or hit.get("relevance") or 0.0
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0

            raw_matches = hit.get("matches")
            if isinstance(raw_matches, list):
                content = "\n\n".join(str(m) for m in raw_matches if m)
            elif raw_matches:
                content = str(raw_matches)
            else:
                content = hit.get("content") or ""

            normalized = {
                "id": hit_id,
                "parent_id": str(hit.get("parent_id"))
                if hit.get("parent_id") is not None
                else hit_id,
                "title": hit.get("title", "") or "",
                "score": score,
                "content": content,
            }

            existing = best_by_id.get(hit_id)
            if existing is None or existing["score"] < normalized["score"]:
                best_by_id[hit_id] = normalized

    # text_search hits have no content; fetch from the owning record so the LLM
    # has something substantive to work with.
    need_content = [c for c in best_by_id.values() if not c["content"]]
    if need_content:
        from open_notebook.domain.notebook import Note, Source, SourceInsight

        for chunk in need_content:
            pid = chunk["parent_id"]
            try:
                if pid.startswith("source:"):
                    src = await Source.get(pid)
                    if src and src.full_text:
                        chunk["content"] = src.full_text[:3000]
                elif pid.startswith("note:"):
                    note = await Note.get(pid)
                    if note:
                        chunk["content"] = note.content or ""
                elif pid.startswith("source_insight:"):
                    insight = await SourceInsight.get(pid)
                    if insight:
                        chunk["content"] = insight.content or ""
            except Exception as e:
                logger.warning(f"Could not enrich {chunk['id']}: {e}")

    all_chunks = sorted(
        best_by_id.values(), key=lambda c: c.get("score", 0.0), reverse=True
    )
    logger.info(
        f"Retrieved {len(all_chunks)} unique chunks (from "
        f"{sum(len(r or []) for r in results_list)} raw hits)"
    )
    return all_chunks


def _sse(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj)}\n\n"


async def _drain_with_keepalive(
    task: "asyncio.Task[Any]",
    keepalive_event: Dict[str, Any],
    interval: float = 3.0,
) -> AsyncGenerator[str, None]:
    """Await `task`, emitting `keepalive_event` every `interval` seconds while it runs.

    Guarantees bytes flow at least every `interval` seconds so proxies (Netlify,
    nginx) don't close the connection during long blocking LLM calls.
    """
    while not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=interval)
        except asyncio.TimeoutError:
            yield _sse(keepalive_event)


async def stream_chat_rag_response(
    session_id: str,
    message: str,
    notebook_id: Optional[str],
    strategy_model_id: str,
    model_override: Optional[str],
) -> AsyncGenerator[str, None]:
    """Stream the full two-agent RAG pipeline.

    Phase 1: planning agent (strategy) — runs as a task with keep-alive ticks.
    Phase 2: multi-search — runs as a task with keep-alive ticks.
    Phase 3: responding agent — streamed token-by-token via model.astream,
             bypassing chat_graph (state persisted manually at the end).
    """
    full_session_id = (
        session_id
        if session_id.startswith("chat_session:")
        else f"chat_session:{session_id}"
    )
    # Trace state: filled incrementally so we can write it even on error paths.
    t_start = time.monotonic()
    trace: Dict[str, Any] = {
        "session_id": full_session_id,
        "notebook_id": notebook_id,
        "question": message,
        "strategy_model": strategy_model_id,
        "chat_model": model_override,
    }
    strategy = None
    chunks: List[Dict[str, Any]] = []
    final_content = ""
    try:
        session = await ChatSession.get(full_session_id)
        if not session:
            yield _sse({"type": "error", "message": "Session not found"})
            return

        final_model_override = (
            model_override
            if model_override is not None
            else getattr(session, "model_override", None)
        )
        trace["chat_model"] = final_model_override

        # Immediate first byte so the proxy sees the stream is alive.
        yield _sse({"type": "planning"})

        # Load prior messages up front so we can condense follow-up questions
        # before the planner sees them. Cheap op (local sqlite).
        current_state = await asyncio.to_thread(
            chat_graph.get_state,
            config=RunnableConfig(configurable={"thread_id": full_session_id}),
        )
        prior_messages = (
            list(current_state.values.get("messages", []))
            if current_state and current_state.values
            else []
        )

        # ---- Phase 0: query condensation (only if there's history) ----
        # Rewrites "tell me more about that" into a standalone question using
        # recent turns, so the planner + retrieval aren't flying blind.
        search_question = message
        if prior_messages:
            t_condense = time.monotonic()
            condense_task = asyncio.create_task(
                condense_question(message, prior_messages, strategy_model_id)
            )
            async for keepalive in _drain_with_keepalive(
                condense_task, {"type": "planning"}
            ):
                yield keepalive
            search_question = await condense_task
            trace["condensed_question"] = search_question
            trace["condense_ms"] = int((time.monotonic() - t_condense) * 1000)

        # ---- Phase 1: planning agent (with keep-alive) ----
        t_phase = time.monotonic()
        planning_task = asyncio.create_task(
            generate_search_strategy(search_question, strategy_model_id)
        )
        async for keepalive in _drain_with_keepalive(
            planning_task, {"type": "planning"}
        ):
            yield keepalive
        strategy = await planning_task
        trace["planning_ms"] = int((time.monotonic() - t_phase) * 1000)

        reasoning = (strategy.reasoning or "").strip()
        # Truncate: we just want to show something is happening, not the full CoT.
        truncated = reasoning[:180] + ("…" if len(reasoning) > 180 else "")
        yield _sse({"type": "plan", "reasoning": truncated})

        # ---- Phase 2: multi-search (with keep-alive) ----
        t_phase = time.monotonic()
        search_task = asyncio.create_task(
            execute_multi_search(strategy, notebook_id=notebook_id)
        )
        async for keepalive in _drain_with_keepalive(
            search_task, {"type": "searching"}
        ):
            yield keepalive
        chunks = await search_task
        trace["search_ms"] = int((time.monotonic() - t_phase) * 1000)

        yield _sse({"type": "strategy", "chunks_retrieved": len(chunks)})

        # ---- Phase 3: responding agent (token-streamed, bypasses chat_graph) ----
        t_phase = time.monotonic()

        # Render context as a clean per-source markdown block. Jinja was
        # previously handed a dict which stringified to Python repr — ugly for
        # the LLM and the citation IDs embedded in the repr were fabricated.
        # Now: one section per real record, real IDs, content straight from
        # vector_search.matches.
        context_blocks: List[str] = []
        for c in chunks:
            content = (c.get("content") or "").strip()
            if not content:
                continue
            header = f"### [{c['id']}] {c.get('title') or c['id']}"
            context_blocks.append(f"{header}\n{content}")
        context_text = "\n\n".join(context_blocks) if context_blocks else ""

        # Keep the structured form around for trace persistence.
        context_data = {
            "sources": [
                {
                    "id": c.get("id"),
                    "content": c.get("content", ""),
                    "title": c.get("title", ""),
                    "score": c.get("score", 0),
                }
                for c in chunks
            ]
        }

        system_prompt = Prompter(prompt_template="chat/system").render(
            data={
                "messages": prior_messages,
                "notebook": None,
                "context": context_text,
                "context_config": None,
                "model_override": final_model_override,
            }
        )

        user_message = HumanMessage(content=message)
        payload = (
            [SystemMessage(content=system_prompt)] + prior_messages + [user_message]
        )

        chat_model = await provision_langchain_model(
            str(payload), final_model_override, "chat", max_tokens=8192
        )

        # Stream tokens, applying clean_thinking_content progressively so <think>
        # blocks don't leak to the client.
        raw_buffer = ""
        emitted_len = 0
        async for chunk in chat_model.astream(payload):
            delta_raw = getattr(chunk, "content", None)
            if delta_raw is None:
                continue
            delta_text = extract_text_content(delta_raw)
            if not delta_text:
                continue
            raw_buffer += delta_text
            cleaned = clean_thinking_content(raw_buffer)
            if len(cleaned) > emitted_len:
                new_text = cleaned[emitted_len:]
                emitted_len = len(cleaned)
                yield _sse({"type": "answer_delta", "content": new_text})

        final_content = clean_thinking_content(raw_buffer)
        trace["answer_ms"] = int((time.monotonic() - t_phase) * 1000)

        # Final consolidated answer event (keeps non-streaming path working).
        yield _sse({"type": "answer", "content": final_content})

        # Persist both user + assistant messages to the checkpointer so future
        # turns see conversation history. `add_messages` reducer handles append.
        ai_msg = AIMessage(content=final_content)
        await asyncio.to_thread(
            chat_graph.update_state,
            RunnableConfig(
                configurable={
                    "thread_id": full_session_id,
                    "model_id": final_model_override,
                }
            ),
            {
                "messages": [user_message, ai_msg],
                "context": context_data,
                "model_override": final_model_override,
            },
            as_node="agent",
        )

        await session.save()

        yield _sse({"type": "complete", "chunks_used": len(chunks)})

    except Exception as e:
        logger.error(f"Error in chat RAG streaming: {str(e)}")
        trace["error"] = str(e)
        yield _sse({"type": "error", "message": str(e)})
    finally:
        trace["total_ms"] = int((time.monotonic() - t_start) * 1000)
        trace["final_answer"] = final_content
        trace["chunks_count"] = len(chunks)
        if strategy is not None:
            trace["strategy_reasoning"] = strategy.reasoning or ""
            trace["strategy_searches"] = [
                {
                    "term": getattr(s, "term", ""),
                    "instructions": getattr(s, "instructions", ""),
                }
                for s in (strategy.searches or [])
            ]
        # Keep chunk snapshots small — only identifiers, scores, and short preview.
        trace["chunks_retrieved"] = [
            {
                "id": str(c.get("id")) if c.get("id") is not None else None,
                "parent_id": str(c.get("parent_id")) if c.get("parent_id") else None,
                "title": c.get("title", ""),
                "score": c.get("score")
                or c.get("similarity")
                or c.get("relevance"),
                "content_preview": (c.get("content") or "")[:500],
            }
            for c in chunks[:20]
        ]
        try:
            from open_notebook.database.repository import repo_create

            await repo_create("rag_trace", trace)
        except Exception as trace_err:
            logger.warning(f"Failed to persist rag_trace: {trace_err}")


async def _resolve_strategy_model(requested: Optional[str]) -> str:
    """Resolve and validate the strategy model id (falling back to defaults)."""
    strategy_model_id = requested
    if not strategy_model_id:
        defaults = await model_manager.get_defaults()
        strategy_model_id = (
            getattr(defaults, "default_tools_model", None)
            or getattr(defaults, "default_transformation_model", None)
            or getattr(defaults, "default_chat_model", None)
        )
        if not strategy_model_id:
            raise HTTPException(
                status_code=400,
                detail="No strategy model specified and no default configured",
            )
    strategy_model = await Model.get(strategy_model_id)
    if not strategy_model:
        raise HTTPException(
            status_code=400,
            detail=f"Strategy model {strategy_model_id} not found",
        )
    return strategy_model_id


@router.post("/chat/rag/execute")
async def execute_chat_rag(request: ChatRAGRequest):
    """Execute chat with advanced RAG retrieval.

    Streams a two-agent pipeline:
    - Phase 1: planning agent produces a multi-search strategy (emits `plan`).
    - Phase 2: parallel vector/text searches retrieve chunks (emits `strategy`).
    - Phase 3: responding agent streams the final answer token-by-token
               (emits `answer_delta` repeatedly, then a consolidated `answer`).

    All phases run inside the streaming generator so the first SSE byte flushes
    immediately — critical for proxies (Netlify Edge) that close idle streams.
    """
    try:
        full_session_id = (
            request.session_id
            if request.session_id.startswith("chat_session:")
            else f"chat_session:{request.session_id}"
        )
        session = await ChatSession.get(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        strategy_model_id = await _resolve_strategy_model(request.strategy_model)

        if request.stream:
            return StreamingResponse(
                stream_chat_rag_response(
                    request.session_id,
                    request.message,
                    request.notebook_id,
                    strategy_model_id,
                    request.model_override,
                ),
                media_type="text/event-stream",
            )

        # Non-streaming path: drain the generator and collect the final answer.
        answer_content = ""
        chunks_used = 0
        async for event_str in stream_chat_rag_response(
            request.session_id,
            request.message,
            request.notebook_id,
            strategy_model_id,
            request.model_override,
        ):
            if not event_str.startswith("data: "):
                continue
            event_data = json.loads(event_str[6:])
            if event_data.get("type") == "answer":
                answer_content = event_data.get("content", "")
            elif event_data.get("type") == "complete":
                chunks_used = event_data.get("chunks_used", 0)
            elif event_data.get("type") == "error":
                raise HTTPException(
                    status_code=500, detail=event_data.get("message", "RAG error")
                )

        return ChatRAGResponse(
            session_id=request.session_id,
            messages=[
                ChatMessage(id="final", type="assistant", content=answer_content)
            ],
            chunks_used=chunks_used,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat RAG execute: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Chat RAG execution failed: {str(e)}"
        )
