"""
Hybrid Chat + RAG endpoint combining:
- Advanced multi-search strategy from /api/search/ask
- Conversational memory from /api/chat/sessions
"""
import asyncio
import json
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


async def execute_multi_search(
    strategy: Strategy, notebook_id: str | None = None
) -> List[Dict[str, Any]]:
    """Execute all searches from strategy and collect chunks."""
    from open_notebook.domain.notebook import text_search

    all_chunks = []
    seen_ids = set()

    # Check if embedding model is available for vector search
    use_vector_search = False
    try:
        embedding_model = await model_manager.get_embedding_model()
        use_vector_search = embedding_model is not None
    except Exception as e:
        logger.warning(f"Could not get embedding model: {e}, will use text search")
        use_vector_search = False

    if use_vector_search:
        logger.info(f"Using vector search for multi-search strategy (notebook_id={notebook_id})")
        try:
            # Execute searches in parallel
            tasks = [
                vector_search(
                    search.term, results=10, source=True, note=True,
                    notebook_id=notebook_id,
                )
                for search in strategy.searches
            ]

            results_list = await asyncio.gather(*tasks)
        except Exception as e:
            logger.warning(f"Vector search failed: {e}, falling back to text search")
            use_vector_search = False

    if not use_vector_search:
        logger.info(f"Using text search for multi-search strategy (notebook_id={notebook_id})")
        # Fallback to text search
        tasks = [
            text_search(
                search.term, results=10, source=True, note=True,
                notebook_id=notebook_id,
            )
            for search in strategy.searches
        ]
        results_list = await asyncio.gather(*tasks)
    
    # Deduplicate and enrich chunks with content
    from open_notebook.database.repository import repo_query, ensure_record_id
    
    for results in results_list:
        if results:
            for chunk in results:
                chunk_id = chunk.get("id")
                parent_id = chunk.get("parent_id") or chunk_id
                
                if chunk_id and chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    
                    # If chunk already has content, use it directly
                    if chunk.get("content") or chunk.get("text") or chunk.get("chunk"):
                        all_chunks.append(chunk)
                        continue
                    
                    # Otherwise, fetch embedding chunks for this source
                    try:
                        if parent_id.startswith("source:"):
                            # Get embedding chunks for this source
                            embedding_chunks = await repo_query(
                                """
                                SELECT content, chunk_index 
                                FROM source_embedding 
                                WHERE source = $source_id 
                                ORDER BY chunk_index 
                                LIMIT 5
                                """,
                                {"source_id": ensure_record_id(parent_id)}
                            )
                            
                            if embedding_chunks:
                                # Add each embedding chunk as a separate result
                                for idx, emb_chunk in enumerate(embedding_chunks):
                                    chunk_copy = chunk.copy()
                                    chunk_copy["content"] = emb_chunk.get("content", "")
                                    chunk_copy["chunk_index"] = emb_chunk.get("chunk_index", idx)
                                    chunk_copy["id"] = f"{parent_id}_chunk_{idx}"
                                    all_chunks.append(chunk_copy)
                            else:
                                # No embeddings, try full text
                                from open_notebook.domain.notebook import Source
                                source = await Source.get(parent_id)
                                if source and source.full_text:
                                    chunk["content"] = source.full_text[:3000]
                                    all_chunks.append(chunk)
                        
                        elif parent_id.startswith("note:"):
                            from open_notebook.domain.notebook import Note
                            note = await Note.get(parent_id)
                            if note:
                                chunk["content"] = note.content or ""
                                all_chunks.append(chunk)
                        
                        elif parent_id.startswith("source_insight:"):
                            from open_notebook.domain.notebook import SourceInsight
                            insight = await SourceInsight.get(parent_id)
                            if insight:
                                chunk["content"] = insight.content or ""
                                all_chunks.append(chunk)
                        
                    except Exception as e:
                        logger.warning(f"Could not enrich chunk {chunk_id}: {e}")
                        # Still add chunk without content
                        all_chunks.append(chunk)
    
    logger.info(f"Enriched {len(all_chunks)} chunks with content")
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

        # Immediate first byte so the proxy sees the stream is alive.
        yield _sse({"type": "planning"})

        # ---- Phase 1: planning agent (with keep-alive) ----
        planning_task = asyncio.create_task(
            generate_search_strategy(message, strategy_model_id)
        )
        async for keepalive in _drain_with_keepalive(
            planning_task, {"type": "planning"}
        ):
            yield keepalive
        strategy = await planning_task

        reasoning = (strategy.reasoning or "").strip()
        # Truncate: we just want to show something is happening, not the full CoT.
        truncated = reasoning[:180] + ("…" if len(reasoning) > 180 else "")
        yield _sse({"type": "plan", "reasoning": truncated})

        # ---- Phase 2: multi-search (with keep-alive) ----
        search_task = asyncio.create_task(
            execute_multi_search(strategy, notebook_id=notebook_id)
        )
        async for keepalive in _drain_with_keepalive(
            search_task, {"type": "searching"}
        ):
            yield keepalive
        chunks = await search_task

        yield _sse({"type": "strategy", "chunks_retrieved": len(chunks)})

        # ---- Phase 3: responding agent (token-streamed, bypasses chat_graph) ----
        # Load prior conversation from the checkpointer.
        current_state = await asyncio.to_thread(
            chat_graph.get_state,
            config=RunnableConfig(configurable={"thread_id": full_session_id}),
        )
        prior_messages = (
            list(current_state.values.get("messages", []))
            if current_state and current_state.values
            else []
        )

        context_data = {
            "sources": [
                {
                    "id": chunk.get("id"),
                    "content": chunk.get("content", ""),
                    "title": chunk.get("title", ""),
                    "score": chunk.get("score", 0),
                }
                for chunk in chunks
            ]
        }

        system_prompt = Prompter(prompt_template="chat/system").render(
            data={
                "messages": prior_messages,
                "notebook": None,
                "context": context_data,
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
        yield _sse({"type": "error", "message": str(e)})


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
