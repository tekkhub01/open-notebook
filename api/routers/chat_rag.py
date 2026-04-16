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
from langchain_core.messages import HumanMessage
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


async def execute_multi_search(strategy: Strategy) -> List[Dict[str, Any]]:
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
        logger.info("Using vector search for multi-search strategy")
        try:
            # Execute searches in parallel
            tasks = [
                vector_search(search.term, results=10, source=True, note=True)
                for search in strategy.searches
            ]
            
            results_list = await asyncio.gather(*tasks)
        except Exception as e:
            logger.warning(f"Vector search failed: {e}, falling back to text search")
            use_vector_search = False
    
    if not use_vector_search:
        logger.info("Using text search for multi-search strategy")
        # Fallback to text search
        tasks = [
            text_search(search.term, results=10, source=True, note=True)
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


async def stream_chat_rag_response(
    session_id: str,
    message: str,
    chunks: List[Dict[str, Any]],
    model_override: Optional[str],
) -> AsyncGenerator[str, None]:
    """Stream chat response with RAG context."""
    try:
        # Get session
        full_session_id = (
            session_id
            if session_id.startswith("chat_session:")
            else f"chat_session:{session_id}"
        )
        session = await ChatSession.get(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Use model override (per-request or session-level)
        final_model_override = (
            model_override
            if model_override is not None
            else getattr(session, "model_override", None)
        )

        # Get current state
        current_state = await asyncio.to_thread(
            chat_graph.get_state,
            config=RunnableConfig(configurable={"thread_id": full_session_id}),
        )

        # Prepare state
        state_values = current_state.values if current_state else {}
        state_values["messages"] = state_values.get("messages", [])
        
        # Build context from chunks
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
        
        state_values["context"] = context_data
        state_values["model_override"] = final_model_override

        # Add user message
        user_message = HumanMessage(content=message)
        state_values["messages"].append(user_message)

        # Stream strategy phase
        strategy_event = {
            "type": "strategy",
            "chunks_retrieved": len(chunks),
        }
        yield f"data: {json.dumps(strategy_event)}\n\n"

        # Execute chat graph
        result = chat_graph.invoke(
            input=state_values,
            config=RunnableConfig(
                configurable={
                    "thread_id": full_session_id,
                    "model_id": final_model_override,
                }
            ),
        )

        # Update session timestamp
        await session.save()

        # Stream the final answer
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            content = (
                last_message.content
                if hasattr(last_message, "content")
                else str(last_message)
            )
            
            answer_event = {
                "type": "answer",
                "content": content,
            }
            yield f"data: {json.dumps(answer_event)}\n\n"

        # Send completion
        completion_event = {
            "type": "complete",
            "chunks_used": len(chunks),
        }
        yield f"data: {json.dumps(completion_event)}\n\n"

    except Exception as e:
        logger.error(f"Error in chat RAG streaming: {str(e)}")
        error_event = {"type": "error", "message": str(e)}
        yield f"data: {json.dumps(error_event)}\n\n"


@router.post("/chat/rag/execute")
async def execute_chat_rag(request: ChatRAGRequest):
    """
    Execute chat with advanced RAG retrieval.
    
    Combines:
    1. Multi-search strategy generation (like /api/search/ask)
    2. Parallel vector searches with deduplication
    3. Conversational memory (like /api/chat/execute)
    
    Returns streaming response with:
    - strategy: Multi-search plan and chunks retrieved
    - answer: AI response with conversational context
    - complete: Final status
    """
    try:
        # Verify session exists
        full_session_id = (
            request.session_id
            if request.session_id.startswith("chat_session:")
            else f"chat_session:{request.session_id}"
        )
        session = await ChatSession.get(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Check embedding model availability (skip for now, will fallback to text search)
        # if not await model_manager.get_embedding_model():
        #     raise HTTPException(
        #         status_code=400,
        #         detail="Vector search requires an embedding model. Please configure one.",
        #     )

        # Get strategy model (use default if not provided)
        strategy_model_id = request.strategy_model
        if not strategy_model_id:
            # Try to get from model_manager defaults
            defaults = await model_manager.get_defaults()
            # Use default_tools_model for strategy, fallback to transformation or chat
            strategy_model_id = (
                getattr(defaults, "default_tools_model", None) or 
                getattr(defaults, "default_transformation_model", None) or
                getattr(defaults, "default_chat_model", None)
            )
            if not strategy_model_id:
                raise HTTPException(
                    status_code=400,
                    detail="No strategy model specified and no default configured",
                )

        # Validate strategy model exists
        strategy_model = await Model.get(strategy_model_id)
        if not strategy_model:
            raise HTTPException(
                status_code=400,
                detail=f"Strategy model {strategy_model_id} not found",
            )

        # Phase 1: Generate search strategy
        logger.info(f"Generating search strategy for: {request.message}")
        strategy = await generate_search_strategy(request.message, strategy_model_id)
        logger.info(
            f"Strategy generated: {len(strategy.searches)} searches planned"
        )

        # Phase 2: Execute multi-search
        logger.info("Executing multi-search...")
        chunks = await execute_multi_search(strategy)
        logger.info(f"Retrieved {len(chunks)} unique chunks")

        # Phase 3: Stream chat response with RAG context
        if request.stream:
            return StreamingResponse(
                stream_chat_rag_response(
                    request.session_id,
                    request.message,
                    chunks,
                    request.model_override,
                ),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming response (collect all events)
            events = []
            async for event_str in stream_chat_rag_response(
                request.session_id,
                request.message,
                chunks,
                request.model_override,
            ):
                if event_str.startswith("data: "):
                    event_data = json.loads(event_str[6:])
                    events.append(event_data)
            
            # Extract final answer
            answer_content = ""
            for event in events:
                if event.get("type") == "answer":
                    answer_content = event.get("content", "")
                    break
            
            return ChatRAGResponse(
                session_id=request.session_id,
                messages=[
                    ChatMessage(
                        id="final",
                        type="assistant",
                        content=answer_content,
                    )
                ],
                chunks_used=len(chunks),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat RAG execute: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Chat RAG execution failed: {str(e)}"
        )
