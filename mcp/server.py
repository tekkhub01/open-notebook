"""MCP server exposing Open Notebook's chat-rag pipeline (and a few helpers)
as tools for MCP clients like lobe-chat.

Transport: Streamable HTTP. Mount path defaults to /mcp/ — point your MCP
client at http://<host>:8080/mcp/.

Auth: every upstream request carries `Authorization: Bearer ${OPEN_NOTEBOOK_PASSWORD}`
matching the FastAPI PasswordAuthMiddleware (api/auth.py).

Defaults pickable via env so a single deployment can target a specific notebook
without the MCP client having to know SurrealDB IDs:
  OPEN_NOTEBOOK_ENDPOINT       e.g. http://open_notebook:5055
  OPEN_NOTEBOOK_PASSWORD       must match the backend's password
  OPEN_NOTEBOOK_NOTEBOOK_ID    optional default for chat_rag / search
  OPEN_NOTEBOOK_STRATEGY_MODEL optional default planner model id
  OPEN_NOTEBOOK_CHAT_MODEL     optional default responder model id
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

ENDPOINT = os.getenv("OPEN_NOTEBOOK_ENDPOINT", "http://open_notebook:5055").rstrip("/")
PASSWORD = os.getenv("OPEN_NOTEBOOK_PASSWORD", "")
DEFAULT_NOTEBOOK_ID = os.getenv("OPEN_NOTEBOOK_NOTEBOOK_ID") or None
DEFAULT_STRATEGY_MODEL = os.getenv("OPEN_NOTEBOOK_STRATEGY_MODEL") or None
DEFAULT_CHAT_MODEL = os.getenv("OPEN_NOTEBOOK_CHAT_MODEL") or None

# 5 minutes — RAG pipelines can be slow; nginx is already set to 300s on this path.
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)


def _auth_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if PASSWORD:
        headers["Authorization"] = f"Bearer {PASSWORD}"
    return headers


async def _create_session(
    client: httpx.AsyncClient,
    notebook_id: str,
    chat_model: str | None,
    title: str = "MCP Chat",
) -> str:
    body = {"notebook_id": notebook_id, "title": title}
    if chat_model:
        body["model_override"] = chat_model
    r = await client.post(
        f"{ENDPOINT}/api/chat/sessions",
        json=body,
        headers=_auth_headers(),
    )
    r.raise_for_status()
    data = r.json()
    sid = data.get("id")
    if not sid:
        raise RuntimeError(f"Session creation returned no id: {data!r}")
    return sid


mcp = FastMCP("open-notebook")


@mcp.tool()
async def chat_rag(
    question: str,
    notebook_id: str | None = None,
    session_id: str | None = None,
    strategy_model: str | None = None,
    chat_model: str | None = None,
) -> dict[str, Any]:
    """Ask a question against an Open Notebook notebook using the full RAG pipeline.

    The pipeline:
      1. condenses the question against prior turns (if a session has history),
      2. plans a multi-search strategy,
      3. runs vector/text search scoped to the notebook,
      4. streams a grounded answer from the responding agent,
      5. emits 3 follow-up suggestions.

    Returns a single consolidated object — the SSE stream is drained internally.

    Args:
      question: the user's natural-language question.
      notebook_id: SurrealDB notebook record id (e.g. "notebook:abc123"). Falls back
        to OPEN_NOTEBOOK_NOTEBOOK_ID env var. Required somewhere.
      session_id: existing chat session to continue. If omitted, a new session is
        created and its id is returned in the response so the caller can keep
        conversation memory across turns.
      strategy_model: override the planning model (SurrealDB model id).
      chat_model: override the responding model (SurrealDB model id).

    Returns: {answer, chunks_used, no_context, followups, session_id}.
    """
    nb_id = notebook_id or DEFAULT_NOTEBOOK_ID
    if not nb_id:
        raise ValueError(
            "notebook_id is required (pass explicitly or set OPEN_NOTEBOOK_NOTEBOOK_ID)"
        )

    strat = strategy_model or DEFAULT_STRATEGY_MODEL
    chat = chat_model or DEFAULT_CHAT_MODEL

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        sid = session_id or await _create_session(client, nb_id, chat)

        body = {
            "session_id": sid,
            "message": question,
            "notebook_id": nb_id,
            "stream": True,
        }
        if strat:
            body["strategy_model"] = strat
        if chat:
            body["model_override"] = chat

        answer_parts: list[str] = []
        final_answer: str | None = None
        chunks_used = 0
        no_context = False
        followups: list[str] = []
        error_msg: str | None = None

        async with client.stream(
            "POST",
            f"{ENDPOINT}/api/chat/rag/execute",
            json=body,
            headers=_auth_headers(),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                try:
                    evt = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                t = evt.get("type")
                if t == "answer_delta":
                    answer_parts.append(evt.get("content", ""))
                elif t == "answer":
                    # Final consolidated answer wins over accumulated deltas.
                    final_answer = evt.get("content", "")
                elif t == "suggestions":
                    items = evt.get("items") or []
                    if isinstance(items, list):
                        followups = [str(x) for x in items]
                elif t == "complete":
                    chunks_used = int(evt.get("chunks_used") or 0)
                    no_context = bool(evt.get("no_context"))
                elif t == "error":
                    error_msg = evt.get("message") or "unknown error"

        if error_msg and not (final_answer or answer_parts):
            raise RuntimeError(f"chat_rag failed: {error_msg}")

        return {
            "answer": final_answer if final_answer is not None else "".join(answer_parts),
            "chunks_used": chunks_used,
            "no_context": no_context,
            "followups": followups,
            "session_id": sid,
        }


@mcp.tool()
async def search_notebook(
    query: str,
    notebook_id: str | None = None,
    search_type: str = "vector",
    limit: int = 10,
) -> dict[str, Any]:
    """Search a notebook's sources and notes without running the full RAG pipeline.

    Args:
      query: search keywords.
      notebook_id: scope to one notebook. Falls back to OPEN_NOTEBOOK_NOTEBOOK_ID;
        pass null/None for a global search across all notebooks.
      search_type: "vector" (semantic, needs an embedding model configured) or
        "text" (keyword fallback).
      limit: max results (1-100).

    Returns: {results, total_count, search_type}.
    """
    if search_type not in ("vector", "text"):
        raise ValueError("search_type must be 'vector' or 'text'")

    body = {
        "query": query,
        "type": search_type,
        "limit": max(1, min(100, limit)),
        "search_sources": True,
        "search_notes": True,
        "notebook_id": notebook_id or DEFAULT_NOTEBOOK_ID,
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(
            f"{ENDPOINT}/api/search",
            json=body,
            headers=_auth_headers(),
        )
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def list_notebooks() -> list[dict[str, Any]]:
    """List all notebooks available in this Open Notebook instance."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.get(
            f"{ENDPOINT}/api/notebooks",
            headers=_auth_headers(),
        )
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def list_sources(
    notebook_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List sources (uploaded documents/URLs) in a notebook.

    Args:
      notebook_id: scope filter. Falls back to OPEN_NOTEBOOK_NOTEBOOK_ID.
      limit: max sources to return (1-100).
    """
    params: dict[str, Any] = {"limit": max(1, min(100, limit))}
    nb = notebook_id or DEFAULT_NOTEBOOK_ID
    if nb:
        params["notebook_id"] = nb
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.get(
            f"{ENDPOINT}/api/sources",
            params=params,
            headers=_auth_headers(),
        )
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    mcp.settings.host = os.getenv("MCP_HOST", "0.0.0.0")
    mcp.settings.port = int(os.getenv("MCP_PORT", "8080"))
    # Deployed on a private LAN/Tailscale segment fronted by nginx — disable the
    # SDK's DNS-rebinding guard so clients can reach the server by IP/hostname.
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    )
    mcp.run(transport="streamable-http")
