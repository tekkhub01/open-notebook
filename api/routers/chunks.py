"""Chunk retrieval endpoint.

Exposes the individual `source_embedding` chunks that the RAG vector search
uses, so that downstream clients can resolve chunk-level citations like
`source:<parent>_chunk_<order>` back to the exact passage of text and (when
possible) locate that passage inside the parent document's full_text.
"""

import re

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from open_notebook.database.repository import repo_query
from open_notebook.domain.notebook import Source

router = APIRouter()


class ChunkResponse(BaseModel):
    id: str
    parent_id: str
    parent_title: str | None = None
    order: int | None = None
    content: str
    offset_start: int | None = None
    offset_end: int | None = None


_CHUNK_SUFFIX_RE = re.compile(r"^(?P<source>source:[A-Za-z0-9_]+)_chunk_(?P<order>\d+)$")


async def _resolve_chunk_record(raw_id: str) -> dict | None:
    """Map an incoming chunk identifier to the source_embedding row.

    Accepts either:
    - `source_embedding:xxx` — direct lookup by record id
    - `source:<parent>_chunk_<n>` — synthesized citation id; resolved by
      (source = $parent, order = $n)
    - `<parent>_chunk_<n>` — same, missing the `source:` prefix
    - bare `<parent>` plus `?order=<n>` form is NOT supported here; pass the
      full synthesized id instead so all callers go through the same path.
    """
    if raw_id.startswith("source_embedding:"):
        rows = await repo_query(
            """
            SELECT id, source.id as parent_id, source.title as parent_title, order, content
            FROM source_embedding
            WHERE id = type::thing($id)
            LIMIT 1
            """,
            {"id": raw_id},
        )
        if isinstance(rows, list) and rows:
            return rows[0]
        return None

    candidate = raw_id if raw_id.startswith("source:") else f"source:{raw_id}"
    match = _CHUNK_SUFFIX_RE.match(candidate)
    if not match:
        return None

    parent_id = match.group("source")
    order = int(match.group("order"))

    rows = await repo_query(
        """
        SELECT id, source.id as parent_id, source.title as parent_title, order, content
        FROM source_embedding
        WHERE source = type::thing($parent) AND order = $order
        LIMIT 1
        """,
        {"parent": parent_id, "order": order},
    )
    if isinstance(rows, list) and rows:
        return rows[0]
    return None


def _locate_offsets(haystack: str | None, needle: str) -> tuple[int | None, int | None]:
    """Best-effort substring search of the chunk inside the parent full_text.

    Returns (start, end) byte offsets when found, (None, None) otherwise. The
    ingest pipeline may have done light normalization (whitespace collapse,
    section-header stripping) so an exact match is not guaranteed. We try
    progressively shorter prefixes — long enough to be unique but resilient
    to trailing edits.
    """
    if not haystack or not needle:
        return None, None
    text = needle.strip()
    if not text:
        return None, None

    direct = haystack.find(text)
    if direct != -1:
        return direct, direct + len(text)

    # Try a long unique-ish prefix (~200 chars). If the ingest reflowed
    # whitespace, this still anchors the chunk to the right region.
    prefix = text[:200]
    idx = haystack.find(prefix)
    if idx != -1:
        return idx, idx + len(text)

    return None, None


@router.get("/chunks/{chunk_id:path}", response_model=ChunkResponse)
async def get_chunk(chunk_id: str):
    """Fetch a single RAG chunk by its citation id."""
    try:
        record = await _resolve_chunk_record(chunk_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Chunk not found: {chunk_id}")

        content = (record.get("content") or "").strip()
        parent_id = str(record.get("parent_id") or "")
        order_raw = record.get("order")
        try:
            order = int(order_raw) if order_raw is not None else None
        except (TypeError, ValueError):
            order = None

        offset_start = offset_end = None
        if parent_id and content:
            try:
                src = await Source.get(parent_id)
                if src and src.full_text:
                    offset_start, offset_end = _locate_offsets(src.full_text, content)
            except Exception as e:
                logger.warning(f"Could not load parent source {parent_id}: {e}")

        return ChunkResponse(
            id=str(record.get("id") or chunk_id),
            parent_id=parent_id,
            parent_title=record.get("parent_title") or None,
            order=order,
            content=content,
            offset_start=offset_start,
            offset_end=offset_end,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching chunk {chunk_id}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching chunk")
