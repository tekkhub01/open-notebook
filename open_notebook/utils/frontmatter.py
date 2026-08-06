"""YAML frontmatter extraction for ingested markdown sources.

Corpora prepared for retrieval (the EU machinery/AI/cyber-resilience acts, for
example) carry a YAML frontmatter block with the act's identity and its place in
the corpus. Those fields are constant for the whole document, so they are stored
once on the `source` record rather than copied onto every chunk — the chunk
search function reads them back through its existing `source_embedding -> source`
join.

The whole block is kept, not a fixed whitelist: adding a field to the corpus
generator should not require a code change here. A handful of fields carry
retrieval semantics and are normalised so they can be compared and ordered
reliably; see NORMALISED_FIELDS.
"""

import re
from datetime import date, datetime
from typing import Any, Dict, Optional

import yaml
from loguru import logger

# A frontmatter block is small by nature. Anything larger is either not
# frontmatter or not something we want to hand to the YAML parser.
MAX_FRONTMATTER_CHARS = 16_000

# Fields whose value drives retrieval behaviour rather than mere display, and
# which are therefore coerced to a predictable type:
#   priorita_fonte   int   — precedence when several acts answer the same question
#   testo_aggiornato bool  — false means the text is not current on some points,
#                            which the answer is expected to flag
#   natura           str   — "dottrina" marks non-binding commentary
#   vincolante       bool  — binding force
NORMALISED_FIELDS = ("priorita_fonte", "testo_aggiornato", "natura", "vincolante")

_FRONTMATTER_RE = re.compile(r"\A﻿?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)

_TRUE = {"true", "yes", "si", "sì", "1"}
_FALSE = {"false", "no", "0"}


def _to_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
    return value


def _to_int(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return value


def _jsonable(value: Any) -> Any:
    """Make a parsed YAML value safe to store.

    An unquoted `consolidato_al: 2023-07-04` is parsed by YAML into a
    `datetime.date`, which is not a value SurrealDB accepts inside a plain
    object — dates are rendered as ISO strings instead. Nested containers are
    walked because list- and mapping-valued fields (`modifiche_integrate`,
    `keywords`) can hold them too.
    """
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _normalise(data: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce the retrieval-bearing fields; leave everything else verbatim."""
    if "priorita_fonte" in data:
        data["priorita_fonte"] = _to_int(data["priorita_fonte"])
    for field in ("testo_aggiornato", "vincolante"):
        if field in data:
            data[field] = _to_bool(data[field])
    natura = data.get("natura")
    if isinstance(natura, str):
        data["natura"] = natura.strip().lower()
    return data


def extract_frontmatter(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the parsed YAML frontmatter of `text`, or None.

    None means "no usable frontmatter" for every failure mode — absent block,
    malformed YAML, or a block that parses to something other than a mapping.
    Ingestion must not fail because a document has no frontmatter, so nothing
    here raises.
    """
    if not text:
        return None

    match = _FRONTMATTER_RE.match(text)
    if not match:
        return None

    block = match.group(1)
    if len(block) > MAX_FRONTMATTER_CHARS:
        logger.warning(
            f"Frontmatter block is {len(block)} chars (max {MAX_FRONTMATTER_CHARS}); "
            "treating the document as having none."
        )
        return None

    try:
        parsed = yaml.safe_load(block)
    except yaml.YAMLError as e:
        logger.warning(f"Could not parse YAML frontmatter, ignoring it: {e}")
        return None

    if not isinstance(parsed, dict) or not parsed:
        return None

    # SurrealDB object keys must be strings; a YAML mapping can key on ints or
    # dates, which would fail on insert.
    data = {str(k): _jsonable(v) for k, v in parsed.items()}
    return _normalise(data)
