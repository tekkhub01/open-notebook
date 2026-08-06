# ADR-008: Retrieval metadata lives on the source, not the chunk

- **Status**: Accepted
- **Date**: 2026-08
- **Related**: migration 27, ADR-006 (migration granularity)

## Context

Corpora prepared specifically for retrieval arrive pre-segmented, with a YAML
frontmatter block identifying the document and its standing — for the EU
machinery corpus that means `celex`, `regulation_number`, `priorita_fonte`
(which act wins when several answer the same question), `natura: dottrina`
(non-binding commentary) and `testo_aggiornato` (whether the text is current).
Answers need those values to cite correctly, to prefer the applicable act, and
to warn when quoting a superseded passage.

Nothing carried them. `chunk_text()` discards the splitter's metadata and returns
`List[str]`, and `source_embedding` is SCHEMAFULL with four fields — `source`,
`order`, `content`, `embedding`. The obvious reading of "propagate metadata onto
the chunk" is to widen that table.

## Decision

Frontmatter is parsed once per document and stored on the **`source`** record, in
a `FLEXIBLE TYPE option<object>` field. Chunk-level search reads it back through
the `source_embedding -> source` join it already performs, and every row of
`fn::vector_search_chunks` carries a `metadata` object (NONE for notes and for
documents without frontmatter).

The whole block is stored, not a whitelist — the frontmatter schema belongs to
the corpus generator. Only fields with retrieval semantics are normalised
(`priorita_fonte` to int, `testo_aggiornato`/`vincolante` to bool, `natura`
lowercased), because those are compared rather than displayed.

## Alternatives considered

- **A `metadata` field on `source_embedding`.** Rejected: these values are
  constant per document, so this copies ~15 fields across every chunk of a
  source — 1,600+ rows for a 10-document corpus — and lets chunks of one source
  drift apart on re-embedding. It buys per-chunk variation that frontmatter, by
  definition, does not have.
- **Per-chunk header metadata (H2 partition, H3 unit) alongside it.** Not needed:
  the markdown splitter runs with `strip_headers=False`, so headings are already
  inside the chunk text and reach the model without a schema change.
- **A fixed whitelist of known fields.** Rejected: adding a field to a corpus
  would require a code change here for no benefit.

## Consequences

- Metadata cannot vary within a source. If a future corpus needs per-unit fields
  that are not already in the chunk text, this is the wrong place and
  `source_embedding` has to grow after all.
- Populated by `embed_source`, which is the only step that always sees the final
  `full_text` and reruns when content changes. A source embedded with
  `embed=false` therefore has no metadata — acceptable, since it is not
  vector-retrievable either, but text search can still reach it.
- Adding a field to the search projection means redefining the whole function:
  SurrealDB has no `ALTER FUNCTION`, so migration 27 restates migration 26's body.
  Expect the same on the next change.
- Retrieval policy built on top: `priorita_fonte` breaks ties **within** a
  similarity band (`SIMILARITY_BAND`, `api/routers/chat_rag.py`) so precedence
  never overrides a clearly better match, and `natura: dottrina` is labelled in
  the prompt rather than filtered out — commentary stays reachable for questions
  about practice and open issues.
- Clients get a **whitelisted projection**, not the block: the `sources` SSE
  event carries `CLIENT_METADATA_FIELDS` per cited chunk, which is what lets a
  citation render as a named act with its standing and an EUR-Lex link. The
  whitelist exists because the stored block is the corpus generator's, so it
  also holds ingestion bookkeeping that has no business on the wire. Adding a
  displayable field means adding it there too.
