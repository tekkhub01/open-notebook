"""Source precedence, provenance labelling and citation payload in the RAG path.

Covers the behaviours the legal corpus depends on: `priorita_fonte` breaking
ties between acts that answer the same question, the provenance line that lets
an answer tell binding text from commentary and current text from stale, the
context budget that keeps the tail of a broad search out of the prompt, and the
metadata the client needs to render a citation as a named act.
"""

from api.routers.chat_rag import (
    CLIENT_METADATA_FIELDS,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHUNKS,
    MAX_PROVENANCE_DETAIL_CHARS,
    SIMILARITY_BAND,
    UNRANKED_PRIORITY,
    WEAK_RETRIEVAL_SCORE,
    _best_score,
    _client_metadata,
    _client_sources,
    _display_title,
    _provenance_line,
    _rank_chunks,
    _select_context_chunks,
)


def _chunk(chunk_id, score, **metadata):
    return {"id": chunk_id, "score": score, "metadata": metadata or {}}


class TestRankChunks:
    def test_precedence_breaks_ties_within_the_band(self):
        """The machinery regulation outranks an act alongside it when the
        retriever cannot separate them."""
        ranked = _rank_chunks(
            [
                _chunk("affiancato", 0.72, priorita_fonte=2),
                _chunk("macchine", 0.70, priorita_fonte=1),
            ]
        )

        assert [c["id"] for c in ranked] == ["macchine", "affiancato"]

    def test_relevance_still_wins_across_bands(self):
        """Precedence must not promote a marginal hit over a clearly better one."""
        ranked = _rank_chunks(
            [
                _chunk("affiancato", 0.90, priorita_fonte=2),
                _chunk("macchine", 0.40, priorita_fonte=1),
            ]
        )

        assert [c["id"] for c in ranked] == ["affiancato", "macchine"]

    def test_full_precedence_order_within_one_band(self):
        ranked = _rank_chunks(
            [
                _chunk("dec768", 0.71, priorita_fonte=3),
                _chunk("macchine", 0.70, priorita_fonte=1),
                _chunk("affiancato", 0.72, priorita_fonte=2),
            ]
        )

        assert [c["id"] for c in ranked] == ["macchine", "affiancato", "dec768"]

    def test_unranked_content_sorts_after_ranked_in_the_same_band(self):
        ranked = _rank_chunks(
            [
                _chunk("nota_utente", 0.72),
                _chunk("dec768", 0.70, priorita_fonte=3),
            ]
        )

        assert [c["id"] for c in ranked] == ["dec768", "nota_utente"]

    def test_score_orders_within_equal_precedence(self):
        # Same band and same precedence, so the raw score is the only thing left
        # to separate them.
        ranked = _rank_chunks(
            [
                _chunk("b", 0.705, priorita_fonte=1),
                _chunk("a", 0.72, priorita_fonte=1),
            ]
        )

        assert [c["id"] for c in ranked] == ["a", "b"]

    def test_band_is_rounded_not_truncated(self):
        """0.70 / 0.05 is 13.999999999999998 in binary floating point; truncating
        would put 0.70 and 0.72 in different bands and lose the tie-break."""
        from api.routers.chat_rag import _similarity_band

        assert _similarity_band(0.70) == _similarity_band(0.72)

    def test_missing_and_malformed_precedence_is_treated_as_unranked(self):
        for bad in ({}, {"priorita_fonte": None}, {"priorita_fonte": "1"},
                    {"priorita_fonte": True}):
            chunk = {"id": "x", "score": 0.5, "metadata": bad}
            # Ranks below an explicit precedence at the same score.
            ranked = _rank_chunks([chunk, _chunk("ranked", 0.5, priorita_fonte=9)])
            assert [c["id"] for c in ranked] == ["ranked", "x"], bad

    def test_survives_absent_metadata_and_score_keys(self):
        ranked = _rank_chunks([{"id": "bare"}, _chunk("scored", 0.9)])

        assert [c["id"] for c in ranked] == ["scored", "bare"]

    def test_band_constant_is_narrow_enough_to_stay_a_tie_break(self):
        # Guards the constant itself: a wide band would let precedence override
        # real relevance differences rather than merely break ties.
        assert 0 < SIMILARITY_BAND <= 0.1
        assert UNRANKED_PRIORITY > 3


class TestProvenanceLine:
    def test_carries_citation_identifiers(self):
        line = _provenance_line(
            {"celex": "32014L0035", "regulation_number": "2014/35/UE"}
        )

        assert "32014L0035" in line
        assert "2014/35/UE" in line

    def test_marks_doctrine_as_non_binding(self):
        line = _provenance_line(
            {"celex": "n/a", "natura": "dottrina", "vincolante": False}
        )

        assert "DOCTRINE" in line
        assert "NOT BINDING" in line
        assert "applicable provision" in line

    def test_binding_act_is_not_marked_as_doctrine(self):
        line = _provenance_line({"celex": "32023R1230", "testo_aggiornato": True})

        assert "DOCTRINE" not in line
        assert "NOT CURRENT" not in line

    def test_flags_a_stale_text_with_its_pending_amendments(self):
        line = _provenance_line(
            {
                "celex": "32023R1230",
                "testo_aggiornato": False,
                "modifiche_da_integrare": "rettifica 4 luglio 2023",
            }
        )

        assert "NOT CURRENT" in line
        assert "rettifica 4 luglio 2023" in line

    def test_flags_a_stale_text_whose_amendments_are_a_list(self):
        """The shape the corpus actually uses: the AI Act carries its pending
        amendments as a YAML list, not a string."""
        line = _provenance_line(
            {
                "celex": "32024R1689",
                "testo_aggiornato": False,
                "modifiche_da_integrare": [
                    "Regolamento (UE) 2026/1744 (GU L, 2026/1744, 24.7.2026)"
                ],
            }
        )

        assert "NOT CURRENT" in line
        assert "Regolamento (UE) 2026/1744" in line

    def test_lists_several_pending_amendments(self):
        line = _provenance_line(
            {
                "testo_aggiornato": False,
                "modifiche_da_integrare": ["rettifica 2023", "regolamento 2026/1744"],
            }
        )

        assert "rettifica 2023" in line
        assert "regolamento 2026/1744" in line

    def test_long_amendment_lists_are_truncated(self):
        """The line is repeated in front of every chunk of the source, so the
        detail must not grow without bound."""
        line = _provenance_line(
            {
                "testo_aggiornato": False,
                "modifiche_da_integrare": [f"Regolamento (UE) 2026/{n}" for n in range(40)],
            }
        )

        assert "NOT CURRENT" in line
        assert "…" in line
        assert len(line) < MAX_PROVENANCE_DETAIL_CHARS + 150

    def test_stale_text_without_detail_still_flags(self):
        for metadata in (
            {"testo_aggiornato": False},
            {"testo_aggiornato": False, "modifiche_da_integrare": []},
            {"testo_aggiornato": False, "modifiche_da_integrare": None},
            {"testo_aggiornato": False, "modifiche_da_integrare": ""},
        ):
            line = _provenance_line(metadata)

            assert "NOT CURRENT" in line, metadata
            # No empty parenthetical where the detail would have gone.
            assert "()" not in line, metadata

    def test_empty_metadata_produces_no_line(self):
        assert _provenance_line(None) == ""
        assert _provenance_line({}) == ""
        # Fields we do not surface must not produce a stray empty line.
        assert _provenance_line({"language": "IT"}) == ""


def _sized_chunk(chunk_id, score, chars, **metadata):
    return {
        "id": chunk_id,
        "score": score,
        "content": "x" * chars,
        "metadata": metadata or {},
    }


class TestSelectContextChunks:
    def test_keeps_everything_when_the_search_is_small(self):
        chunks = [_sized_chunk(f"c{n}", 0.8, 100) for n in range(5)]

        assert _select_context_chunks(chunks) == chunks

    def test_caps_the_number_of_chunks(self):
        chunks = [_sized_chunk(f"c{n}", 0.8, 10) for n in range(MAX_CONTEXT_CHUNKS + 12)]

        assert len(_select_context_chunks(chunks)) == MAX_CONTEXT_CHUNKS

    def test_caps_the_character_budget(self):
        # Three of these fit; the fourth would overrun.
        size = MAX_CONTEXT_CHARS // 3
        chunks = [_sized_chunk(f"c{n}", 0.8, size) for n in range(6)]

        selected = _select_context_chunks(chunks)

        assert len(selected) == 3
        assert sum(len(c["content"]) for c in selected) <= MAX_CONTEXT_CHARS

    def test_truncation_takes_the_head_of_the_ranked_list(self):
        """Dropping from the tail is what makes the cut safe: the answer cites
        the same passages it would have cited from the untruncated list."""
        chunks = [_sized_chunk(f"c{n}", 0.9 - n / 100, 10) for n in range(30)]

        selected = _select_context_chunks(chunks)

        assert [c["id"] for c in selected] == [c["id"] for c in chunks[:MAX_CONTEXT_CHUNKS]]

    def test_one_oversized_chunk_is_still_kept(self):
        """Better a single overlong passage than an empty context: the budget
        only ever drops chunks that come after something already selected."""
        chunks = [_sized_chunk("huge", 0.9, MAX_CONTEXT_CHARS * 2)]

        assert _select_context_chunks(chunks) == chunks

    def test_survives_chunks_without_content(self):
        chunks = [{"id": "bare", "score": 0.9}]

        assert _select_context_chunks(chunks) == chunks


class TestBestScore:
    def test_reads_the_highest_score_not_the_first_row(self):
        """Ranking puts the most authoritative chunk of the best band on top,
        which is not necessarily the closest match."""
        ranked = _rank_chunks(
            [_chunk("affiancato", 0.72, priorita_fonte=2), _chunk("macchine", 0.70, priorita_fonte=1)]
        )

        assert ranked[0]["id"] == "macchine"
        assert _best_score(ranked) == 0.72

    def test_missing_and_malformed_scores_do_not_raise(self):
        assert _best_score([{"id": "bare"}, {"id": "bad", "score": "abc"}]) == 0.0

    def test_empty_set_scores_zero(self):
        assert _best_score([]) == 0.0

    def test_threshold_sits_between_the_floor_and_real_answers(self):
        # The floor is min_similarity=0.5 in execute_multi_search, so a threshold
        # at or below it could never fire. The upper guard is measured, not
        # arbitrary: the worst-scoring answerable question in the trace history
        # scored 0.604, and a threshold above that labels good answers as thin.
        assert 0.5 < WEAK_RETRIEVAL_SCORE < 0.604


class TestClientMetadata:
    def test_passes_through_the_fields_the_ui_renders(self):
        metadata = _client_metadata(
            {
                "title": "Regolamento (UE) 2023/1230 — Regolamento macchine",
                "celex": "32023R1230",
                "eli": "http://data.europa.eu/eli/reg/2023/1230/oj",
                "priorita_fonte": 1,
                "testo_aggiornato": True,
            }
        )

        assert metadata["celex"] == "32023R1230"
        assert metadata["eli"].endswith("/oj")
        assert metadata["priorita_fonte"] == 1
        assert metadata["testo_aggiornato"] is True

    def test_drops_ingestion_bookkeeping(self):
        metadata = _client_metadata(
            {
                "celex": "32023R1230",
                "pipeline": "reg-macchine-rag v2.1",
                "fonte_pdf": "Regolamento.pdf",
                "unita_recuperabili": 206,
                "generato": "2026-08-06",
            }
        )

        assert set(metadata) == {"celex"}

    def test_keeps_false_flags_that_carry_a_warning(self):
        """`vincolante: false` is the whole point of the field — dropping falsey
        values would silently unlabel every non-binding document."""
        metadata = _client_metadata({"vincolante": False, "testo_aggiornato": False})

        assert metadata["vincolante"] is False
        assert metadata["testo_aggiornato"] is False

    def test_flattens_the_amendment_list_to_one_line(self):
        metadata = _client_metadata(
            {"modifiche_da_integrare": ["rettifica 2023", "regolamento 2026/1744"]}
        )

        assert metadata["modifiche_da_integrare"] == "rettifica 2023; regolamento 2026/1744"

    def test_truncates_a_long_amendment_list(self):
        metadata = _client_metadata(
            {"modifiche_da_integrare": [f"Regolamento (UE) 2026/{n}" for n in range(40)]}
        )

        assert len(metadata["modifiche_da_integrare"]) <= MAX_PROVENANCE_DETAIL_CHARS

    def test_empty_and_missing_metadata_produce_an_empty_object(self):
        assert _client_metadata(None) == {}
        assert _client_metadata({}) == {}
        assert _client_metadata({"celex": "", "natura": None, "keywords": []}) == {}

    def test_whitelist_covers_what_the_badges_are_built_from(self):
        for field in ("natura", "vincolante", "testo_aggiornato", "avvertenza", "eli"):
            assert field in CLIENT_METADATA_FIELDS


class TestDisplayTitle:
    def test_prefers_the_label_sized_title(self):
        chunk = {
            "title": "01-SCHEDA-SINTESI.md",
            "metadata": {
                "titolo_breve": "Scheda di sintesi sul regolamento macchine",
                "title": "Scheda di sintesi — domande frequenti sul regolamento macchine",
            },
        }

        assert _display_title(chunk) == "Scheda di sintesi sul regolamento macchine"

    def test_falls_back_to_the_official_title(self):
        chunk = {
            "title": "reg-2023-1230-macchine.md",
            "metadata": {"title": "Regolamento (UE) 2023/1230 — Regolamento macchine"},
        }

        assert _display_title(chunk) == "Regolamento (UE) 2023/1230 — Regolamento macchine"

    def test_falls_back_to_the_filename_without_its_extension(self):
        chunk = {"title": "Norme_generali_CSI_Brescia_2025-2026.md", "metadata": {}}

        assert _display_title(chunk) == "Norme_generali_CSI_Brescia_2025-2026"

    def test_falls_back_to_the_citation_id_when_there_is_no_title(self):
        assert _display_title({"id": "source:abc_chunk_2"}) == "source:abc_chunk_2"


class TestClientSources:
    def test_describes_a_citation_without_shipping_its_text(self):
        items = _client_sources(
            [
                {
                    "id": "source:abc_chunk_4",
                    "parent_id": "source:abc",
                    "title": "reg-2023-1230-macchine.md",
                    "order": 4,
                    "score": 0.812345,
                    "content": "un passaggio lungo",
                    "metadata": {"title": "Regolamento (UE) 2023/1230", "celex": "32023R1230"},
                }
            ]
        )

        assert items == [
            {
                "id": "source:abc_chunk_4",
                "parent_id": "source:abc",
                "title": "Regolamento (UE) 2023/1230",
                "order": 4,
                "score": 0.812,
                "metadata": {"title": "Regolamento (UE) 2023/1230", "celex": "32023R1230"},
            }
        ]

    def test_keyed_by_the_citation_id_the_model_is_told_to_quote(self):
        """The client resolves `[source:abc_chunk_4]` in the answer text against
        these ids; anything else would leave every citation unresolved."""
        items = _client_sources([_sized_chunk("source:abc_chunk_4", 0.8, 10)])

        assert items[0]["id"] == "source:abc_chunk_4"

    def test_survives_a_chunk_with_no_metadata_or_score(self):
        items = _client_sources([{"id": "note:xyz", "title": "Appunto"}])

        assert items[0]["metadata"] == {}
        assert items[0]["score"] == 0.0
        assert items[0]["title"] == "Appunto"
