"""Tests for YAML frontmatter extraction on ingested markdown sources."""

from open_notebook.utils.frontmatter import extract_frontmatter

# Trimmed from a real corpus file (dir-2014-35-bassa-tensione.md): keeps the
# shapes that matter — a YAML comment inside the block, an unquoted int and
# bool, a quoted date, a list, and a field whose value contains a colon.
REAL_FRONTMATTER = '''---
title: "Direttiva 2014/35/UE — Direttiva bassa tensione (LVD)"
id: "dir-2014-35-bassa-tensione"
regulation_number: "2014/35/UE"
type: "Direttiva UE"
date_adoption: "2014-02-26"
celex: "32014L0035"
eli: "http://data.europa.eu/eli/dir/2014/35/oj"
abrogates: ["Direttiva 2006/95/CE"]

# collocazione nel sistema delle fonti e nel corpus
rango: "diritto derivato dell'Unione"
efficacia: "vincola gli Stati membri quanto al risultato: si applica agli operatori"
ruolo_nel_corpus: "normativa integrativa"
priorita_fonte: 2
consolidato_al: "nessuna modifica integrata"
testo_aggiornato: true

keywords:
 - materiale elettrico
 - bassa tensione
---

<!-- TIPO: Direttiva UE | CELEX: 32014L0035 -->

# Direttiva 2014/35/UE del Parlamento europeo e del Consiglio
'''


def test_extracts_real_corpus_frontmatter():
    data = extract_frontmatter(REAL_FRONTMATTER)

    assert data is not None
    assert data["celex"] == "32014L0035"
    assert data["regulation_number"] == "2014/35/UE"
    assert data["eli"] == "http://data.europa.eu/eli/dir/2014/35/oj"
    assert data["abrogates"] == ["Direttiva 2006/95/CE"]
    assert data["keywords"] == ["materiale elettrico", "bassa tensione"]
    # The YAML comment is not a field.
    assert not any(k.startswith("#") for k in data)
    # A colon inside a quoted value must not split the field.
    assert data["efficacia"].startswith("vincola gli Stati membri")


def test_retrieval_bearing_fields_keep_their_type():
    data = extract_frontmatter(REAL_FRONTMATTER)

    assert data is not None
    assert data["priorita_fonte"] == 2
    assert isinstance(data["priorita_fonte"], int)
    assert data["testo_aggiornato"] is True


def test_body_is_not_consumed_past_the_closing_fence():
    """The non-greedy match must stop at the first `---`, not at a later one."""
    text = "---\ncelex: X\n---\n\nbody\n\n---\n\nmore body\n"
    data = extract_frontmatter(text)

    assert data == {"celex": "X"}


def test_normalises_quoted_scalars():
    text = '---\npriorita_fonte: "3"\ntesto_aggiornato: "false"\nnatura: "Dottrina"\nvincolante: "no"\n---\nbody\n'
    data = extract_frontmatter(text)

    assert data is not None
    assert data["priorita_fonte"] == 3
    assert data["testo_aggiornato"] is False
    assert data["natura"] == "dottrina"
    assert data["vincolante"] is False


def test_dates_become_strings():
    """An unquoted YAML date parses to datetime.date, which SurrealDB rejects
    inside a plain object."""
    data = extract_frontmatter("---\nconsolidato_al: 2023-07-04\n---\nbody\n")

    assert data == {"consolidato_al": "2023-07-04"}


def test_returns_none_when_there_is_no_usable_frontmatter():
    # Ingestion must never fail over frontmatter, so every bad shape is None.
    assert extract_frontmatter(None) is None
    assert extract_frontmatter("") is None
    assert extract_frontmatter("# Just a heading\n\nSome text.\n") is None
    # Malformed YAML.
    assert extract_frontmatter("---\ncelex: [unclosed\n---\nbody\n") is None
    # Parses, but not a mapping.
    assert extract_frontmatter("---\n- a\n- b\n---\nbody\n") is None
    # A horizontal rule mid-document is not frontmatter.
    assert extract_frontmatter("Some text.\n\n---\n\nMore text.\n") is None


def test_oversized_block_is_ignored():
    from open_notebook.utils.frontmatter import MAX_FRONTMATTER_CHARS

    huge = "---\n" + ("k: v\n" * (MAX_FRONTMATTER_CHARS // 4)) + "---\nbody\n"

    assert extract_frontmatter(huge) is None
