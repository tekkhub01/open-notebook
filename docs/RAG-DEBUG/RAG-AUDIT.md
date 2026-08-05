# RAG Quality Audit — Notebook "Regolamento Macchine UE"

Audit of `chat_rag` answer quality against ground-truth source documents, using the `rag_trace` SurrealDB table as the trace record.

- **Notebook**: `notebook:ui3cujj5t2ti6ykjdn1d`
- **Sources audited**:
  - `source:94hk65lvyulpk1rcdlp6` → `Regolamento_UE_2023_1230_Macchine_RAG.md` (385 KB, the regulation)
  - `source:gj7nea4ybjt9jvdgdsf2` → `Regolamento (UE) 2023 1230 sulle macchine e criticità applicative.md` (29 KB, criticality analysis)
  - `source_insight:fg69c7aucauk1jrfvy0w` → dense summary of the regulation (3.7 KB)
- **Audit date**: 2026-04-27

---

## Findings — 4 traces audited

### Summary table

| Trace ID | Question | Chunks | Top score | Total ms | Faithfulness | Grade |
|---|---|---|---|---|---|---|
| `rag_trace:gm2pi59lgw7rx8ceu0c3` | Software responsibility | 2 | 0.666 | 21,474 | ✅ 0 hallucinations | A |
| `rag_trace:e6skcc76xrcvc34tjxdx` | Definizione importatore (multi-turn) | 2 | 0.715 | 19,569 | ✅ 0 hallucinations | A+ |
| `rag_trace:gq082010zz01pfxbz4hu` | Requisiti essenziali | 2 | 0.624 | 17,611 | ✅ 0 hallucinations | A |
| `rag_trace:4cnvb59stmwq3bn3ywwq` | Macchine alto rischio | 3 | 0.782 | 22,751 | ✅ 0 hallucinations | A+ |

**Overall verdict**: 0 hallucinations across 4 answers. Every cited fact, article number, annex reference, RESS section ID, and module name is verifiable in the source documents. Citation attribution between the two sources is correct in every spot-check.

### Per-trace verification

#### `gm2pi` — Software responsibility

| Claim | Source | Line(s) |
|---|---|---|
| Software (funzione di sicurezza, immesso indipendentemente) = componente di sicurezza | Reg, Considerando 19 | 141 |
| Codice sorgente / logica programmazione su richiesta motivata autorità | Reg, Art. 10 §6 / Art. 11 / Allegato IV | 904, 1009, 1159, 4962 |
| EN IEC 62443 attributed to `[source:gj7n…]` | Criticità | 68, 113, 124 |
| RESS 1.1.9 "Protezione dall'alterazione" | Reg, Allegato III | 2799 |
| RESS 1.2.1 + retention "cinque anni" | Reg, Allegato III §1.2.1 | 2826, 2849 |

Citation attribution is correct: EN IEC 62443 has 0 hits in the regulation but is correctly attributed to the criticality doc.

#### `e6skc` — Definizione importatore

| Claim | Source | Line(s) |
|---|---|---|
| Definizione (paese terzo) | Reg, Art. 3 punto 20 | 629–630 |
| Art. 13 = importatori macchine/prodotti correlati | Reg | 1110 |
| Art. 14 = importatori quasi-macchine | Reg | 1175 |
| Documentazione tecnica = Allegato IV | Reg | 4904 |
| RESS = Allegato III | Reg | 2551 |
| Lingua facilmente comprensibile | Reg, Art. 13 §3 | 1129–1130, 963 |
| Informa fabbricante + autorità se non conforme | Reg, Art. 13 §7 | 1123 |

This trace also exercises the condense step: `"vediamo la definizione di importatore"` was correctly rewritten to `"Qual è la definizione di importatore secondo il Regolamento (UE) 2023/1230 relativo alle macchine?"` using prior-turn context.

#### `gq082` — Requisiti essenziali

| Claim | Source | Line(s) |
|---|---|---|
| RESS in Allegato III | Reg | 2551 |
| Allegato III — sei capi | Reg, correlation table | 5844–5854 |
| RESS 1.1.9 "Protezione dall'alterazione" | Reg | 2799 |
| RESS 1.2.1 + cinque anni | Reg | 2826, 2849 |
| Apprendimento automatico / componenti AI | Reg, Considerando 55 + Allegato I | 276, 2393, 2396, 2537 |
| Specifiche comuni via atti di esecuzione | Reg, Considerando 45 | 240 |
| Punto 1.1.2 integrazione sicurezza | Reg | 5844 |

⚠️ **Minor completeness gap (not an error)**: Allegato III also has `PARTE A` (Definizioni) and `PARTE B` (Principi generali) before the 6 capi. The answer's "articolato in sei capi" is correct for the technical sections but omits the Parte A/B preamble.

#### `4cnvb` — Macchine alto rischio

| Claim | Source | Line(s) |
|---|---|---|
| Allegato I = categorie con procedure specifiche (rif. Art. 25 §§2–3) | Reg | 2375–2378 |
| Parte A → organismo notificato obbligatorio (Art. 25 §2) | Criticità + Reg | crit:83; reg:1533 |
| Parte B → controllo interno se norme armonizzate (Art. 25 §3) | Criticità + Reg | crit:84; reg:1545 |
| Moduli B (allegato VII) / C (VIII) / H (IX) / G (X) | Reg, Art. 25 | 1535–1554 |
| Approccio dinamico, supera liste statiche, dati incidenti | Criticità | 82 |
| AI safety-related rientra in Parte A | Criticità + Reg | crit:86; reg:2393 |
| `[source_insight:fg69c…]` correctly distinguished from `[source:…]` | Dense summary | — |

The exact module-to-annex mapping (B→VII, C→VIII, H→IX, G→X) was reproduced perfectly from Art. 25 §§2–3.

---

## Pipeline observations

### Strengths
- **Citation discipline**: all 4 use `[source:<id>]` per-claim, not just at the end. Correctly distinguishes `[source_insight:…]` from `[source:…]`.
- **Source attribution** between the two documents is accurate even for content that appears in only one (EN IEC 62443, dynamic-approach commentary).
- **Condense step** works (see `e6skc`).
- **Followups** (3 per trace, 12/12 examined) are all on-topic and drill into specific claims from the answer.

### Weaknesses
- **Strategy over-plans for small corpora**: each turn generates 3–5 search queries, but the notebook has only 2 documents. After dedup, 2–3 chunks come back. The extra searches mostly burn `search_ms`.
- **Retrieval latency dominates total time** (7–12s of 17–23s total). Largest opportunity for optimization.
- **`source_insight` under-retrieved**: only `4cnvb` surfaced the dense summary (score 0.51, 3rd place). The other 3 questions would have benefitted but the insight was filtered by top-k.
- **Trace previews are too short for audit**: `chunks_retrieved[].content_preview` is 500 chars, which is not enough to verify per-claim faithfulness without re-fetching the source. This made the initial audit pass appear to flag hallucinations that weren't actually hallucinations.

### Recommendations
1. **Cap strategy searches by corpus size** — for a 2-source notebook, 1–2 queries is enough. Saves 4–6s per turn.
2. **Persist full chunk content** (or 2–3k chars, or byte offsets) in `rag_trace.chunks_retrieved` so faithfulness can be audited from the trace alone.
3. **Boost `source_insight` records in retrieval** — they're cheap, summary-shaped, and currently under-surfaced.
4. **Add an automated faithfulness score** at write time (LLM judge over full chunks) — would turn this manual audit into a one-shot signal.
5. **Split `search_ms`** into embed / vector-search / merge / rerank components to find the actual bottleneck within retrieval.

---

## How to run a future audit

### Prerequisites
- SurrealDB running at the URL in `.env` (default `ws://localhost:8000/rpc`)
- `uv` installed; project venv synced (`uv sync`)
- Ground-truth source markdown in `docs/RAG-DEBUG/` (or fetched from DB — see step 2)

### Step 1 — Pick traces to audit

Get recent traces for a notebook:

```bash
set -a && . ./.env && set +a && uv run python -c "
import asyncio, json
from open_notebook.database.repository import repo_query
async def main():
    r = await repo_query('SELECT id, question, created FROM rag_trace WHERE notebook_id = notebook:ui3cujj5t2ti6ykjdn1d ORDER BY created DESC LIMIT 20;')
    print(json.dumps(r, default=str, indent=2, ensure_ascii=False))
asyncio.run(main())
"
```

### Step 2 — Fetch the trace records

```bash
set -a && . ./.env && set +a && uv run python -c "
import asyncio, json
from open_notebook.database.repository import repo_query
async def main():
    ids = ['gm2pi59lgw7rx8ceu0c3', 'e6skcc76xrcvc34tjxdx']  # replace
    for tid in ids:
        r = await repo_query(f'SELECT * FROM rag_trace:{tid};')
        print(f'===== rag_trace:{tid} =====')
        print(json.dumps(r, default=str, indent=2, ensure_ascii=False))
asyncio.run(main())
"
```

Each trace contains:
- `question`, `condensed_question` (if multi-turn)
- `strategy_reasoning`, `strategy_searches[]` (planning step)
- `chunks_retrieved[]` (with `content_preview`, score, source/parent IDs)
- `final_answer`
- `followups[]`
- Timing fields: `planning_ms`, `condense_ms`, `search_ms`, `answer_ms`, `followups_ms`, `total_ms`

### Step 3 — Make sure ground-truth files are available

If sources are in `docs/RAG-DEBUG/` use them directly. To dump from the DB:

```bash
set -a && . ./.env && set +a && uv run python -c "
import asyncio, os
from open_notebook.database.repository import repo_query
async def main():
    out = '/tmp/rag_audit'
    os.makedirs(out, exist_ok=True)
    for sid, fname in [('source:94hk65lvyulpk1rcdlp6', 'reg.md'),
                       ('source:gj7nea4ybjt9jvdgdsf2', 'crit.md'),
                       ('source_insight:fg69c7aucauk1jrfvy0w', 'summary.md')]:
        r = await repo_query(f'SELECT * FROM {sid};')
        if r:
            text = r[0].get('full_text') or r[0].get('content') or ''
            with open(f'{out}/{fname}', 'w') as f:
                f.write(text)
            print(f'{fname}: {len(text)} chars')
asyncio.run(main())
"
```

### Step 4 — Verify each claim

For every cited fact in `final_answer`, grep for it in the source files. Examples:

```bash
cd "docs/RAG-DEBUG"
REG="Regolamento_UE_2023_1230_Macchine_RAG.md"
CRIT="Regolamento (UE) 2023 1230 sulle macchine e criticità applicative.md"

# Article references
grep -nE '^### Articolo (13|14) —' "$REG"

# RESS section IDs
grep -nE '^1\.1\.9|^1\.2\.1' "$REG"

# Annexes
grep -nE 'ALLEGATO (I|III|IV)\b' "$REG"

# Cross-source attribution check (term should appear in cited source only)
grep -cE 'EN IEC 62443' "$REG"   # expect 0
grep -nE 'EN IEC 62443' "$CRIT"  # expect hits
```

### Step 5 — Score each trace

For each trace, score along three axes:

- **Faithfulness**: every cited claim verifiable in the cited source? (Y/N + list any unsupported claims)
- **Completeness**: does the answer cover the question fully, or skip relevant material from the corpus?
- **Citation precision**: are `[source:<id>]` IDs the *correct* one for each claim? (Cross-source mix-ups are the most common defect.)

Mark traces as A+ / A / B / C and record in this file's summary table.

### Step 6 — Look for retrieval/strategy issues

Independent of answer quality, check:

- `chunks_count` vs. `len(strategy_searches)` — large gap = over-planning.
- Top score < 0.65 — possible chunking/embedding mismatch for the question's terminology.
- `source_insight` records not surfaced — top-k or threshold issue.
- `search_ms` > 8s — retrieval bottleneck candidate.

---

## Source layout reference

```
docs/RAG-DEBUG/
├── RAG-AUDIT.md  ← this file
├── Regolamento_UE_2023_1230_Macchine_RAG.md         (source:94hk65lvyulpk1rcdlp6)
└── Regolamento (UE) 2023 1230 sulle macchine e criticità applicative.md  (source:gj7nea4ybjt9jvdgdsf2)
```

The dense summary (`source_insight:fg69c7aucauk1jrfvy0w`) lives only in the DB; dump via the snippet in Step 3 if needed.

---

## Changelog

- **2026-04-27** — Initial audit of 4 traces on notebook "Regolamento Macchine UE". 0 hallucinations. Pipeline weakness identified: 500-char `content_preview` is too short for trace-only audits.
