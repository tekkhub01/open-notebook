# RAG Quality Audit — Notebook "Calcio-a-7" (CSI Brescia)

Audit of `chat_rag` answer quality against ground-truth source documents, using the `rag_trace` SurrealDB table as the trace record. Methodology follows [docs/RAG-DEBUG/RAG-AUDIT.md](../RAG-DEBUG/RAG-AUDIT.md).

- **Notebook**: `notebook:wcey1gczvhr6vbdxyyn5` ("Calcio-a-7")
- **Sources audited**:
  - `source:o96lyfb7pgus91ukip88` → `Regolamento_Calcio_a_7_CSI_Brescia_2025-2026.md` (main CA7 regulation)
  - `source:mhdy24qn3g0qjkl7qjfs` → `Norme_generali_CSI_Brescia_2025-2026.md` (general CSI norms)
  - `source:tnn1sfdhe5y5ck30611c` → `Giustizia-sportiva_CSI_Brescia_2025-2026.md` (sport justice / disciplinary)
  - `source:l9objoq2lg90ldjpug1r` → `Integrazione_CA7_OVER_2025-2026.md` (Over 35/40 integration)
  - Plus 5 further category integrations (Giovanile, U15 Femm., Under, Open M. SerieA/B, Open Femm.)
- **Cross-notebook trace**: `rag_trace:4m8mat066ko4l01dvxct` is on `notebook:ui3cujj5t2ti6ykjdn1d` (Macchine) — included here because it exercises the no-context refusal path that's relevant to the same `chat_rag` pipeline.
- **Audit date**: 2026-04-27

---

## Findings — 5 traces audited

### Summary table

| Trace ID | Question | Chunks | Top score | Total ms | Faithfulness | Grade |
|---|---|---|---|---|---|---|
| `rag_trace:3eu00gsxnzo3opcnx7x2` | "boffolotti" (nonsense term) | 7 | 0.633 | 18,343 | ✅ 0 hallucinations + correct refusal of unknown term | A |
| `rag_trace:czbi20p0ggmt6pkvnnh8` | Cartellino azzurro | 4 | 0.661 | 26,519 | ✅ 0 hallucinations | A+ |
| `rag_trace:vu88o3x47pzfwodm1pc2` | Tesseramento Prima Categoria | 4 | 0.630 | 26,391 | ✅ 0 hallucinations | A+ |
| `rag_trace:ytdtdyu5y0w27qd6ufr0` | Ammonizioni prima della squalifica | 5 | 0.636 | 18,891 | ✅ 0 hallucinations | A+ |
| `rag_trace:4m8mat066ko4l01dvxct` | Vincitore Serie A 2023 (out-of-scope) | 0 | — | 13,047 | ✅ Graceful refusal (no_context_refusal) | A+ |

**Overall verdict**: 0 hallucinations across 5 answers. Every cited article number, RESS section, deadline, sanction, and category-specific rule is verifiable in the source documents. Citation attribution between the 9 sources is correct in every spot-check, including multi-source answers (T1, T2). The refusal path is exercised correctly twice — once for an unknown term inside the corpus (T1: "boffolotti"), once for an entirely out-of-scope question (T4: Serie A 2023).

### Per-trace verification

#### `3eu00` — "Boffolotti" (nonsense / goliardic term)

The model correctly identified that "boffolotti" does not appear in the regulations, declined to invent a meaning, and pivoted to verifiable adjacent context, ending with a clarifying question.

| Claim | Source | Line(s) |
|---|---|---|
| "boffolotti" not in any regulation | grep across all 9 source files: 0 hits | — |
| Fallo di Posizione → punizione indiretto | Reg [o96] | 84–89 |
| 6 m per riprese / 2 m per rimessa laterale | Reg [o96] | 93–95 |
| Cartellino azzurro = espulsione temporanea 5 minuti | Reg [o96] | 487–488 |
| Portiere Under 9-10-11 non contrastabile in area | Reg [o96] | 866 (sezione), 896–897 |

⚠️ Pipeline observation: 6 of 7 chunks (scores 0.54–0.57) were tiny title/front-matter snippets from sibling integration files. Only the top chunk (0.633) had real signal. The synthesis correctly relied on the relevant chunk and ignored the noise.

#### `czbi` — Cartellino azzurro

| Claim | Source | Line(s) |
|---|---|---|
| Durata 5 minuti | Reg [o96] | 487 |
| Notificata via cartellino azzurro | Reg [o96] | 488 |
| 5 minuti effettivi, esclusi time-out | Reg [o96] | 492–494 |
| Fine tempo → sanzione scontata | Reg [o96] | 497 |
| 2 cartellini azzurri → rosso, non sostituibile | Reg [o96] | 502–505 |
| Ammonito + azzurro → espulsione definitiva | Reg [o96] | 510 |
| Fallo di Mano DOGSO-H | Reg [o96] | 514–518 |
| DOGSO-F (trattenere/spingere/tirare) | Reg [o96] | 520–535 |
| Esclusione: proteste/offese | Reg [o96] | 548–549 |
| Linguaggio blasfemo: atleti azzurro / dirigenti rosso | Reg [o96] | 554–562 |
| Tagliandino fine gara, firma società | Norme [mhd] | 1963–1966 |

Cross-source attribution is correct: the dirigenti/firma claim correctly cites both `[o96]` and `[mhd]`.

#### `vu88` — Tesseramento Prima Categoria

| Claim | Source | Line(s) |
|---|---|---|
| Open/Top Junior CA7 M: solo Seconda e Terza cat. FIGC (Prima esclusa) | Reg [o96] | 711–719 |
| Over: nessuna gara federale dal 01 luglio | Reg [o96] | 798 |
| Fuori quota Over 35/40: NON tesserato FIGC | Reg [o96] + Over [l9o] | 786, 791 / 246 |
| Giovanili fino a Juniores: nessun vincolo FIGC maschi | Reg [o96] | 741–742 |
| Decorrenza tesseramento: giorno successivo | Norme [mhd] | 962 |
| Termine tesseramento Over: 28 febbraio 2026 | Over [l9o] | 120 |
| Partecipazione = ingresso in campo, non panchina | Reg [o96] | 727–728 |

The "Prima Categoria esclusa" inference is logically correct (the rule explicitly lists Seconda+Terza, so Prima is by exclusion). The answer correctly partitioned the response by CSI category (Open/Top Junior vs. Over vs. Giovanili).

⚠️ Minor completeness gap: doesn't surface the Calcio a 5 alternative path (Reg line 719) — the answer focused only on Calcio a 11 federal status. Acceptable for a CA7-focused notebook.

#### `ytdtd` — Ammonizioni prima della squalifica

| Claim | Source | Line(s) |
|---|---|---|
| Calcio: squalifica alla 4ª ammonizione | Giustizia [tnn] | 562 |
| Pallavolo: 3ª penalità | Giustizia [tnn] | 563 |
| Pallacanestro: 3ª ammonizione | Giustizia [tnn] | 564 |
| Tornei/Coppe: 3ª ammonizione/penalità | Giustizia [tnn] | 566 |
| Manifestazioni Brevissime: 2ª ammonizione | Giustizia [tnn] | 570 |
| Recidive: −1 progressivo | Giustizia [tnn] | 568 |
| Diffida dirigenti alla 3ª (Calcio/Pallavolo/Pallacanestro) | Giustizia [tnn] | 177 |
| Cumulo non automatico, decorre 24 h dopo CU | Giustizia [tnn] | 578–580 |

All claims sourced from a single document; citations are precise.

#### `4m8m` — "Chi ha vinto Serie A 2023?" (out-of-scope, Macchine notebook)

- **Notebook**: `notebook:ui3cujj5t2ti6ykjdn1d` (Macchine, not Calcio)
- **Result**: `chunks_count: 0`, `no_context_refusal: true`
- **Answer**: *"Mi dispiace, ma non ho trovato informazioni rilevanti…"*

The pipeline correctly:
1. Ran 3 strategy searches (Serie A 2023 / classifica 2022-2023 / scudetto 2023).
2. Returned 0 chunks (no related content in the EU Machinery corpus).
3. Triggered the `no_context_refusal` path (commit `1d31f70`).
4. Refused without inventing facts.

This is the **happy path for a corpus-unrelated question** and validates that the refusal logic works as designed.

⚠️ Minor inefficiency: 6.8 s on retrieval for a query a corpus-aware planner could have rejected upfront — same observation as the Macchine audit.

---

## Pipeline observations

### Strengths
- **Faithfulness across 4 substantive answers + 2 refusals**: 0 hallucinations.
- **Citation discipline**: per-claim `[source:<id>]` (not just at the end). Multi-source claims (T1: `[o96]`+`[mhd]`; T2: `[o96]`+`[l9o]`+`[mhd]`) are correctly attributed.
- **Two refusal paths verified**:
  - In-corpus unknown term (T1 "boffolotti"): model declines to invent, pivots to adjacent verifiable context, asks for clarification.
  - Out-of-corpus question (T4 "Serie A 2023"): pipeline-level `no_context_refusal` after 0-chunk retrieval.
- **Robust to noisy retrieval**: T1 had 6 of 7 chunks as low-quality header snippets (0.54–0.57); the synthesis only used the one relevant chunk.
- **Followups** (3 per trace, 12/12 examined for the 4 substantive traces) are on-topic and drill into specific claims.

### Weaknesses
- **`source_insight` records under-retrieved**: 0 of 5 traces surfaced any insight chunk. Same pattern as the Macchine audit — likely a top-k or threshold issue.
- **Strategy over-plans on corpus-unknown topics**: T2 generated 4 generic FIGC/LND searches; T4 generated 3 Serie-A searches; T1 generated 3 boffolotti variants. None hit because the corpus is CSI-specific. Burns `search_ms` on questions where 1 search would have produced the same outcome.
- **Header-chunk inflation**: when query terminology doesn't match content (T1), the vector index falls back to title/front-matter chunks across all 9 documents. This inflates `chunks_count` without contributing usable context.
- **Retrieval latency dominates**: T1 10.2 s / T2 16.9 s / T3 10.6 s / T4 6.8 s — same bottleneck as the Macchine notebook (7–12 s of 13–26 s total).
- **`content_preview` 500-char cap** still constrains trace-only audits (had to cross-reference source files for every claim — same as the Macchine audit).

### Recommendations
1. **Cap strategy searches by corpus size or topical match** — for T2/T4, a corpus-aware planner could short-circuit to 0–1 searches and save 5–10 s.
2. **Boost `source_insight` records in retrieval** — repeating the recommendation from the Macchine audit; 0/5 surfaced here.
3. **Down-weight title/front-matter chunks** in the vector index — they crowd out signal when the query has no semantic anchor (T1 pattern).
4. **Persist 2–3 k of chunk content** in `rag_trace.chunks_retrieved` to enable trace-only audits.
5. **Split `search_ms`** into embed / vector-search / merge / rerank components — same recommendation as before; without breakdown it's hard to act on the latency.

---

## Cross-audit consistency (vs. Macchine notebook)

| Property | Macchine (4 traces) | Calcio (5 traces) |
|---|---|---|
| Hallucinations | 0 | 0 |
| Citation discipline | per-claim, correct | per-claim, correct |
| `source_insight` surfaced | 1 of 4 | 0 of 5 |
| Strategy over-plan on small/topic-mismatched corpus | yes | yes |
| Refusal path tested | implicit | explicit (T1 unknown term + T4 out-of-corpus) |
| Search latency dominant | yes (7–12 s) | yes (7–17 s) |

**Conclusion**: The faithfulness and citation discipline of `chat_rag` are consistent across two notebooks with different corpus sizes (2 docs vs. 9 docs) and different domains. The same pipeline-level weaknesses (over-planning, insight under-retrieval, search latency) recur in both audits and are corpus-independent — they belong on the pipeline backlog, not in any specific notebook's data.

---

## How to run a future audit

See [docs/RAG-DEBUG/RAG-AUDIT.md](../RAG-DEBUG/RAG-AUDIT.md) for the full step-by-step procedure (trace fetch → ground-truth dump → claim verification → scoring). For this notebook specifically:

### Pick traces to audit

```bash
set -a && . ./.env && set +a && uv run python -c "
import asyncio, json
from open_notebook.database.repository import repo_query
async def main():
    r = await repo_query('SELECT id, question, created FROM rag_trace WHERE notebook_id = notebook:wcey1gczvhr6vbdxyyn5 ORDER BY created DESC LIMIT 20;')
    print(json.dumps(r, default=str, indent=2, ensure_ascii=False))
asyncio.run(main())
"
```

### Source-side grep cheatsheet

```bash
cd "docs/REG-DEBUG2"
REG="Regolamento_Calcio_a_7_CSI_Brescia_2025-2026.md"
NORM="Norme_generali_CSI_Brescia_2025-2026.md"
GIUST="Giustizia-sportiva_CSI_Brescia_2025-2026.md"
OVER="Integrazione_CA7_OVER_2025-2026.md"

# Articles in main regulation
grep -nE '^## Art\. [0-9]+' "$REG"

# Cartellino azzurro / espulsione temporanea
grep -nE 'cartellino azzurro|espulsione temporanea|DOGSO|blasfem' "$REG"

# Tesseramento and FIGC categories
grep -nE 'Top Junior|Open|Seconda e Terza|JUNIORES|fuori quota|Federale' "$REG"

# Discipline / cumulo ammonizioni
grep -nE 'ammonizion|squalific|diffida|cumulo|recidiv|Brevissim|24 ore|Comunicato Ufficiale' "$GIUST"

# Over deadlines
grep -nE '28 febbraio|tesseramento|fuoriquota' "$OVER"
```

---

## Source layout reference

```
docs/REG-DEBUG2/
├── RAG-AUDIT.md  ← this file
├── Regolamento_Calcio_a_7_CSI_Brescia_2025-2026.md            (source:o96lyfb7pgus91ukip88)
├── Norme_generali_CSI_Brescia_2025-2026.md                    (source:mhdy24qn3g0qjkl7qjfs)
├── Giustizia-sportiva_CSI_Brescia_2025-2026.md                (source:tnn1sfdhe5y5ck30611c)
├── Integrazione_CA7_OVER_2025-2026.md                         (source:l9objoq2lg90ldjpug1r)
├── Integrazione_CA7_GIOVANILE_2025-2026.md                    (source:o6o2fixg00z3hly67uev)
├── Integrazione_CA7_U15_Femminile_2025-2026.md                (source:qc0hxp5v50qpx64bzqc0)
├── Integrazione_CA7_UNDER_2025-2026.md                        (source:72kjbq5ifoavfol8gff7)
├── Integrazione_CA7_OPEN_Maschile_SerieA_B_2025-2026.md       (source:x8vb1ut2r9d7o15gszkq)
└── Integrazione_CA7_OPEN_Femminile_2025-2026.md               (source:qhc7xgbvlasijxy1qstp)
```

---

## Changelog

- **2026-04-27** — Initial audit of 5 traces on notebook "Calcio-a-7" (4 substantive + 1 cross-notebook refusal validation). 0 hallucinations. Confirms pipeline-level findings from the Macchine audit are corpus-independent: under-retrieved `source_insight` records, strategy over-planning on topic-mismatched corpora, and search latency as the dominant cost.
