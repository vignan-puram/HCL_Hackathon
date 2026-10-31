# Enterprise Documentation Q&A Assistant

HCLTech Future Ready AI Engineer Hackathon, Use Case 2. Upload PDF, HTML, Markdown and TXT
documents, ask questions in natural language, and get **concise answers grounded only in the
retrieved passages**, with a **citation on every claim**, **visible conflicts** between sources, and
an honest **"not found"** or **clarifying question** when the documents don't support an answer.

![Pipeline](https://img.shields.io/badge/pipeline-hybrid%20RAG%20%2B%20evidence%20gate-blue)
![Tests](https://img.shields.io/badge/tests-pytest-green)

## Quick start (Windows, Python 3.12)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          # add GEMINI_API_KEY (free key from Google AI Studio)
python -m scripts.check_llm     # verifies the key and the model fallback chain
streamlit run app.py            # http://localhost:8501 → "Load sample corpus" → ask
```

Other commands (run from the repo root):

| Command | What it does |
|---|---|
| `pytest` | 30 tests: parsers, chunking, retrieval, validator, gate, conflicts, LLM fallback |
| `python -m eval.run_eval` | Full evaluation → `eval/reports/` (metrics.json, eval_report.md, examples.md) |
| `python -m eval.run_eval --retrieval-only` | Retrieval metrics + gate calibration, no LLM calls |
| `python -m rag.engine index-corpus` | Index `eval/corpus/` from the command line |
| `python -m rag.engine ask "question" [--retrieve --mode bm25\|dense\|hybrid]` | Query from the command line |
| `python -m ingestion.pipeline <file>` | Show how one file is parsed and chunked |
| `python -m eval.build_corpus` | Regenerate the two sample PDFs |

No API key? The app still works: answers fall back to **extractive mode** (top passages quoted
with citations, marked *degraded*).

## Requirement coverage

| Req | Capability | Where |
|---|---|---|
| FR-01 | Multi-format ingestion with title, section, page, version, date metadata | `ingestion/parsers.py`, `ingestion/pipeline.py` |
| FR-02 | Cleaning + section-aware chunking with breadcrumbs | `ingestion/chunker.py` |
| FR-03 | Hybrid retrieval: BM25 + dense embeddings fused with RRF | `rag/retriever.py`, `rag/index.py`, `rag/embeddings.py` |
| FR-04 | Grounded answers: one JSON-mode LLM call over numbered evidence | `rag/generator.py`, `rag/llm.py` |
| FR-05 | Claim-level citations: document, section, page, version, date, excerpt, identifier | `rag/validator.py` |
| FR-06 | Conflict detection: rules propose, LLM confirms, both sides shown with dates | `rag/conflicts.py` |
| FR-07 | Confidence score, not-found gate before the LLM, clarifying questions | `rag/confidence.py` |
| FR-08 | Labelled set (28 questions), Recall@k/MRR, ablation, status metrics | `eval/` |

## Architecture

One Python process. The Streamlit UI imports the engine directly; `rag/engine.py` is the only
entry point used by the app, the CLI and the evaluation runner.

```
              ┌──────────────────── ingestion ────────────────────┐
 file ──────► │ parse (PDF/HTML/MD/TXT) → clean → metadata → chunk │ ──► embed ──► index (NumPy + BM25, on disk)
              └────────────────────────────────────────────────────┘
                                                                          │
 question ──► hybrid retrieve (dense top-20 + BM25 top-20 → RRF → top-5) ◄┘
          ──► confidence gate ── weak evidence ──► not_found   (no LLM call)
                              └─ vague, split topics ──► needs_clarification (no LLM call)
          ──► conflict rules (same unit, different values, same topic)
          ──► ONE LLM call: numbered evidence [S1..S5] + conflict hints → JSON claims
          ──► validator: drop invented ids & uncited claims, renumber, pick excerpts, support check
          ──► merge conflicts ──► QueryResult (status, answer, claims, citations, conflicts, confidence)
                LLM down? ──► extractive fallback (quoted passages, degraded=true)
```

```
app.py                 Streamlit UI: Ask / Documents / Evaluation tabs
config.py              paths, model names, thresholds
common/schemas.py      Pydantic data contracts (Chunk, Citation, Conflict, QueryResult, ...)
ingestion/             parsers.py, chunker.py, pipeline.py, errors.py
rag/                   embeddings, index, retriever, confidence, conflicts, llm, generator, validator, engine
eval/                  corpus/, build_corpus.py, questions.jsonl, metrics.py, run_eval.py, reports/
tests/                 pytest suite
docs/                  corpus.md, demo_script.md, blueprint/ (original design + problem brief)
```

## Supported formats and ingestion

| Format | Parser | Structure recovered |
|---|---|---|
| PDF | PyMuPDF | Headings by font size (≥ 1.15 × median body size, ranked into levels), 1-based page numbers, repeated headers/footers removed. Scanned/image-only PDFs → clear `EMPTY_DOCUMENT` error (no OCR). |
| HTML | BeautifulSoup (`html.parser`) | `h1`–`h6` tree; `p`, `li`, `pre`, tables (rows kept as `a \| b \| c`); `script/style/nav/header/footer` dropped; `<title>` and `<meta>` read. |
| Markdown | own parser | ATX `#` headings, YAML front matter (title/version/date/kind), code fences kept whole. |
| TXT | own heuristics | `===`/`---` underlined headings, numbered headings (`2.1 Title`), ALL-CAPS headings, blank-line paragraphs. |

**Metadata:** title (front matter / `<title>` / PDF metadata / first heading / filename),
version (`Version 3.0`, `v2.4`, front matter), effective date (ISO date near *effective, updated,
released, reviewed*), document kind (front matter or filename keywords). Values entered at upload
time override extracted ones. Version and date are copied onto every chunk so retrieval and
conflict rules never need a lookup.

**Chunking (section-aware):** a chunk never crosses a top-level section; each (sub)section becomes
one chunk when it fits (target 300, max 450 words); longer sections split on paragraphs then
sentences with 40 words of overlap; sections under 40 words merge into a sibling under the same
parent (keeping their heading as a line of text); tables and code blocks are never split. Each chunk
is embedded as `"{title} > {section path}\n{text}"`, so a passage like "choose Reset password" is
findable from "how do I reset a user's password". Duplicate uploads are detected by SHA-256.

## Retrieval

* **Dense:** `BAAI/bge-small-en-v1.5` (384-d) through **fastembed** (ONNX, CPU, no PyTorch), with the
  bge query instruction prefix; brute-force cosine over a NumPy matrix (milliseconds at this scale).
* **Keyword:** BM25Okapi over the breadcrumbed text; the tokenizer keeps codes such as `ERR-429` whole
  *and* split, so both "ERR-429" and "429 error" match.
* **Fusion:** Reciprocal Rank Fusion, `score = Σ 1 / (60 + rank)`, with no score normalisation needed.
  Raw cosine, dense rank and BM25 rank are kept for the gate and the "Why this answer" panel.
* Filters by document, version or product are supported by `engine.retrieve(..., filters=...)`.

## Answer generation and grounding

Hallucination is fought in four layers:
1. **Evidence gate before the LLM:** if the best cosine is below `T_LOW` the answer is *not found*
   and no LLM call is made.
2. **Prompt:** only numbered evidence `[S1]…[S5]` (with document, section, page, version, date); every
   claim must cite an S-id; disagreeing sources must both be reported; insufficient evidence →
   `not_found`; vague question → `needs_clarification`.
3. **Structured output:** Gemini `responseSchema` enforces `claims[{text, evidence[]}]`, `conflicts[]`
   and `status` for every model in the fallback chain; one repair call on invalid JSON or claims
   without evidence ids.
4. **Validation:** invented ids are dropped, claims left without a valid citation are removed (and
   counted), citations are renumbered `[1..n]` by first use, each citation gets the best-matching
   sentence as its excerpt, and claims whose numbers or wording are absent from the cited passage are
   marked *weakly supported* in the UI.

**LLM access:** plain HTTPS via `requests` (no SDKs). Default provider is Gemini with a fallback
chain (`GEMINI_MODEL` → `GEMINI_FALLBACK_MODELS`); a model that times out is skipped for 10 minutes.
Any OpenAI-compatible endpoint (e.g. Groq) works with `LLM_PROVIDER=openai`. Responses are cached on
disk by prompt hash, so repeated questions are instant and the demo survives rate limits.

## Conflict handling

A rule proposes a conflict when two retrieved passages from **different documents** are about the
same thing (embedding similarity ≥ 0.75, both close to the best match, and the sentences stating
the values share a topic word) and state the **same unit with disjoint values** (days, hours,
requests per minute, GB, users, %, $). Requiring *disjoint* values keeps scoped statements
("180 days for Enterprise; 90 days for Free and Pro" vs "90 days") from being flagged. Documents
with different version numbers give a `version_sensitive` conflict, otherwise `value_mismatch`.

Rule hits are passed to the LLM as hints; the LLM confirms them or reports its own. Conflicts the
LLM reports are kept (`detected_by: llm` or `rules+llm`); a rule conflict the LLM ignored is still
shown as a *possible conflict* when the answer cites both sides. The UI shows the positions side by
side with version, date and a **NEWER** badge, plus a hint to confirm which applies.

## Confidence and fallback

`score = 0.6 · relevance + 0.2 · agreement + 0.2 · support`, where relevance is the top cosine rescaled
above `T_LOW`, agreement means the top passage is in both the dense and BM25 top 5, and support counts
passages close to the top score. Levels: high ≥ 0.70, medium ≥ 0.45. Reasons are shown in the UI.
Statuses: `answered`, `partial`, `conflict`, `needs_clarification`, `not_found`. If the LLM is
unavailable, the extractive fallback quotes the top passages with citations (`degraded: true`).
The UI never shows a stack trace.

## Evaluation

See [`eval/reports/eval_report.md`](eval/reports/eval_report.md) for the full report and
[`eval/reports/examples.md`](eval/reports/examples.md) for evidence-backed examples (direct,
paraphrase, terminology, two conflicts, a scoped non-conflict, an ambiguous and an unsupported
question). The same numbers appear in the app's **Evaluation** tab.

* **Test set:** 28 questions (7 direct, 5 paraphrase, 4 terminology, 4 conflict, 1 scoped
  non-conflict, 3 ambiguous, 4 unanswerable) in `eval/questions.jsonl`.
* **Labels:** gold = `{document, verbatim evidence substring}`; a retrieved chunk is relevant when it
  comes from that document and contains the substring, so labels survive any re-chunking.
* **Retrieval:** Recall@1/3/5, Hit@k, Precision@k, MRR for BM25-only, dense-only and hybrid, overall
  and per category; not-found gate calibration (`T_LOW` = midpoint between the highest top score of an
  unanswerable question and the lowest of an answerable one).
* **Behaviour:** status accuracy + confusion matrix, not-found recall, false-refusal rate,
  clarification rate, conflict recall (status `conflict` *and* both sides cited), conflict
  false-positive rate, citation precision, answer keyword coverage, claims removed by the validator,
  LLM calls avoided by the gate, p50/p95 latency.

## Assumptions

* **Synthetic corpus:** no document collection was supplied, so `eval/corpus/` contains 10 short
  documents about a fictional product, "Nimbus Workspace", in all four formats, with planted
  conflicts, a version change, a scoped non-conflict, exact error codes, synonyms (SSO/SAML,
  seats/users, workspace/tenant) and deliberate gaps. See [`docs/corpus.md`](docs/corpus.md).
* The 28 evaluation questions and their labels were written by the developer from the corpus;
  paraphrase and terminology questions deliberately avoid the documents' wording.
* Free-tier LLM: the documents are synthetic, so sending them to a hosted model is acceptable.
  Don't upload confidential documents when using a free tier.

## Design trade-offs

| Decision | Why | Cost |
|---|---|---|
| Single Streamlit app, engine imported directly | One process, no HTTP layer to break; the brief accepts a web UI | No separate REST API (the engine API is one file and could be wrapped by FastAPI) |
| NumPy + rank_bm25, no vector DB | < 10k chunks: exact search in milliseconds, zero setup | Linear scan won't scale to millions of chunks |
| fastembed (ONNX) instead of sentence-transformers | Same bge model, installs in ~1 min without PyTorch | ONNX model download (~70 MB) on first run |
| One LLM call per question, JSON output | Cheap, fast, debuggable; citations are validated in code | No multi-hop reasoning across many documents |
| Rules + LLM for conflicts | Rules are deterministic and explainable; LLM covers wording rules miss | Rules only see numeric/unit disagreements |
| Gate before the LLM | Unanswerable questions cost nothing and can't hallucinate | Threshold tuned on a small set |
| No LangChain/LlamaIndex | Every judged step (gate, citations, conflicts) is visible code | More code to write |

## Limitations

* Thresholds (`T_LOW`, confidence levels, conflict similarity) are tuned on the same 28-question set
  they are evaluated on; the gap between answerable and unanswerable top scores is small (≈0.02), so
  a new corpus needs re-calibration (`python -m eval.run_eval --retrieval-only` prints the suggestion).
* PDF heading detection relies on font size; unusual layouts fall back to page-level citations.
  No OCR for scanned PDFs, and complex PDF tables are read as text lines.
* Conflict rules detect numeric/unit disagreements; purely verbal contradictions depend on the LLM.
* The clarify rule is heuristic (short, generic question whose close matches span several sections).
* Hosted free-tier models can be slow or overloaded; the fallback chain, cache and extractive mode
  keep the app usable but latency varies.
* PyMuPDF is AGPL-licensed (fine for a hackathon prototype).
