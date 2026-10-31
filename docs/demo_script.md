# Demo script (6–7 minutes)

**Before the demo:** run `python -m eval.run_eval` once, which pre-warms the LLM cache for every
question below, then `streamlit run app.py`, press **Reset index** and keep a terminal with
`pytest` ready.

| # | Time | Action | What to point at |
|---|---|---|---|
| 1 | 20 s | Frame the problem | "Answers are scattered across manuals, FAQs and release notes, and sometimes contradict each other. We built an assistant that proves every claim and admits what it doesn't know." |
| 2 | 40 s | Sidebar → **Load sample corpus** → **Documents** tab | 10 files in 4 formats; detected title, version, effective date, pages and chunk count. Open the admin guide's chunks: section breadcrumbs and page numbers. |
| 3 | 40 s | Ask **"What does ERR-429 mean?"** | Cited answer; open a citation: document, section, excerpt, identifier. Exact-code match is BM25's strength. |
| 4 | 40 s | Ask **"My teammate can't get into their account, what should an admin do?"** | Paraphrase → password-reset section of the PDF (page 3). Open **Why this answer**: dense vs BM25 ranks, RRF. |
| 5 | 60 s | Ask **"What's the refund window for annual plans?"** then **"What's the API rate limit?"** | Conflict cards: 30 vs 14 days and 100 vs 300 requests/min, side by side with version, date and the **NEWER** badge. No silent choice. |
| 6 | 20 s | Ask **"How long is deleted data kept on the Free plan?"** | 90 vs 180 days is *scoped* (Enterprise only), so it's correctly **not** flagged as a conflict. |
| 7 | 30 s | Ask **"What's the limit?"** | Clarifying question listing users, storage and rate limits. |
| 8 | 30 s | Ask **"What is the parental leave policy?"** | Clear not-found; "no LLM call was made". |
| 9 | 90 s | **Evaluation** tab | Recall@5, MRR, ablation (BM25 vs dense vs hybrid), status confusion matrix, one honest failure case from the report. |
| 10 | — | Invite a judge's own question; upload one of their files | Close on the architecture diagram in the README. |

**If the LLM is rate-limited:** answers come from the cache; otherwise the app shows the extractive
fallback (quoted passages with citations, marked *degraded*). This is also worth demonstrating.
