# Evaluation report

_Auto-generated tables by `python -m eval.run_eval`; analysis sections appended from `eval/report_analysis.md`._

## Test set

28 labelled questions over the 10-document synthetic corpus (see `docs/corpus.md`): 3 ambiguous, 4 conflict, 7 direct, 5 paraphrase, 1 scoped, 4 terminology, 4 unanswerable.

## Retrieval (top-5, answerable questions)

| mode | recall@1 | hit@1 | precision@1 | recall@3 | hit@3 | precision@3 | recall@5 | hit@5 | precision@5 | mrr |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.571 | 0.667 | 0.667 | 0.905 | 0.952 | 0.381 | 0.929 | 0.952 | 0.238 | 0.786 |
| dense | 0.762 | 0.905 | 0.905 | 1.0 | 1.0 | 0.429 | 1.0 | 1.0 | 0.257 | 0.952 |
| hybrid | 0.762 | 0.905 | 0.905 | 0.976 | 1.0 | 0.413 | 1.0 | 1.0 | 0.257 | 0.944 |

### By category

| category | questions | bm25 recall@5 | bm25 mrr | dense recall@5 | dense mrr | hybrid recall@5 | hybrid mrr |
|---|---|---|---|---|---|---|---|
| conflict | 4 | 0.875 | 0.833 | 1.0 | 1.0 | 1.0 | 1.0 |
| direct | 7 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| paraphrase | 5 | 1.0 | 0.633 | 1.0 | 0.9 | 1.0 | 1.0 |
| scoped | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| terminology | 4 | 0.75 | 0.5 | 1.0 | 0.875 | 1.0 | 0.708 |

### Not-found gate calibration (top dense cosine)

| T_LOW | min_top_score_answerable | max_top_score_unanswerable | suggested_T_LOW | separable |
|---|---|---|---|---|
| 0.59 | 0.601 | 0.58 | 0.591 | True |

## Answer quality and behaviour (all questions, end to end)

| questions | status_accuracy | not_found_recall | false_refusal_rate | clarification_rate | conflict_recall | conflict_false_positive_rate | citation_precision | answer_keyword_coverage | claims_removed_by_validator | llm_calls_avoided_by_gate | degraded_answers | latency_ms_p50 | latency_ms_p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 28 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | 0.867 | 0.952 | 0 | 6 | 0 | 531 | 1462 |

### Status confusion matrix (rows = expected, columns = predicted)

| expected | answered | partial | conflict | needs_clarification | not_found |
|---|---|---|---|---|---|
| answered | 17 | 0 | 0 | 0 | 0 |
| conflict | 0 | 0 | 4 | 0 | 0 |
| needs_clarification | 0 | 0 | 0 | 3 | 0 |
| not_found | 0 | 0 | 0 | 0 | 4 |

### Questions with an unexpected status

- none

## How the test set and labels were built

* **Corpus:** 10 synthetic documents about the fictional product "Nimbus Workspace" in PDF, HTML,
  Markdown and TXT (`docs/corpus.md`). They contain planted cases: two conflicts (refund window
  30 vs 14 days across a 2025 policy and a 2023 FAQ; API rate limit 100 vs 300 requests/min across
  API reference v2.4 and release notes v3.0), one *scoped* non-conflict (90 days generally vs
  180 days for Enterprise only), exact error codes, synonyms, and four topics that appear in no
  document.
* **Questions:** written by the developer from the corpus. Paraphrase and terminology questions
  deliberately avoid the documents' wording ("bounces" for *payment fails*, "seats" for *users*,
  "IdP" for *identity provider*, "workspace" where the document says *tenant*).
* **Gold labels:** `{document, evidence substring copied verbatim}`. A retrieved chunk counts as
  relevant when it comes from that document and contains the substring (case- and
  whitespace-insensitive), so labels survive any change to chunk size or chunk ids.
* **Expected status** is labelled for every question (`answered`, `conflict`,
  `needs_clarification`, `not_found`).

## Answer-quality method

* **Groundedness by construction:** every displayed claim must cite at least one evidence id that
  was actually retrieved; the validator drops invented ids and uncited claims, and its counts are
  reported above (`claims_removed_by_validator`).
* **Support check:** claims whose numbers or wording don't appear in the cited passage are marked
  *weakly supported* in the UI.
* **Citation precision:** share of citations (on answerable questions) that point to a
  gold-relevant chunk. Citations to related but non-gold passages (e.g. the API reference's error
  table cited alongside the troubleshooting article for ERR-429) count against it, so the metric is
  conservative.
* **Answer keyword coverage:** share of expected key facts (e.g. "30" and "14") present in the
  answer text.
* A manual read of the answers in `examples.md` confirmed that the claims match their excerpts.
  We did not use an LLM-as-judge: it would share the generator's biases.

## Retrieval analysis

* Dense retrieval is already very strong on this small, clean corpus (MRR ≈ 0.95), and hybrid
  matches it on recall (Recall@5 = 1.0). BM25 alone is clearly weaker on paraphrases (it ranks
  "Workspaces per Account" first for the locked-account question), but it wins on exact tokens:
  BM25 ranks the `ERR-403` section first for the bare code, and in hybrid it lifts the
  troubleshooting article to rank 1 for "What does ERR-429 mean?".
* The honest reading of the ablation is that hybrid buys **robustness** (exact codes, product
  names, version strings) at no cost in recall here, rather than a large average gain. On a larger
  or noisier corpus with more codes and identifiers we would expect the BM25 contribution to matter
  more.

## Failure cases (found during evaluation)

1. **Terminology refusal (t1, first run).** "How many seats does the Pro plan include?" retrieved
   the right passage at rank 1 (cosine 0.72), but the fallback model (`gemini-flash-lite-latest`,
   used because the primary model was overloaded) answered `not_found` because the document says
   *users*, not *seats*. Fix: a general prompt rule to treat clearly equivalent terms as the same.
   Lesson: retrieval can handle synonyms while a cautious generator still refuses, so both layers
   need testing.
2. **Fallback model ignored the output format (second run).** With the primary model overloaded,
   `gemini-flash-lite-latest` returned claims as plain strings without evidence ids for 6 of 21
   answerable questions. The validator did its job: claims without citations were removed, which
   turned those answers into `not_found` (no unsupported text reached the user, but status accuracy
   fell). Fix: Gemini structured output (`responseSchema`) enforces `{text, evidence[]}` for every
   model in the chain, plus one repair call if any claim still lacks evidence ids. Lesson: grounding
   checks must fail closed, and the output contract must be enforced by the API, not only the prompt.
3. **Conflict reported only in `conflicts[]` (third run).** For 3 of 4 conflict questions the model
   returned both positions (each with an evidence id) inside `conflicts[]` and left `claims[]`
   empty, which the engine first treated as "no supported claims" → `not_found`. Fix: when claims are
   empty but validated conflict positions exist, each position becomes a cited claim
   ("Billing Policy v2.0 (2025-03-01): ... [1]"). Covered by a unit test.
4. **Gate passes an unanswerable question (u4).** "How do we sign a GDPR data processing
   agreement?" scored 0.61, just above `T_LOW` = 0.59 (it shares vocabulary with the retention and
   security documents). The LLM layer correctly returned `not_found`, so the final answer was right,
   but one LLM call was spent. This is why there are two refusal layers.
5. **Narrow calibration margin.** The lowest-scoring answerable question (0.601) and the
   highest-scoring unanswerable one (0.580) are only 0.02 apart. `T_LOW` sits at the midpoint, so a
   slightly different phrasing could flip either way. The score is shown in the UI so the
   calibration stays visible.
6. **Model availability.** The configured primary model (`gemini-3.8-flash`) timed out during the
   run; the fallback chain answered with other Gemini models. That explains the p95 latency
   (first call pays the timeout, after which the model is skipped for 10 minutes). The final run's
   latency figures were measured with a warm LLM cache; the previous cold run had p50 ≈ 4.2 s and
   p95 ≈ 14 s end to end (retrieval alone takes tens of milliseconds).
7. **Clarify heuristic is coarse.** "How long is it kept?" is correctly treated as vague, but the
   rule (short generic question whose close matches span several sections) could also ask for
   clarification on a terse but specific question. The LLM's own `needs_clarification` status is
   a second path.

## Limitations

* Small test set (28 questions) and thresholds tuned on the same set they are evaluated on; the
  numbers show that the mechanisms work, not how well they generalise.
* Synthetic corpus with clean structure; real PDFs (multi-column layouts, scanned pages, complex
  tables) would lower extraction quality. No OCR.
* Conflict rules cover numeric/unit disagreements; verbal contradictions ("supported" vs
  "deprecated") rely on the LLM.
* Hosted free-tier LLM: latency and availability vary; answers are cached by prompt hash.
