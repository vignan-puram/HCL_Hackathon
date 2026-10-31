"""Run the labelled question set and write the evaluation report.

  python -m eval.run_eval                  # retrieval + end-to-end (calls the LLM)
  python -m eval.run_eval --retrieval-only # retrieval metrics and gate calibration only

Indexes eval/corpus into its own data_eval/ directory so the app's index is untouched.
The LLM cache is shared with the app, so an eval run also pre-warms the demo questions.
Outputs: eval/reports/metrics.json, results.jsonl, eval_report.md, examples.md
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("DATA_DIR", str(ROOT / "data_eval"))

import config  # noqa: E402

config.LLM_CACHE_DIR = ROOT / "data" / "llm_cache"

from common.schemas import QueryResult  # noqa: E402
from eval.metrics import ANSWERABLE, behaviour_metrics, mean_scores, retrieval_scores  # noqa: E402
from rag import engine  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.jsonl"
MODES = ("bm25", "dense", "hybrid")
EXAMPLE_IDS = ["d1", "p1", "t1", "c1", "c2", "s1", "a1", "u1"]


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_index() -> None:
    engine.reset()
    for path in sorted(config.CORPUS_DIR.iterdir()):
        if path.suffix.lower() in config.ALLOWED_EXTENSIONS:
            engine.ingest_file(path)
    stats = engine.index_stats()
    print(f"indexed {stats.documents} documents, {stats.chunks} chunks")


def run_retrieval(items: list[dict]) -> dict:
    scored = [i for i in items if i["gold"]]
    out: dict = {"retrieval": {}, "retrieval_by_category": []}
    per_mode_cat: dict[str, dict[str, list]] = {m: defaultdict(list) for m in MODES}
    for mode in MODES:
        rows = []
        for item in scored:
            hits = engine.retrieve(item["question"], k=5, mode=mode)
            row = retrieval_scores([h.chunk for h in hits], item["gold"])
            rows.append(row)
            per_mode_cat[mode][item["category"]].append(row)
        out["retrieval"][mode] = {"overall": mean_scores(rows)}
    for cat in sorted({i["category"] for i in scored}):
        row = {"category": cat, "questions": len(per_mode_cat["hybrid"][cat])}
        for mode in MODES:
            m = mean_scores(per_mode_cat[mode][cat])
            row[f"{mode} recall@5"] = m["recall@5"]
            row[f"{mode} mrr"] = m["mrr"]
        out["retrieval_by_category"].append(row)

    # gate calibration: top cosine of answerable vs unanswerable questions
    top = {i["id"]: engine.retrieve(i["question"], k=1)[0].dense_score for i in items}
    ans = [top[i["id"]] for i in items if i["category"] in ANSWERABLE]
    unans = [top[i["id"]] for i in items if i["category"] == "unanswerable"]
    out["gate_calibration"] = {
        "T_LOW": config.T_LOW,
        "min_top_score_answerable": round(min(ans), 3),
        "max_top_score_unanswerable": round(max(unans), 3),
        "suggested_T_LOW": round((min(ans) + max(unans)) / 2, 3),
        "separable": min(ans) > max(unans),
    }
    return out


def run_end_to_end(items: list[dict], pause: float) -> tuple[list[QueryResult], list[dict]]:
    results, rows = [], []
    for n, item in enumerate(items, start=1):
        res = engine.query_documents(item["question"], debug=True)
        results.append(res)
        ok = "OK " if res.status == item["expected_status"] else "XX "
        print(f"[{n:02d}/{len(items)}] {ok}{item['id']:3} expected={item['expected_status']:19} got={res.status:19} "
              f"{res.timings_ms.get('total', 0):>6} ms  {item['question']}")
        rows.append({"id": item["id"], "category": item["category"], "question": item["question"],
                     "expected_status": item["expected_status"], **res.model_dump(mode="json", exclude={"debug"}),
                     "debug": res.debug})
        if res.debug and res.debug.get("llm_called") and not res.debug.get("llm_cached"):
            time.sleep(pause)  # respect free-tier rate limits
    return results, rows


# ---------------------------------------------------------------- report


def _table(rows: list[dict]) -> str:
    if not rows:
        return ""
    head = list(rows[0])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(r[h]) for h in head) + " |" for r in rows]
    return "\n".join(lines)


def _example_md(row: dict) -> str:
    lines = [f"### {row['id']} ({row['category']}): {row['question']}",
             f"**Expected:** `{row['expected_status']}` · **Got:** `{row['status']}` · "
             f"confidence {row['confidence']['level']} ({row['confidence']['score']}) · model `{row['model']}`", "",
             "> " + row["answer"].replace("\n", "\n> "), ""]
    if row.get("clarifying_question") and row["status"] == "needs_clarification":
        lines += [f"Clarifying question: _{row['clarifying_question']}_", ""]
    for c in row["citations"]:
        page = f", p. {c['page']}" if c["page"] else ""
        meta = ", ".join(x for x in [f"v{c['version']}" if c["version"] else "", c["effective_date"] or ""] if x)
        lines.append(f"- **[{c['id']}]** {c['doc_title']} · {c['section']}{page} ({meta or 'no version'}) "
                     f"`{c['source_url']}`: \"{c['excerpt']}\"")
    for conf in row["conflicts"]:
        sides = "; ".join(f"[{p['citation_id']}] {p['statement']}" for p in conf["positions"])
        lines.append(f"- **Conflict** ({conf['type']}, detected by {conf['detected_by']}): {sides}. "
                     f"{conf.get('resolution_hint') or ''}")
    dbg = row.get("debug") or {}
    if not dbg.get("llm_called"):
        lines.append(f"- No LLM call made (gate decision: `{dbg.get('gate', {}).get('decision')}`)")
    return "\n".join(lines) + "\n"


def write_reports(items: list[dict], metrics: dict, rows: list[dict]) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if rows:
        with open(config.REPORTS_DIR / "results.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        by_id = {r["id"]: r for r in rows}
        examples = ["# Evidence-backed examples", "",
                    "Generated by `python -m eval.run_eval` from the live system on the sample corpus.", ""]
        examples += [_example_md(by_id[i]) for i in EXAMPLE_IDS if i in by_id]
        (config.REPORTS_DIR / "examples.md").write_text("\n".join(examples), encoding="utf-8")

    cats = defaultdict(int)
    for i in items:
        cats[i["category"]] += 1
    md = ["# Evaluation report", "",
          "_Auto-generated tables by `python -m eval.run_eval`; analysis sections appended from "
          "`eval/report_analysis.md`._", "",
          "## Test set", "",
          f"{len(items)} labelled questions over the 10-document synthetic corpus (see `docs/corpus.md`): "
          + ", ".join(f"{n} {c}" for c, n in sorted(cats.items())) + ".", "",
          "## Retrieval (top-5, answerable questions)", "",
          _table([{"mode": m, **v["overall"]} for m, v in metrics["retrieval"].items()]), "",
          "### By category", "", _table(metrics["retrieval_by_category"]), "",
          "### Not-found gate calibration (top dense cosine)", "",
          _table([metrics["gate_calibration"]]), ""]
    beh = metrics.get("behaviour")
    if beh:
        md += ["## Answer quality and behaviour (all questions, end to end)", "",
               _table([{k: v for k, v in beh.items() if k != "confusion"}]), "",
               "### Status confusion matrix (rows = expected, columns = predicted)", ""]
        statuses = ["answered", "partial", "conflict", "needs_clarification", "not_found"]
        md += [_table([{"expected": e, **{s: beh["confusion"].get(e, {}).get(s, 0) for s in statuses}}
                       for e in statuses if e in beh["confusion"]]), ""]
        wrong = [r for r in rows if r["status"] != r["expected_status"]]
        md += ["### Questions with an unexpected status", ""]
        md += [f"- `{r['id']}` {r['question']}: expected `{r['expected_status']}`, got `{r['status']}`" for r in wrong] or ["- none"]
        md.append("")
    analysis = ROOT / "eval" / "report_analysis.md"
    if analysis.exists():
        md += [analysis.read_text(encoding="utf-8")]
    (config.REPORTS_DIR / "eval_report.md").write_text("\n".join(md), encoding="utf-8")


def main(argv: list[str]) -> None:
    items = load_questions()
    build_index()
    metrics = run_retrieval(items)
    print(json.dumps({m: v["overall"] for m, v in metrics["retrieval"].items()}, indent=1))
    print("gate calibration:", metrics["gate_calibration"])
    rows: list[dict] = []
    if "--retrieval-only" not in argv:
        results, rows = run_end_to_end(items, pause=float(os.getenv("EVAL_PAUSE_S", "2")))
        chunk_text = {c.chunk_id: c.text for d in engine.list_documents() for c in engine.document_chunks(d.doc_id)}
        metrics["behaviour"] = behaviour_metrics(items, results, chunk_text)
        print(json.dumps({k: v for k, v in metrics["behaviour"].items() if k != "confusion"}, indent=1))
    elif (config.REPORTS_DIR / "metrics.json").exists():  # keep the last end-to-end numbers
        previous = json.loads((config.REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
        metrics["behaviour"] = previous.get("behaviour", {})
        if (config.REPORTS_DIR / "results.jsonl").exists():
            rows = [json.loads(line) for line in (config.REPORTS_DIR / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    write_reports(items, metrics, rows)
    print(f"wrote {config.REPORTS_DIR / 'eval_report.md'}")


if __name__ == "__main__":
    main(sys.argv[1:])
