"""Enterprise Documentation Q&A Assistant -- Streamlit UI.

Run from repo root: streamlit run app.py
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from common.schemas import QueryResult, UserMeta
from ingestion.errors import IngestionError
from rag import engine, llm

st.set_page_config(page_title="Nimbus Docs Q&A", page_icon="📚", layout="wide")

STATUS_STYLE = {
    "answered": ("Answered", "#1a7f37"),
    "partial": ("Partial answer", "#9a6700"),
    "conflict": ("Sources disagree", "#bc4c00"),
    "needs_clarification": ("Needs clarification", "#0969da"),
    "not_found": ("Not found in documents", "#6e7781"),
}
LEVEL_COLOR = {"high": "#1a7f37", "medium": "#9a6700", "low": "#cf222e"}
EXAMPLES = [
    ("Direct", "What does ERR-429 mean?"),
    ("Paraphrase", "My teammate can't get into their account, what should an admin do?"),
    ("Conflict", "What's the refund window for annual plans?"),
    ("Version conflict", "What's the API rate limit?"),
    ("Ambiguous", "What's the limit?"),
    ("Unanswerable", "What is the parental leave policy?"),
]

st.markdown(
    """
    <style>
    .badge {display:inline-block; padding:2px 10px; border-radius:999px; color:#fff;
            font-size:0.78rem; font-weight:600; margin-right:6px;}
    .reasons {font-size:0.8rem; opacity:0.75; margin-top:4px;}
    .cite {font-weight:600; color:#0969da;}
    .card {border:1px solid rgba(128,128,128,0.35); border-radius:10px; padding:12px 14px; height:100%;}
    .card .meta {font-size:0.8rem; opacity:0.75;}
    .newer {background:#1a7f37; color:#fff; border-radius:6px; padding:1px 6px; font-size:0.7rem; margin-left:6px;}
    .stat {font-size:0.85rem; opacity:0.8;}
    </style>
    """,
    unsafe_allow_html=True,
)


def badge(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{text}</span>'


def esc(text: str) -> str:
    return text.replace("$", "\\$")  # keep Streamlit from rendering "$12 ... $10" as LaTeX


# ---------------------------------------------------------------- sidebar


def ingest_uploads(files, meta: UserMeta | None) -> None:
    for f in files:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f.name
            path.write_bytes(f.getvalue())
            try:
                status, rec = engine.ingest_file(path, meta, filename=f.name)
                if status == "duplicate":
                    st.sidebar.info(f"{f.name}: already indexed")
                else:
                    st.sidebar.success(f"{f.name}: {rec.num_chunks} chunks")
            except IngestionError as exc:
                st.sidebar.error(f"{f.name}: {exc.message}")


def sidebar() -> None:
    sb = st.sidebar
    sb.header("📚 Knowledge base")
    stats = engine.index_stats()
    sb.markdown(f'<div class="stat">{stats.documents} documents · {stats.chunks} chunks indexed</div>',
                unsafe_allow_html=True)

    files = sb.file_uploader("Upload documents", type=["pdf", "html", "htm", "md", "markdown", "txt"],
                             accept_multiple_files=True, key=f"uploader_{st.session_state.get('upload_round', 0)}")
    with sb.expander("Optional metadata (applies to this batch)"):
        version = st.text_input("Version", placeholder="e.g. 2.0")
        date = st.text_input("Effective date", placeholder="YYYY-MM-DD")
        kind = st.selectbox("Document kind", ["(auto)", "manual", "faq", "release_notes", "policy",
                                              "support_article", "other"])
    if sb.button("Index uploaded files", type="primary", disabled=not files, width="stretch"):
        meta = None
        if version or date or kind != "(auto)":
            if date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                sb.error("Effective date must be YYYY-MM-DD.")
                return
            meta = UserMeta(version=version or None, effective_date=date or None,
                            doc_kind=None if kind == "(auto)" else kind)
        with st.spinner("Parsing, chunking and embedding..."):
            ingest_uploads(files, meta)
        st.session_state["upload_round"] = st.session_state.get("upload_round", 0) + 1
        st.rerun()

    if sb.button("Load sample corpus (Nimbus Workspace)", width="stretch"):
        with st.spinner("Indexing sample corpus..."):
            for path in sorted(config.CORPUS_DIR.iterdir()):
                if path.suffix.lower() in config.ALLOWED_EXTENSIONS:
                    engine.ingest_file(path)
        st.rerun()

    sb.divider()
    model = llm.models()[0] if llm.is_configured() else None
    sb.markdown(f"**LLM:** {config.LLM_PROVIDER} · `{model}`" if model else
                "**LLM:** not configured: answers are extractive only")
    if sb.checkbox("Allow index reset"):
        if sb.button("Reset index", width="stretch"):
            engine.reset()
            st.session_state["history"] = []
            st.rerun()


# ---------------------------------------------------------------- answer rendering


def render_answer_text(text: str) -> str:
    return re.sub(r"\[(\d+(?:,\s*\d+)*)\]", r'<span class="cite">[\1]</span>', esc(text)).replace("\n", "<br>")


def render_conflicts(result: QueryResult) -> None:
    by_id = {c.id: c for c in result.citations}
    for conflict in result.conflicts:
        label = "⚠️ Possible conflict" if conflict.detected_by == "rules" else "⚠️ Conflict"
        st.markdown(f"**{label}: {esc(conflict.topic)}** · _{conflict.type.replace('_', ' ')}_ "
                    f"· detected by {conflict.detected_by}")
        dates = [p.effective_date for p in conflict.positions if p.effective_date]
        newest = max(dates) if len(set(dates)) > 1 else None
        cols = st.columns(len(conflict.positions))
        for col, pos in zip(cols, conflict.positions):
            cite = by_id.get(pos.citation_id)
            doc = cite.doc_title if cite else "Unknown source"
            newer = '<span class="newer">NEWER</span>' if newest and pos.effective_date == newest else ""
            meta = " · ".join(x for x in [f"v{pos.version}" if pos.version else "",
                                          pos.effective_date or "", f"[{pos.citation_id}]"] if x)
            col.markdown(
                f'<div class="card"><b>{esc(pos.statement)}</b>{newer}'
                f'<div class="meta">{esc(doc)} · {meta}</div></div>',
                unsafe_allow_html=True,
            )
        if conflict.resolution_hint:
            st.caption(f"💡 {conflict.resolution_hint}")


def render_citations(result: QueryResult, key: str) -> None:
    st.markdown("**Sources**")
    for c in result.citations:
        parts = [c.doc_title, c.section]
        if c.page:
            parts.append(f"p. {c.page}")
        if c.version:
            parts.append(f"v{c.version}")
        if c.effective_date:
            parts.append(c.effective_date)
        with st.expander(f"[{c.id}] " + " · ".join(parts)):
            st.markdown(f"> {esc(c.excerpt)}")
            st.caption(f"Source: `{c.source_url}` · chunk `{c.chunk_id}` · relevance {c.relevance:.2f}")
            chunk = next((ch for ch in engine.document_chunks(c.doc_id) if ch.chunk_id == c.chunk_id), None)
            if chunk and st.toggle("Show full passage", key=f"full_{key}_{c.id}"):
                st.text(chunk.text)
            original = engine.original_file(c.doc_id)
            if original and original.exists():
                st.download_button("Download original", original.read_bytes(), file_name=c.filename,
                                   key=f"dl_{key}_{c.id}")


def render_debug(result: QueryResult) -> None:
    dbg = result.debug or {}
    with st.expander("🔎 Why this answer"):
        gate = dbg.get("gate", {})
        if gate:
            st.markdown(f"**Gate decision:** `{gate.get('decision')}` · top cosine "
                        f"{gate.get('top_score', 0):.3f} (threshold {gate.get('threshold', 0):.2f})")
        st.markdown(f"**LLM called:** {'yes' if dbg.get('llm_called') else 'no'}"
                    + (f" · model `{result.model}`" if result.model else "")
                    + (" · served from cache" if dbg.get("llm_cached") else ""))
        val = dbg.get("validator")
        if val:
            st.markdown(f"**Citation validator:** {val.get('dropped_claims', 0)} unsupported claim(s) removed, "
                        f"{len(val.get('invalid_ids', []))} invented evidence id(s) dropped")
        if dbg.get("conflict_candidates"):
            st.markdown("**Rule-based conflict hints:** " + "; ".join(dbg["conflict_candidates"]))
        if dbg.get("hits"):
            st.dataframe(pd.DataFrame(dbg["hits"]), hide_index=True, width="stretch")
        st.caption(f"Timings (ms): {result.timings_ms}")


def render_result(result: QueryResult, key: str) -> None:
    label, color = STATUS_STYLE[result.status]
    conf = result.confidence
    st.markdown(
        badge(label, color) + badge(f"Confidence: {conf.level} ({conf.score:.2f})", LEVEL_COLOR[conf.level])
        + (badge("Degraded: extractive", "#6e7781") if result.degraded else "")
        + (f'<div class="reasons">{" · ".join(esc(r) for r in conf.reasons)}</div>' if conf.reasons else ""),
        unsafe_allow_html=True,
    )
    if result.degraded:
        st.warning("The language model was unavailable, so these are the most relevant passages, quoted directly.")
    if result.status == "needs_clarification" and result.clarifying_question:
        st.info(f"❓ {result.clarifying_question}")
    elif result.status == "not_found":
        st.info(result.answer)
        if not (result.debug or {}).get("llm_called"):
            st.caption("Evidence was too weak, so no LLM call was made: nothing could be invented.")
    if result.status != "not_found" and result.answer:
        st.markdown(render_answer_text(result.answer), unsafe_allow_html=True)
    if any(cl.support == "weak" for cl in result.claims):
        st.caption("Some statements are only weakly supported by the cited passages.")
    if result.conflicts:
        render_conflicts(result)
    if result.citations:
        render_citations(result, key)
    render_debug(result)


# ---------------------------------------------------------------- tabs


def ask_tab() -> None:
    history: list[tuple[str, QueryResult]] = st.session_state.setdefault("history", [])
    st.caption("Try an example:")
    cols = st.columns(len(EXAMPLES))
    clicked = None
    for col, (label, question) in zip(cols, EXAMPLES):
        if col.button(label, help=question, width="stretch"):
            clicked = question

    for i, (question, result) in enumerate(history):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            render_result(result, key=str(i))

    question = st.chat_input("Ask a question about your documents...") or clicked
    if question:
        if engine.index_stats().chunks == 0:
            st.warning("Upload documents or load the sample corpus first.")
            return
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and drafting a grounded answer..."):
                result = engine.query_documents(question.strip()[:1000], debug=True)
            render_result(result, key=str(len(history)))
        history.append((question, result))


def documents_tab() -> None:
    docs = engine.list_documents()
    if not docs:
        st.info("No documents indexed yet.")
        return
    st.dataframe(
        pd.DataFrame([{
            "File": d.filename, "Title": d.title, "Type": d.doc_type, "Kind": d.doc_kind,
            "Version": d.version or "", "Effective date": d.effective_date or "",
            "Pages": str(d.pages) if d.pages else "", "Chunks": d.num_chunks,
        } for d in docs]),
        hide_index=True, width="stretch",
    )
    names = {d.filename: d for d in docs}
    pick = st.selectbox("Inspect how a document was chunked", list(names))
    doc = names[pick]
    for c in engine.document_chunks(doc.doc_id):
        page = f" · p. {c.page_start}" if c.page_start else ""
        with st.expander(f"#{c.ordinal} · {' > '.join(c.section_path) or '(introduction)'}{page} · {c.word_count} words"):
            st.text(c.text)
    if st.button(f"Remove {pick} from the index"):
        engine.delete_document(doc.doc_id)
        st.rerun()


def evaluation_tab() -> None:
    path = config.REPORTS_DIR / "metrics.json"
    if not path.exists():
        st.info("No evaluation results yet. Run `python -m eval.run_eval` to generate them.")
        return
    m = json.loads(path.read_text(encoding="utf-8"))
    ret = m.get("retrieval", {}).get("hybrid", {}).get("overall", {})
    beh = m.get("behaviour", {})
    cols = st.columns(5)
    cols[0].metric("Recall@5 (hybrid)", f"{ret.get('recall@5', 0):.2f}")
    cols[1].metric("MRR (hybrid)", f"{ret.get('mrr', 0):.2f}")
    cols[2].metric("Status accuracy", f"{beh.get('status_accuracy', 0):.2f}")
    cols[3].metric("Not-found recall", f"{beh.get('not_found_recall', 0):.2f}")
    cols[4].metric("Conflict recall", f"{beh.get('conflict_recall', 0):.2f}")

    st.subheader("Retrieval ablation")
    rows = [{"mode": mode, **vals["overall"]} for mode, vals in m.get("retrieval", {}).items()]
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    if m.get("retrieval_by_category"):
        st.subheader("Hybrid retrieval by question category")
        st.dataframe(pd.DataFrame(m["retrieval_by_category"]).round(3), hide_index=True, width="stretch")
    if beh:
        st.subheader("Answer behaviour")
        st.dataframe(pd.DataFrame([{k: v for k, v in beh.items() if not isinstance(v, dict)}]).round(3),
                     hide_index=True, width="stretch")
        if beh.get("confusion"):
            st.subheader("Status confusion matrix (rows = expected)")
            st.dataframe(pd.DataFrame(beh["confusion"]).fillna(0).astype(int).T, width="stretch")
    report = config.REPORTS_DIR / "eval_report.md"
    if report.exists():
        with st.expander("Full evaluation report"):
            st.markdown(esc(report.read_text(encoding="utf-8")))


def main() -> None:
    sidebar()
    st.title("Enterprise Documentation Q&A")
    st.caption("Grounded answers from your manuals, FAQs, policies and release notes, with a citation on every "
               "claim, visible conflicts between sources, and an honest \"not found\" when the documents don't say.")
    ask, docs, evaluation = st.tabs(["💬 Ask", "📄 Documents", "📊 Evaluation"])
    with ask:
        ask_tab()
    with docs:
        documents_tab()
    with evaluation:
        evaluation_tab()


main()
