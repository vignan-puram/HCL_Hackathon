# Enterprise Documentation Q&A Assistant

HCLTech Future Ready AI Engineer Hackathon -- Use Case 2. Upload PDF, HTML, Markdown
and TXT documents, ask natural-language questions, and get concise answers grounded
only in retrieved passages, with per-claim citations, visible conflicts between
sources, and a clear not-found / clarify fallback.

> Work in progress. Member 3 owns the full README (formats, indexing and retrieval
> approach, evaluation summary, assumptions, trade-offs). Design reference:
> [`docs/blueprint/engineering_blueprint.pdf`](docs/blueprint/engineering_blueprint.pdf);
> problem brief: [`docs/blueprint/usecase2_brief.docx`](docs/blueprint/usecase2_brief.docx).

## Setup (Windows, Python 3.12)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
copy .env.example .env   # add GEMINI_API_KEY / GROQ_API_KEY
python -m scripts.check_llm
pytest
```

## Run (from repo root)

```powershell
uvicorn backend.main:app --reload --port 8000
streamlit run frontend/app.py
```

## Ownership

| Member | Owns |
|---|---|
| M1 -- AI/RAG | `rag/`, `config.py`, `tests/test_retrieval.py`, `scripts/check_llm.py`; gatekeeper of `common/schemas.py` |
| M2 -- Backend + ingestion | `backend/`, `ingestion/`, `tests/test_ingestion.py`, `tests/test_api.py`, `tests/test_contract.py`, `scripts/smoke_test.py`, `requirements.txt`, `.gitignore` |
| M3 -- Frontend, eval, demo | `frontend/`, `eval/`, `README.md`, `docs/` |

`common/schemas.py` is frozen: additive optional fields only, agreed by two members.
`rag/engine.py` and `ingestion/pipeline.py` are the only cross-team imports.
