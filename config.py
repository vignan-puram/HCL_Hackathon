"""Central configuration. Owner: Member 1.

Thresholds are PLACEHOLDERS -- tune them from eval output. bge cosine scores are
compressed (unrelated text often scores 0.4-0.6): set T_LOW to the midpoint
between the max top score of unanswerable questions and the min of answerable ones.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# ---- paths (DATA_DIR is overridable so the eval runner uses its own index)
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
LLM_CACHE_DIR = DATA_DIR / "llm_cache"
DB_PATH = DATA_DIR / "app.db"
FIXTURE_QUERY_RESULT = ROOT / "eval" / "fixtures" / "sample_query_result.json"

# ---- ingestion
ALLOWED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".markdown", ".txt"}
MAX_FILE_MB = 20
CHUNK_TARGET_WORDS = 300
CHUNK_MAX_WORDS = 450
CHUNK_OVERLAP_WORDS = 40
MIN_SECTION_WORDS = 40

# ---- embeddings + retrieval
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DENSE_K = 20
BM25_K = 20
RRF_K = 60
TOP_K = 5
USE_RERANKER = False
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---- confidence gate + conflicts
T_LOW = 0.45
CONF_HIGH = 0.70
CONF_MEDIUM = 0.45
CONFLICT_SIM = 0.75

# ---- LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # gemini | groq
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
LLM_TIMEOUT_S = 30
LLM_RETRIES = 2

# ---- frontend
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
CLIENT_MODE = os.getenv("CLIENT_MODE", "HTTP")  # MOCK | HTTP | DIRECT
