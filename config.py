"""Central configuration.

Thresholds are tuned from eval output (eval/reports/metrics.json). bge cosine
scores are compressed (unrelated text often scores 0.4-0.6), so T_LOW sits
between the top scores of unanswerable and answerable questions.
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
CORPUS_DIR = ROOT / "eval" / "corpus"
REPORTS_DIR = ROOT / "eval" / "reports"
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

# ---- confidence gate + conflicts
T_LOW = 0.60
CONF_HIGH = 0.70
CONF_MEDIUM = 0.45
CONFLICT_SIM = 0.75
CLARIFY_MARGIN = 0.03

# ---- LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # gemini | openai
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# tried in order when the primary model is overloaded or times out
GEMINI_FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash,gemini-flash-lite-latest").split(",") if m.strip()]
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")
LLM_TIMEOUT_S = 30
LLM_RETRIES = 1  # per model
LLM_COOLDOWN_S = 600  # skip a model this long after it fails
