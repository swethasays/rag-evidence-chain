import os
from dotenv import load_dotenv

load_dotenv()


# ── DATABASE ─────────────────────────────────────
DB_PATH = "data/contracts.db"

# ── NVIDIA API ───────────────────────────────────
NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY")
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"

# ── LLM ──────────────────────────────────────────
LLM_MODEL       = "meta/llama-3.3-70b-instruct"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS  = 1500

# ── EVALUATION ───────────────────────────────────────────
# Deliberately a different model family from LLM_MODEL to avoid self-grading bias.
# LLM_MODEL is Meta LLaMA — judge uses Google Gemma 3 (different architecture,
# different pre-training) so evaluation is truly independent.
LLM_JUDGE_MODEL = "google/gemma-3-4b-it"

# ── EMBEDDINGS ───────────────────────────────────
# nv-embedqa-e5-v5: 1024-dim retrieval model
# Use input_type="passage" for chunks, input_type="query" for questions
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
EMBEDDING_DIM   = 1024

# ── VECTOR STORE ─────────────────────────────────
VECTOR_STORE_TYPE = "faiss"  # swap to "pinecone" in production
FAISS_INDEX_PATH = "data/faiss.index"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "rag-evidence-chain"

# ── STORAGE ──────────────────────────────────────
STORAGE_TYPE = "local"  # swap to "gcp" or "s3" in production
LOCAL_DATA_PATH = "data/contracts"
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

# ── RETRIEVAL ────────────────────────────────────
TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 5
MIN_CONFIDENCE_SCORE = 0.5

# ── CHUNKING ─────────────────────────────────────
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# ── REDIS CACHE ──────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = 3600  # 1 hour

# ── LANGSMITH ────────────────────────────────────
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = "rag-evidence-chain"

# ── WANDB ────────────────────────────────────────
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = "rag-evidence-chain"

# ── API ──────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
RATE_LIMIT = "10/minute"
API_KEY = os.getenv("API_KEY")  # if unset, auth is skipped (dev mode)

# ── PIPELINE MESSAGES ────────────────────────────
# Single source of truth — used in graph.py (appended) and api/main.py (stripped)
HUMAN_REVIEW_FLAG = (
    "[⚠️ This answer has been flagged for human review "
    "due to low confidence scores.]"
)

# ── EVALUATION WEIGHTS ───────────────────────────────────
# Weights for overall score calculation
# Must sum to 1.0
# Retrieval score is only meaningful when ground truth exists in DuckDB.
# When ground truth IS found, retrieval is weighted alongside faithfulness/relevance.
# When ground truth is NOT found, retrieval defaults to 0.5 (noise) so it is
# excluded and faithfulness/relevance are renormalized to fill the full weight.
EVAL_WEIGHT_RETRIEVAL    = 0.3
EVAL_WEIGHT_FAITHFULNESS = 0.4
EVAL_WEIGHT_RELEVANCE    = 0.3
