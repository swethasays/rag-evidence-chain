import os
from dotenv import load_dotenv

load_dotenv()


# ── DATABASE ─────────────────────────────────────
DB_PATH = "data/contracts.db"

# ── LLM ──────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"

# ── EMBEDDINGS ───────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

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
MIN_CONFIDENCE_SCORE = 0.7

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
