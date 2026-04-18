"""
agents/retrieval.py

Agent 1 — Retrieval

Responsible for finding the most relevant chunks from the database
for a given user question. Uses hybrid search (FAISS + BM25) followed
by cross-encoder re-ranking to return the top-k most relevant chunks.

Flow:
    question
        → embed question
        → FAISS semantic search  (top 20)
        → BM25 keyword search    (top 20)
        → merge and deduplicate  (up to 40)
        → cross-encoder re-rank  (top 5)
        → return top 5 chunks with metadata
"""

import logging
import os
import sys

import duckdb
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DB_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    TOP_K_RETRIEVAL,
    TOP_K_RERANK,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cross-encoder model for re-ranking
# Reads question + chunk together for more accurate scoring
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def load_embedding_model() -> SentenceTransformer:
    """
    Load the sentence embedding model.

    We use all-MiniLM-L6-v2 — small, fast, good quality.
    384 dimensions means each chunk becomes 384 numbers.
    Downloads automatically on first run (~80MB).
    """
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded.")
    return model


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Convert a list of strings into a 2D numpy array of embeddings.

    Args:
        model:  The loaded SentenceTransformer model
        texts:  List of strings to embed

    Returns:
        numpy array of shape (len(texts), 384)
        Each row is one embedding vector.
    """
    embeddings = model.encode(
        texts,
        batch_size=64,          # process 64 chunks at a time
        show_progress_bar=True, # shows progress for large batches
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )
    return embeddings.astype(np.float32)  # FAISS requires float32


# ---------------------------------------------------------------------------
# FAISS Index
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index from a matrix of embeddings.

    We use IndexFlatIP (Inner Product) which is equivalent to
    cosine similarity when embeddings are L2-normalized.

    Args:
        embeddings: numpy array of shape (n_chunks, 384)

    Returns:
        A searchable FAISS index
    """
    n_chunks, dim = embeddings.shape
    logger.info("Building FAISS index for %d chunks (dim=%d)", n_chunks, dim)

    # IndexFlatIP = exact search using inner product (cosine similarity)
    # For production with millions of vectors, use IndexIVFFlat instead
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logger.info("FAISS index built with %d vectors.", index.ntotal)
    return index


def save_faiss_index(index: faiss.Index, path: str = FAISS_INDEX_PATH) -> None:
    """Save FAISS index to disk so we don't rebuild it every time."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)
    logger.info("FAISS index saved to %s", path)


def load_faiss_index(path: str = FAISS_INDEX_PATH) -> faiss.Index:
    """Load a previously saved FAISS index from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"FAISS index not found at '{path}'.\n"
            "Run build_vector_index() first."
        )
    logger.info("Loading FAISS index from %s", path)
    return faiss.read_index(path)


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------

def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """
    Build a BM25 keyword search index from the chunk texts.

    BM25 works on tokens (words) so we do a simple whitespace
    tokenization here. For production you'd use a proper tokenizer.

    Args:
        chunks: List of chunk dicts with 'text' key

    Returns:
        A searchable BM25Okapi index
    """
    logger.info("Building BM25 index for %d chunks", len(chunks))

    # Tokenize: lowercase and split on whitespace
    # "Termination of Agreement" → ["termination", "of", "agreement"]
    tokenized = [chunk["text"].lower().split() for chunk in chunks]

    index = BM25Okapi(tokenized)
    logger.info("BM25 index built.")
    return index


# ---------------------------------------------------------------------------
# Load chunks from database
# ---------------------------------------------------------------------------

def load_chunks_from_db(db_path: str = DB_PATH) -> list[dict]:
    """
    Load all chunks from DuckDB into memory.

    We need them in memory for BM25 (which requires all texts upfront)
    and to map FAISS result indices back to chunk metadata.

    Returns:
        List of dicts: {id, contract_id, chunk_index, text, char_start, char_end}
    """
    conn = duckdb.connect(db_path)

    rows = conn.execute("""
        SELECT
            c.id,
            c.contract_id,
            c.chunk_index,
            c.text,
            c.char_start,
            c.char_end,
            ct.title as contract_title
        FROM chunks c
        JOIN contracts ct ON c.contract_id = ct.id
        ORDER BY c.contract_id, c.chunk_index
    """).fetchall()

    conn.close()

    chunks = [
        {
            "id": row[0],
            "contract_id": row[1],
            "chunk_index": row[2],
            "text": row[3],
            "char_start": row[4],
            "char_end": row[5],
            "contract_title": row[6],
        }
        for row in rows
    ]

    logger.info("Loaded %d chunks from database.", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Vector index builder (run once)
# ---------------------------------------------------------------------------
def build_vector_index(
    model: SentenceTransformer,
    chunks: list[dict],
) -> faiss.Index:
    """
    One-time setup: embed all chunks and build the FAISS index.

    This takes ~2-3 minutes for 7,764 chunks.
    The index is saved to disk so subsequent runs load it instantly.

    Returns:
        faiss_index
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(model, texts)
    index = build_faiss_index(embeddings)
    save_faiss_index(index)
    return index

# ---------------------------------------------------------------------------
# Retrieval Agent
# ---------------------------------------------------------------------------

class RetrievalAgent:
    """
    Agent 1 — finds the most relevant chunks for a given question.

    Uses hybrid search (FAISS semantic + BM25 keyword) followed by
    cross-encoder re-ranking for maximum accuracy.

    Usage:
        agent = RetrievalAgent()
        results = agent.search("What is the termination clause?")
    """

    def __init__(self):
        logger.info("Initialising RetrievalAgent...")

        # Load all chunks from DuckDB into memory
        self.chunks = load_chunks_from_db()

        # Load embedding model for encoding questions
        self.embed_model = load_embedding_model()

        # Load or build FAISS index
        if os.path.exists(FAISS_INDEX_PATH):
            self.faiss_index = load_faiss_index()

            # Guard — verify FAISS and chunks are in sync
            # If ingest.py was re-run without rebuilding FAISS,
            # the index would return wrong chunks silently
            if self.faiss_index.ntotal != len(self.chunks):
                logger.warning(
                    "FAISS index is out of sync with database! "
                    "Index has %d vectors but database has %d chunks. "
                    "Rebuilding FAISS index...",
                    self.faiss_index.ntotal,
                    len(self.chunks),
                )
                self.faiss_index = build_vector_index(
                    self.embed_model, self.chunks
                )
        else:
            logger.info("No FAISS index found — building from scratch...")
            # Pass already-loaded model and chunks — avoids loading twice
            self.faiss_index = build_vector_index(self.embed_model, self.chunks)

        # Build BM25 index (fast, in-memory)
        self.bm25_index = build_bm25_index(self.chunks)

        # Load cross-encoder for re-ranking
        logger.info("Loading cross-encoder re-ranker...")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        logger.info("RetrievalAgent ready. %d chunks indexed.", len(self.chunks))

    def _semantic_search(self, question: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """
        Search using FAISS (semantic / meaning-based search).

        Embeds the question and finds the chunks whose embeddings
        are most similar using cosine similarity.

        Returns list of chunk dicts with added 'semantic_score'.
        """
        # Embed the question — same model as chunks
        q_embedding = self.embed_model.encode(
            [question],
            normalize_embeddings=True,
        ).astype(np.float32)

        # Search FAISS — returns distances and indices
        scores, indices = self.faiss_index.search(q_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            chunk = self.chunks[idx].copy()
            chunk["semantic_score"] = float(score)
            results.append(chunk)

        return results

    def _keyword_search(self, question: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """
        Search using BM25 (keyword / exact-match search).

        Tokenizes the question and scores chunks by term frequency
        and rarity. Good for exact legal terms like "ROFR", "indemnify".

        Returns list of chunk dicts with added 'bm25_score'.
        """
        # Tokenize question same way as we tokenized chunks
        tokens = question.lower().split()

        # BM25 scores all chunks — higher is better
        scores = self.bm25_index.get_scores(tokens)

        # Get top_k indices sorted by score descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] == 0:  # skip chunks with zero relevance
                continue
            chunk = self.chunks[idx].copy()
            chunk["bm25_score"] = float(scores[idx])
            results.append(chunk)

        return results

    def _merge_results(
        self,
        semantic_results: list[dict],
        keyword_results: list[dict],
    ) -> list[dict]:
        """
        Merge semantic and keyword results, removing duplicates.

        A chunk can appear in both result sets. We keep it once
        and note that it was found by both methods — a strong signal.

        Returns deduplicated list with 'found_by' metadata.
        """
        seen_ids = set()
        merged = []

        # Add semantic results first
        for chunk in semantic_results:
            seen_ids.add(chunk["id"])
            chunk["found_by"] = "semantic"
            merged.append(chunk)

        # Add keyword results — skip duplicates
        for chunk in keyword_results:
            if chunk["id"] not in seen_ids:
                seen_ids.add(chunk["id"])
                chunk["found_by"] = "keyword"
                merged.append(chunk)
            else:
                # Found by both — mark it (strong relevance signal)
                for m in merged:
                    if m["id"] == chunk["id"]:
                        m["found_by"] = "both"
                        break

        return merged

    def _rerank(self, question: str, candidates: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
        """
        Re-rank candidates using a cross-encoder model.

        Unlike embedding similarity (which encodes question and chunk
        separately), a cross-encoder reads them TOGETHER — much more
        accurate for relevance scoring.

        Args:
            question:   The user's question
            candidates: List of candidate chunks to re-rank
            top_k:      How many to return after re-ranking

        Returns:
            Top-k chunks sorted by cross-encoder score descending.
        """
        if not candidates:
            return []

        # Cross-encoder takes [question, chunk_text] pairs
        pairs = [[question, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)

        # Add rerank score to each candidate
        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = float(score)

        # Sort by rerank score descending and return top_k
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]

    

    def search(
        self,
        question: str,
        filters: dict = None,
    ) -> list[dict]:
        """
        Main entry point — find the most relevant chunks for a question.

        Args:
            question: The user's natural language question
            filters:  Optional dict to narrow search scope
                      Supported keys:
                        - contract_id    : only search this contract
                        - contract_title : only search this contract title

        Returns:
            Top-5 chunk dicts with scores and metadata.
            Returns empty list if no chunks match the filters.
        """
        logger.info("Searching for: '%s'", question[:80])

        if filters:
            logger.info("Applying filters: %s", filters)

        semantic = self._semantic_search(question)
        keyword  = self._keyword_search(question)
        merged   = self._merge_results(semantic, keyword)

        # Apply metadata filters AFTER retrieval
        if filters:
            merged = self._apply_filters(merged, filters)

        # Guard — if filters eliminated everything, warn and return early
        if not merged:
            logger.warning(
                "No chunks remain after filtering. "
                "Filters may be too strict: %s", filters
            )
            return []

        results = self._rerank(question, merged)

        logger.info(
            "Retrieved %d chunks (semantic=%d, keyword=%d, after rerank=%d)",
            len(merged), len(semantic), len(keyword), len(results),
        )

        return results
    
    def _apply_filters(
        self,
        chunks: list[dict],
        filters: dict,
    ) -> list[dict]:
        """
        Filter retrieved chunks by metadata fields.

        Called after hybrid search so we filter from already-relevant
        results rather than limiting the search space upfront.

        Args:
            chunks:  Candidate chunks from hybrid search
            filters: Dict of field → value to filter by

        Returns:
            Filtered list of chunks
        """
        filtered = chunks

        if "contract_id" in filters:
            filtered = [
                c for c in filtered
                if c["contract_id"] == filters["contract_id"]
            ]

        if "contract_title" in filters:
            filtered = [
                c for c in filtered
                if filters["contract_title"].lower()
                in c["contract_title"].lower()
            ]

        logger.info(
            "After filtering: %d → %d chunks",
            len(chunks), len(filtered),
        )

        return filtered

# ---------------------------------------------------------------------------
# Entry point — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test — search for a termination clause
    agent = RetrievalAgent()

    question = "What is the termination clause?"
    results = agent.search(question)

    print(f"\nQuestion: {question}")
    print(f"Top {len(results)} results:\n")

    for i, chunk in enumerate(results, 1):
        print(f"{'─' * 60}")
        print(f"Rank {i} | Score: {chunk['rerank_score']:.3f} | Found by: {chunk['found_by']}")
        print(f"Contract: {chunk['contract_title'][:60]}")
        print(f"Text: {chunk['text'][:200]}...")
        print()

    # Test metadata filtering — same question, scoped to one contract
    print("\n" + "═" * 60)
    print("FILTERED SEARCH — WHITESMOKE contract only")
    print("═" * 60)

    filtered_results = agent.search(
        question,
        filters={"contract_title": "WHITESMOKE"}
    )

    print(f"\nQuestion: {question}")
    print(f"Top {len(filtered_results)} results:\n")

    for i, chunk in enumerate(filtered_results, 1):
        print(f"{'─' * 60}")
        print(f"Rank {i} | Score: {chunk['rerank_score']:.3f} | Found by: {chunk['found_by']}")
        print(f"Contract: {chunk['contract_title'][:60]}")
        print(f"Text: {chunk['text'][:200]}...")
        print()

    # Test empty filter — should warn and return []
    print("\n" + "═" * 60)
    print("EMPTY FILTER TEST — nonexistent contract")
    print("═" * 60)

    empty_results = agent.search(
        question,
        filters={"contract_title": "NONEXISTENT_CONTRACT_XYZ"}
    )
    print(f"Results returned: {len(empty_results)}")