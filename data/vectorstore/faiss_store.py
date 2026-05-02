"""
data/vectorstore/faiss_store.py

FAISS vector store implementing BaseVectorStore.

Wraps a faiss.IndexFlatIP behind the shared interface so retrieval.py
can swap to Pinecone by changing one line in config.py.

Uses inner product (IP) search, which equals cosine similarity for
L2-normalised embeddings — the same approach as retrieval.py today.

IDs are persisted alongside the index in a sidecar .ids file so the
integer↔string mapping survives a save/load cycle.
"""

import os
import sys

import faiss
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import EMBEDDING_DIM
from data.vectorstore.base import BaseVectorStore
from observability.logging import get_logger

logger = get_logger(__name__)


class FaissStore(BaseVectorStore):
    """
    Local FAISS vector store backed by IndexFlatIP (exact search).

    Suitable for up to ~1M vectors. Switch to PineconeStore for
    larger datasets or managed, replicated storage.

    Args:
        dim: Embedding dimension (default from config, must match the
             model used to produce the vectors)
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []
        logger.info("FaissStore initialized (dim=%d).", dim)

    def add(self, ids: list[str], embeddings: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            ids:        String ID for each vector (1-to-1 with embeddings)
            embeddings: Float32 array of shape (n, dim)
        """
        if len(ids) != len(embeddings):
            raise ValueError(
                f"ids length ({len(ids)}) must match embeddings rows "
                f"({len(embeddings)})"
            )
        self._index.add(embeddings.astype(np.float32))
        self._ids.extend(ids)
        logger.info(
            "Added %d vectors. Total in store: %d.", len(ids), self._index.ntotal
        )

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """
        Find nearest neighbours by inner product (cosine similarity).

        Args:
            query: 1-D float array of length dim
            top_k: Number of results to return

        Returns:
            List of (id, score) sorted by score descending.
            Empty list if the store has no vectors.
        """
        if self._index.ntotal == 0:
            return []

        q = query.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._ids[idx], float(score)))
        return results

    def save(self, path: str) -> None:
        """
        Persist the FAISS index and ID map to disk.

        Saves two files:
            {path}      — the FAISS binary index
            {path}.ids  — newline-delimited string IDs
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        faiss.write_index(self._index, path)
        with open(f"{path}.ids", "w", encoding="utf-8") as f:
            f.write("\n".join(self._ids))
        logger.info(
            "FaissStore saved to %s (%d vectors).", path, self._index.ntotal
        )

    def load(self, path: str) -> None:
        """
        Load a previously saved FAISS index and ID map from disk.

        Args:
            path: Path used when save() was called
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"FAISS index not found at '{path}'.")
        ids_path = f"{path}.ids"
        if not os.path.exists(ids_path):
            raise FileNotFoundError(
                f"FAISS ID map not found at '{ids_path}'. "
                "Re-run ingestion to rebuild."
            )
        self._index = faiss.read_index(path)
        with open(ids_path, encoding="utf-8") as f:
            self._ids = [line.rstrip("\n") for line in f if line.strip()]
        logger.info(
            "FaissStore loaded from %s (%d vectors).", path, self._index.ntotal
        )

    def count(self) -> int:
        """Return the number of vectors currently in the store."""
        return self._index.ntotal
