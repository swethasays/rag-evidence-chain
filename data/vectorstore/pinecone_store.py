"""
data/vectorstore/pinecone_store.py

Pinecone vector store implementing BaseVectorStore.

Stores embeddings in Pinecone for production deployments that need
managed, replicated, and scalable vector search beyond FAISS limits.

Requires:
    PINECONE_API_KEY    — API key from console.pinecone.io
    PINECONE_INDEX_NAME — name of the index (created automatically if absent)

Swap to this backend by setting VECTOR_STORE_TYPE=pinecone in config.py.

Install dependency:
    pip install pinecone-client
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import EMBEDDING_DIM, PINECONE_API_KEY, PINECONE_INDEX_NAME
from data.vectorstore.base import BaseVectorStore
from observability.logging import get_logger

logger = get_logger(__name__)

_UPSERT_BATCH = 100  # Pinecone recommends batches of 100


class PineconeStore(BaseVectorStore):
    """
    Pinecone-backed vector store for production deployments.

    The index is created automatically on first use if it does not exist.
    Uses cosine similarity to match the normalised embeddings produced
    by all-MiniLM-L6-v2.

    save() and load() are no-ops because Pinecone persists data server-side.

    Args:
        api_key:    Pinecone API key (default from config)
        index_name: Index to connect to (default from config)
        dim:        Embedding dimension (default from config)
    """

    def __init__(
        self,
        api_key: str = PINECONE_API_KEY,
        index_name: str = PINECONE_INDEX_NAME,
        dim: int = EMBEDDING_DIM,
    ):
        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY is not set. "
                "Add PINECONE_API_KEY=your-key to your .env file."
            )
        if not index_name:
            raise ValueError(
                "PINECONE_INDEX_NAME is not set. "
                "Add PINECONE_INDEX_NAME=your-index to config.py."
            )
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError:
            raise ImportError(
                "pinecone-client is not installed. "
                "Run: pip install pinecone-client"
            )

        self._dim = dim
        pc = Pinecone(api_key=api_key)

        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            logger.info(
                "Pinecone index '%s' not found — creating (dim=%d, metric=cosine).",
                index_name, dim,
            )
            pc.create_index(
                name=index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self._index = pc.Index(index_name)
        logger.info("PineconeStore ready — index=%s.", index_name)

    def add(self, ids: list[str], embeddings: np.ndarray) -> None:
        """
        Upsert vectors into Pinecone in batches of 100.

        Args:
            ids:        String ID for each vector
            embeddings: Float array of shape (n, dim)
        """
        if len(ids) != len(embeddings):
            raise ValueError(
                f"ids length ({len(ids)}) must match embeddings rows "
                f"({len(embeddings)})"
            )
        vectors = embeddings.tolist()
        for i in range(0, len(ids), _UPSERT_BATCH):
            batch = list(zip(ids[i:i + _UPSERT_BATCH], vectors[i:i + _UPSERT_BATCH]))
            self._index.upsert(vectors=batch)
        logger.info("Upserted %d vectors to Pinecone.", len(ids))

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """
        Query Pinecone for nearest neighbours.

        Args:
            query: 1-D float array of length dim
            top_k: Number of results to return

        Returns:
            List of (id, score) sorted by score descending.
        """
        response = self._index.query(
            vector=query.tolist(),
            top_k=top_k,
            include_metadata=False,
        )
        return [
            (match["id"], float(match["score"]))
            for match in response["matches"]
        ]

    def save(self, path: str) -> None:
        """No-op — Pinecone persists data server-side automatically."""
        logger.info(
            "PineconeStore.save() called — no-op (Pinecone is a managed service)."
        )

    def load(self, path: str) -> None:
        """No-op — Pinecone persists data server-side automatically."""
        logger.info(
            "PineconeStore.load() called — no-op (Pinecone is a managed service)."
        )

    def count(self) -> int:
        """Return the total number of vectors in the Pinecone index."""
        stats = self._index.describe_index_stats()
        return stats["total_vector_count"]
