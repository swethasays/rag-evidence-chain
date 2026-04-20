"""
data/vectorstore/base.py

Abstract base class for vector store backends.

Defines the interface all vector store implementations must follow.
Swap backends by changing VECTOR_STORE_TYPE in config.py:
    "faiss"    → faiss_store.py   (default, dev)
    "pinecone" → pinecone_store.py (production)
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseVectorStore(ABC):
    """Interface for vector store backends."""

    @abstractmethod
    def add(self, ids: list[str], embeddings: np.ndarray) -> None:
        """Add embeddings to the store."""
        ...

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """
        Search for nearest neighbours.

        Returns:
            List of (id, score) tuples sorted by score descending.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the index from disk."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return number of vectors in the store."""
        ...