"""
data/storage/base.py

Abstract base class for contract storage backends.

Defines the interface all storage implementations must follow.
Swap backends by changing STORAGE_TYPE in config.py:
    "local"  → local.py   (default, dev)
    "gcp"    → gcp.py     (production)
    "s3"     → s3.py      (production)
"""

from abc import ABC, abstractmethod


class BaseStorage(ABC):
    """Interface for contract document storage."""

    @abstractmethod
    def save(self, contract_id: str, text: str) -> None:
        """Save a contract document."""
        ...

    @abstractmethod
    def load(self, contract_id: str) -> str:
        """Load a contract document by ID."""
        ...

    @abstractmethod
    def exists(self, contract_id: str) -> bool:
        """Check if a contract exists."""
        ...

    @abstractmethod
    def delete(self, contract_id: str) -> None:
        """Delete a contract document."""
        ...