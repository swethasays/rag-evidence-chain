"""
data/storage/local.py

Local filesystem storage implementation.

Stores contract documents as UTF-8 text files under LOCAL_DATA_PATH.
Default backend for development — no cloud credentials required.

Swap to gcp.py or s3.py for production by setting STORAGE_TYPE in config.py.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import LOCAL_DATA_PATH
from data.storage.base import BaseStorage
from observability.logging import get_logger

logger = get_logger(__name__)


class LocalStorage(BaseStorage):
    """
    Stores contracts as plain text files on the local filesystem.

    Each contract is saved as {base_path}/{contract_id}.txt.
    Directory is created automatically on first use.

    Args:
        base_path: Root directory for contract files (default from config)
    """

    def __init__(self, base_path: str = LOCAL_DATA_PATH):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        logger.info("LocalStorage ready — base_path=%s", base_path)

    def _path(self, contract_id: str) -> str:
        return os.path.join(self.base_path, f"{contract_id}.txt")

    def save(self, contract_id: str, text: str) -> None:
        """Write contract text to disk, overwriting if it exists."""
        with open(self._path(contract_id), "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("Saved contract %s to local storage.", contract_id)

    def load(self, contract_id: str) -> str:
        """Read contract text from disk."""
        path = self._path(contract_id)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Contract '{contract_id}' not found at '{path}'."
            )
        with open(path, encoding="utf-8") as f:
            return f.read()

    def exists(self, contract_id: str) -> bool:
        """Return True if the contract file exists on disk."""
        return os.path.exists(self._path(contract_id))

    def delete(self, contract_id: str) -> None:
        """Delete the contract file from disk."""
        path = self._path(contract_id)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Contract '{contract_id}' not found at '{path}'."
            )
        os.remove(path)
        logger.info("Deleted contract %s from local storage.", contract_id)
