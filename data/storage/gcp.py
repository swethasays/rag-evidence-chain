"""
data/storage/gcp.py

Google Cloud Storage implementation.

Stores contract documents as objects in a GCS bucket.
Requires GCP_BUCKET_NAME in config.py and valid GCP credentials
(GOOGLE_APPLICATION_CREDENTIALS env var or Application Default Credentials).

Swap to this backend by setting STORAGE_TYPE=gcp in config.py.

Install dependency:
    pip install google-cloud-storage
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import GCP_BUCKET_NAME
from data.storage.base import BaseStorage
from observability.logging import get_logger

logger = get_logger(__name__)


class GCPStorage(BaseStorage):
    """
    Stores contracts as objects in a Google Cloud Storage bucket.

    Each contract is stored at {prefix}{contract_id}.txt.
    Credentials are read from the environment — either
    GOOGLE_APPLICATION_CREDENTIALS (path to service account key)
    or Application Default Credentials (gcloud auth).

    Args:
        bucket_name: GCS bucket name (default from config)
        prefix:      Object name prefix, e.g. "contracts/"
    """

    def __init__(
        self,
        bucket_name: str = GCP_BUCKET_NAME,
        prefix: str = "contracts/",
    ):
        if not bucket_name:
            raise ValueError(
                "GCP_BUCKET_NAME is not set. "
                "Add GCP_BUCKET_NAME=your-bucket to your .env file."
            )
        try:
            from google.cloud import storage as gcs
        except ImportError:
            raise ImportError(
                "google-cloud-storage is not installed. "
                "Run: pip install google-cloud-storage"
            )

        self.prefix = prefix
        client = gcs.Client()
        self.bucket = client.bucket(bucket_name)
        logger.info(
            "GCPStorage ready — bucket=%s, prefix=%s", bucket_name, prefix
        )

    def _blob_name(self, contract_id: str) -> str:
        return f"{self.prefix}{contract_id}.txt"

    def save(self, contract_id: str, text: str) -> None:
        """Upload contract text to GCS."""
        blob = self.bucket.blob(self._blob_name(contract_id))
        blob.upload_from_string(text, content_type="text/plain; charset=utf-8")
        logger.info("Saved contract %s to GCS.", contract_id)

    def load(self, contract_id: str) -> str:
        """Download contract text from GCS."""
        blob = self.bucket.blob(self._blob_name(contract_id))
        if not blob.exists():
            raise FileNotFoundError(
                f"Contract '{contract_id}' not found in GCS bucket."
            )
        return blob.download_as_text(encoding="utf-8")

    def exists(self, contract_id: str) -> bool:
        """Return True if the object exists in GCS."""
        return self.bucket.blob(self._blob_name(contract_id)).exists()

    def delete(self, contract_id: str) -> None:
        """Delete the contract object from GCS."""
        blob = self.bucket.blob(self._blob_name(contract_id))
        if not blob.exists():
            raise FileNotFoundError(
                f"Contract '{contract_id}' not found in GCS bucket."
            )
        blob.delete()
        logger.info("Deleted contract %s from GCS.", contract_id)
