"""
data/storage/s3.py

AWS S3 storage implementation.

Stores contract documents as objects in an S3 bucket.
Requires AWS_BUCKET_NAME in config.py and valid AWS credentials
(env vars AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, or an IAM role).

Swap to this backend by setting STORAGE_TYPE=s3 in config.py.

Install dependency:
    pip install boto3
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import AWS_BUCKET_NAME
from data.storage.base import BaseStorage
from observability.logging import get_logger

logger = get_logger(__name__)


class S3Storage(BaseStorage):
    """
    Stores contracts as objects in an AWS S3 bucket.

    Each contract is stored at {prefix}{contract_id}.txt.
    Credentials are read from the environment — either
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars
    or an attached IAM role (recommended for EC2/ECS).

    Args:
        bucket_name: S3 bucket name (default from config)
        prefix:      Object key prefix, e.g. "contracts/"
    """

    def __init__(
        self,
        bucket_name: str = AWS_BUCKET_NAME,
        prefix: str = "contracts/",
    ):
        if not bucket_name:
            raise ValueError(
                "AWS_BUCKET_NAME is not set. "
                "Add AWS_BUCKET_NAME=your-bucket to your .env file."
            )
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError(
                "boto3 is not installed. Run: pip install boto3"
            )

        self.bucket_name = bucket_name
        self.prefix = prefix
        self._s3 = boto3.client("s3")
        self._ClientError = ClientError
        logger.info(
            "S3Storage ready — bucket=%s, prefix=%s", bucket_name, prefix
        )

    def _key(self, contract_id: str) -> str:
        return f"{self.prefix}{contract_id}.txt"

    def save(self, contract_id: str, text: str) -> None:
        """Upload contract text to S3."""
        self._s3.put_object(
            Bucket=self.bucket_name,
            Key=self._key(contract_id),
            Body=text.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )
        logger.info("Saved contract %s to S3.", contract_id)

    def load(self, contract_id: str) -> str:
        """Download contract text from S3."""
        try:
            response = self._s3.get_object(
                Bucket=self.bucket_name,
                Key=self._key(contract_id),
            )
            return response["Body"].read().decode("utf-8")
        except self._ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                raise FileNotFoundError(
                    f"Contract '{contract_id}' not found in S3 bucket "
                    f"'{self.bucket_name}'."
                )
            raise

    def exists(self, contract_id: str) -> bool:
        """Return True if the object exists in S3."""
        try:
            self._s3.head_object(
                Bucket=self.bucket_name,
                Key=self._key(contract_id),
            )
            return True
        except self._ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise

    def delete(self, contract_id: str) -> None:
        """Delete the contract object from S3."""
        if not self.exists(contract_id):
            raise FileNotFoundError(
                f"Contract '{contract_id}' not found in S3 bucket "
                f"'{self.bucket_name}'."
            )
        self._s3.delete_object(Bucket=self.bucket_name, Key=self._key(contract_id))
        logger.info("Deleted contract %s from S3.", contract_id)
