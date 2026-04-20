"""
data/storage/gcp.py

Google Cloud Storage implementation.

Stores contract documents in a GCP bucket.
Requires GCP_BUCKET_NAME in config.py and
GOOGLE_APPLICATION_CREDENTIALS in environment.

Swap to this backend by setting STORAGE_TYPE=gcp in config.py.
"""

# Implementation planned for production deployment.