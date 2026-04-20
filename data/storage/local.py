"""
data/storage/local.py

Local filesystem storage implementation.

Stores contract documents as text files under LOCAL_DATA_PATH.
Used in development — no cloud credentials required.

Swap to gcp.py or s3.py for production by changing
STORAGE_TYPE in config.py.
"""

# Implementation planned for production deployment.
# Currently ingest.py stores contracts directly in DuckDB.