"""
data/storage/

Storage abstraction layer for contract documents.

Backends:
    local  → LocalStorage  (dev, default)
    gcp    → GCPStorage    (production)
    s3     → S3Storage     (production)

Swap by changing STORAGE_TYPE in config.py.
"""