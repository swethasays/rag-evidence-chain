"""
data/ingest.py

Loads the CUAD legal contracts dataset, chunks each contract into
overlapping text windows, and stores everything in a local DuckDB
database that the retrieval agent will query later.

Usage:
    python data/ingest.py
"""

import json
import logging
import os
import sys
import hashlib 

import duckdb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_PATH, LOCAL_DATA_PATH, STORAGE_TYPE
from data.chunker import Chunker
from observability.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CUAD_PATH = os.path.join(LOCAL_DATA_PATH, "CUAD_v1.json")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def setup_database(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    """
    Create the DuckDB database and the three tables we need.

    Returns an open connection intentionally — caller is responsible
    for closing it. run_ingestion() wraps this in try/finally.
    """
    conn = duckdb.connect(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS contracts (
            id         VARCHAR PRIMARY KEY,
            title      VARCHAR,
            text       VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id          VARCHAR PRIMARY KEY,
            contract_id VARCHAR,
            chunk_index INTEGER,
            text        VARCHAR,
            char_start  INTEGER,
            char_end    INTEGER,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ground_truth (
            id           VARCHAR PRIMARY KEY,
            contract_id  VARCHAR,
            question     VARCHAR,
            answer       VARCHAR,
            answer_start INTEGER
        )
    """)

    logger.info("Database ready at %s", db_path)
    return conn


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_cuad(path: str = CUAD_PATH) -> dict:
    """
    Read the CUAD JSON file from disk.

    The file must already be downloaded and placed at:
        data/contracts/CUAD_v1.json

    Returns the parsed JSON dict with a top-level 'data' key
    containing a list of contracts.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CUAD file not found at '{path}'.\n"
            "Download CUAD_v1.json from Kaggle and place it in data/contracts/."
        )

    logger.info("Loading CUAD from %s", path)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    total = len(data["data"])
    logger.info("Found %d contracts in dataset", total)
    return data


# Module-level Chunker instance — created once, reused for all contracts
_chunker = Chunker()


# ---------------------------------------------------------------------------
# Storage backend factory
# ---------------------------------------------------------------------------

def get_storage():
    """
    Return the configured storage backend for raw contract text.

    Backend is selected by STORAGE_TYPE in config.py:
        "local"  → files under LOCAL_DATA_PATH  (default, no credentials needed)
        "gcp"    → Google Cloud Storage bucket  (requires GCP_BUCKET_NAME + credentials)
        "s3"     → AWS S3 bucket                (requires AWS_BUCKET_NAME + credentials)
    """
    if STORAGE_TYPE == "local":
        from data.storage.local import LocalStorage
        return LocalStorage()
    elif STORAGE_TYPE == "gcp":
        from data.storage.gcp import GCPStorage
        return GCPStorage()
    elif STORAGE_TYPE == "s3":
        from data.storage.s3 import S3Storage
        return S3Storage()
    else:
        raise ValueError(
            f"Unknown STORAGE_TYPE: '{STORAGE_TYPE}'. "
            "Set to 'local', 'gcp', or 's3' in config.py."
        )

# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def store_contract(conn: duckdb.DuckDBPyConnection, contract_id: str, title: str, text: str) -> None:
    """Insert or replace a contract row."""
    conn.execute(
        "INSERT OR REPLACE INTO contracts (id, title, text) VALUES (?, ?, ?)",
        [contract_id, title, text],
    )


def store_chunks(conn: duckdb.DuckDBPyConnection, contract_id: str, chunks: list[dict]) -> None:
    """Bulk-insert all chunks for a single contract."""
    rows = [
        (
            f"{contract_id}_chunk_{i}",
            contract_id,
            i,
            chunk["text"],
            chunk["char_start"],
            chunk["char_end"],
        )
        for i, chunk in enumerate(chunks)
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO chunks
            (id, contract_id, chunk_index, text, char_start, char_end)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def store_ground_truth(
    conn: duckdb.DuckDBPyConnection,
    contract_id: str,
    question: str,
    answer: str,
    answer_start: int,
) -> None:
    """Insert a single QA pair used later by the evaluation agent."""
    question_hash = hashlib.sha256(question.encode()).hexdigest()[:16] 
    gt_id = f"{contract_id}_{question_hash}"                          
    conn.execute(
        """
        INSERT OR REPLACE INTO ground_truth
            (id, contract_id, question, answer, answer_start)
        VALUES (?, ?, ?, ?, ?)
        """,
        [gt_id, contract_id, question, answer, answer_start],
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_ingestion(max_contracts: int = 50) -> tuple[int, int]:
    """
    End-to-end ingestion pipeline:

      1. Load CUAD JSON
      2. For each contract: chunk the text and store in DuckDB
      3. Store ground-truth QA pairs for the evaluation agent

    Args:
        max_contracts: How many contracts to process (default 50 for dev).
                       Set to 510 to process the full dataset.

    Returns:
        (contracts_processed, total_chunks)
    """
    os.makedirs(LOCAL_DATA_PATH, exist_ok=True)

    storage = get_storage()
    conn = setup_database()
    contracts_processed = 0
    total_chunks = 0

    try:
        data = load_cuad()

        for contract in data["data"][:max_contracts]:
            title = contract["title"]
            full_text = " ".join(p["context"] for p in contract["paragraphs"])
            contract_id = hashlib.sha256(full_text.encode()).hexdigest()

            # Persist raw text to the configured storage backend
            # (local files, GCS, or S3 depending on STORAGE_TYPE)
            storage.save(contract_id, full_text)

            store_contract(conn, contract_id, title, full_text)

            chunks = _chunker.chunk(full_text)
            store_chunks(conn, contract_id, chunks)
            total_chunks += len(chunks)

            for para in contract["paragraphs"]:
                for qa in para["qas"]:
                    if qa["answers"]:
                        store_ground_truth(
                            conn,
                            contract_id=contract_id,
                            question=qa["question"],
                            answer=qa["answers"][0]["text"],
                            answer_start=qa["answers"][0]["answer_start"],
                        )

            contracts_processed += 1
            logger.info(
                "✅ %d/%d  —  %s  —  %d chunks",
                contracts_processed, max_contracts,
                title[:60], len(chunks),
            )

    finally:
        # Always close — even if exception raised mid-ingestion
        conn.close()

    logger.info("-" * 60)
    logger.info("Ingestion complete")
    logger.info("  Contracts : %d", contracts_processed)
    logger.info("  Chunks    : %d", total_chunks)
    logger.info("  Database  : %s", DB_PATH)
    logger.info("-" * 60)

    return contracts_processed, total_chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_ingestion(max_contracts=510)