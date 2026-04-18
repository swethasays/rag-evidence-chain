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

import duckdb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHUNK_OVERLAP, CHUNK_SIZE, LOCAL_DATA_PATH

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = "data/contracts.db"
CUAD_PATH = os.path.join(LOCAL_DATA_PATH, "CUAD_v1.json")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def setup_database(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    """
    Create the DuckDB database and the three tables we need:
      - contracts    : one row per contract
      - chunks       : one row per text chunk
      - ground_truth : one row per QA pair (used by the eval agent)
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


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split contract text into overlapping chunks using recursive
    character splitting — preserves paragraph and sentence boundaries
    wherever possible before falling back to smaller units.

    Args:
        text:       Full contract text
        chunk_size: Target size in characters per chunk
        overlap:    Shared characters between consecutive chunks

    Returns:
        List of dicts: {text, char_start, char_end}
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Try to split on natural boundaries first
    # Falls back to smaller units if chunk is still too big
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # LangChain gives us strings — we need to add position metadata
    raw_chunks = splitter.split_text(text)

    chunks = []
    search_start = 0

    for raw in raw_chunks:
        # Find where this chunk starts in the original text
        char_start = text.find(raw[:50], search_start)
        char_end = char_start + len(raw)

        chunks.append({
            "text": raw,
            "char_start": char_start,
            "char_end": char_end,
        })

        # Move search forward to avoid matching earlier occurrences
        search_start = max(0, char_start - overlap)

    return chunks

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
    gt_id = f"{contract_id}_{abs(hash(question))}"
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

    conn = setup_database()
    data = load_cuad()

    contracts_processed = 0
    total_chunks = 0

    for contract in data["data"][:max_contracts]:
        title = contract["title"]

        # Concatenate all paragraph contexts into one document string.
        full_text = " ".join(p["context"] for p in contract["paragraphs"])

        store_contract(conn, title, title, full_text)

        chunks = chunk_text(full_text)
        store_chunks(conn, title, chunks)
        total_chunks += len(chunks)

        # Ground-truth QA pairs — one per annotated question.
        for para in contract["paragraphs"]:
            for qa in para["qas"]:
                if qa["answers"]:
                    store_ground_truth(
                        conn,
                        contract_id=title,
                        question=qa["question"],
                        answer=qa["answers"][0]["text"],
                        answer_start=qa["answers"][0]["answer_start"],
                    )

        contracts_processed += 1
        logger.info(
            "✅ %d/%d  —  %s  —  %d chunks",
            contracts_processed,
            max_contracts,
            title[:60],
            len(chunks),
        )

    logger.info("-" * 60)
    logger.info("Ingestion complete")
    logger.info("  Contracts : %d", contracts_processed)
    logger.info("  Chunks    : %d", total_chunks)
    logger.info("  Database  : %s", DB_PATH)
    logger.info("-" * 60)

    conn.close()
    return contracts_processed, total_chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_ingestion(max_contracts=50)