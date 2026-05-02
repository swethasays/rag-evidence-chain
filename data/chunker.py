"""
data/chunker.py

Text chunking utilities for contract documents.

Provides the Chunker class used by ingest.py to split contracts into
overlapping windows that fit within the embedding model's context limit.

Usage:
    from data.chunker import Chunker
    chunker = Chunker()
    chunks = chunker.chunk("Full contract text...")
    # Each chunk: {"text": str, "char_start": int, "char_end": int}
"""

import os
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHUNK_OVERLAP, CHUNK_SIZE
from observability.logging import get_logger

logger = get_logger(__name__)


class Chunker:
    """
    Splits contract text into overlapping chunks with character positions.

    Uses recursive character splitting — tries to preserve paragraph and
    sentence boundaries before splitting at smaller units. Tracks character
    offsets so retrieval results can highlight the exact source passage.

    Args:
        chunk_size:    Target characters per chunk (default from config)
        chunk_overlap: Shared characters between adjacent chunks (default from config)

    Usage:
        chunker = Chunker()
        chunks = chunker.chunk(text)
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str) -> list[dict]:
        """
        Split text into overlapping chunks with character positions.

        Args:
            text: Full contract text

        Returns:
            List of dicts with keys: text, char_start, char_end
        """
        if not text or not text.strip():
            logger.warning("Empty text passed to chunker — returning empty list.")
            return []

        raw_chunks = self._splitter.split_text(text)
        chunks = []
        search_start = 0

        for raw in raw_chunks:
            char_start = text.find(raw[:50], search_start)

            if char_start == -1:
                logger.warning(
                    "Could not locate chunk in original text — "
                    "using search_start as fallback. Chunk: '%s...'",
                    raw[:40],
                )
                char_start = search_start

            char_end = char_start + len(raw)
            chunks.append({
                "text": raw,
                "char_start": char_start,
                "char_end": char_end,
            })

            search_start = max(0, char_start - self.chunk_overlap)

        logger.info(
            "Chunked %d chars → %d chunks (size=%d, overlap=%d)",
            len(text), len(chunks), self.chunk_size, self.chunk_overlap,
        )
        return chunks
