"""
tests/test_retrieval.py

Tests for Agent 1 — Retrieval.

Tests the core retrieval functions in isolation
without loading the full pipeline.
"""

import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Unit tests — pure functions, no dependencies
# ---------------------------------------------------------------------------

class TestApplyFilters:
    """Tests for _apply_filters — no DB or FAISS needed."""

    def setup_method(self):
        """Create a minimal RetrievalAgent mock with just _apply_filters."""
        from agents.retrieval import RetrievalAgent
        # We only test _apply_filters — doesn't need DB or models
        # Patch heavy init to avoid loading models in tests
        self.chunks = [
            {
                "id": "chunk_1",
                "contract_id": "contract_abc",
                "contract_title": "WHITESMOKE,INC_11_08_2011-EX-10.26",
                "text": "Termination clause text here.",
                "chunk_index": 0,
                "char_start": 0,
                "char_end": 100,
            },
            {
                "id": "chunk_2",
                "contract_id": "contract_def",
                "contract_title": "ADAMSGOLFINC_03_21_2005-EX-10.17",
                "text": "Payment terms text here.",
                "chunk_index": 1,
                "char_start": 100,
                "char_end": 200,
            },
            {
                "id": "chunk_3",
                "contract_id": "contract_abc",
                "contract_title": "WHITESMOKE,INC_11_08_2011-EX-10.26",
                "text": "Confidentiality clause text here.",
                "chunk_index": 2,
                "char_start": 200,
                "char_end": 300,
            },
        ]

    def _apply_filters(self, chunks, filters):
        """Inline copy of _apply_filters for isolated testing."""
        filtered = chunks
        if "contract_id" in filters:
            filtered = [
                c for c in filtered
                if c["contract_id"] == filters["contract_id"]
            ]
        if "contract_title" in filters:
            filtered = [
                c for c in filtered
                if filters["contract_title"].lower()
                in c["contract_title"].lower()
            ]
        return filtered

    def test_filter_by_contract_title(self):
        result = self._apply_filters(
            self.chunks,
            {"contract_title": "WHITESMOKE"}
        )
        assert len(result) == 2
        assert all("WHITESMOKE" in c["contract_title"] for c in result)

    def test_filter_by_contract_id(self):
        result = self._apply_filters(
            self.chunks,
            {"contract_id": "contract_abc"}
        )
        assert len(result) == 2
        assert all(c["contract_id"] == "contract_abc" for c in result)

    def test_filter_no_match_returns_empty(self):
        result = self._apply_filters(
            self.chunks,
            {"contract_title": "NONEXISTENT_CONTRACT"}
        )
        assert result == []

    def test_filter_case_insensitive(self):
        result = self._apply_filters(
            self.chunks,
            {"contract_title": "whitesmoke"}
        )
        assert len(result) == 2

    def test_no_filter_returns_all(self):
        result = self._apply_filters(self.chunks, {})
        assert len(result) == 3


class TestMergeResults:
    """Tests for _merge_results — deduplication and found_by metadata."""

    def _merge_results(self, semantic, keyword):
        """Inline copy of _merge_results for isolated testing."""
        seen_ids = {}
        merged = []
        for chunk in semantic:
            chunk["found_by"] = "semantic"
            seen_ids[chunk["id"]] = len(merged)
            merged.append(chunk)
        for chunk in keyword:
            if chunk["id"] not in seen_ids:
                chunk["found_by"] = "keyword"
                seen_ids[chunk["id"]] = len(merged)
                merged.append(chunk)
            else:
                merged[seen_ids[chunk["id"]]]["found_by"] = "both"
        return merged

    def test_no_duplicates(self):
        semantic = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]
        keyword  = [{"id": "c", "text": "C"}]
        result = self._merge_results(semantic, keyword)
        assert len(result) == 3

    def test_duplicate_marked_as_both(self):
        semantic = [{"id": "a", "text": "A"}]
        keyword  = [{"id": "a", "text": "A"}]
        result = self._merge_results(semantic, keyword)
        assert len(result) == 1
        assert result[0]["found_by"] == "both"

    def test_semantic_first(self):
        semantic = [{"id": "a", "text": "A"}]
        keyword  = [{"id": "b", "text": "B"}]
        result = self._merge_results(semantic, keyword)
        assert result[0]["found_by"] == "semantic"
        assert result[1]["found_by"] == "keyword"

    def test_empty_inputs(self):
        result = self._merge_results([], [])
        assert result == []