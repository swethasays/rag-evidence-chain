"""
tests/test_retrieval.py

Tests for Agent 1 — Retrieval.

Tests pure instance methods via a lightweight fixture that bypasses
the heavy __init__ (database, FAISS, CrossEncoder) so these run
without any infrastructure.
"""

import sys
import os
import pytest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.retrieval import RetrievalAgent


@pytest.fixture
def agent():
    """RetrievalAgent with __init__ skipped — tests pure logic methods only."""
    with patch.object(RetrievalAgent, "__init__", return_value=None):
        instance = object.__new__(RetrievalAgent)
    return instance


@pytest.fixture
def sample_chunks():
    return [
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


class TestApplyFilters:
    """Tests for RetrievalAgent._apply_filters."""

    def test_filter_by_contract_title(self, agent, sample_chunks):
        result = agent._apply_filters(sample_chunks, {"contract_title": "WHITESMOKE"})
        assert len(result) == 2
        assert all("WHITESMOKE" in c["contract_title"] for c in result)

    def test_filter_by_contract_id(self, agent, sample_chunks):
        result = agent._apply_filters(sample_chunks, {"contract_id": "contract_abc"})
        assert len(result) == 2
        assert all(c["contract_id"] == "contract_abc" for c in result)

    def test_filter_no_match_returns_empty(self, agent, sample_chunks):
        result = agent._apply_filters(sample_chunks, {"contract_title": "NONEXISTENT"})
        assert result == []

    def test_filter_case_insensitive(self, agent, sample_chunks):
        result = agent._apply_filters(sample_chunks, {"contract_title": "whitesmoke"})
        assert len(result) == 2

    def test_no_filter_returns_all(self, agent, sample_chunks):
        result = agent._apply_filters(sample_chunks, {})
        assert len(result) == 3


class TestMergeResults:
    """Tests for RetrievalAgent._merge_results."""

    def test_no_duplicates(self, agent):
        semantic = [{"id": "a", "text": "A"}]
        keyword  = [{"id": "b", "text": "B"}]
        result = agent._merge_results(semantic, keyword)
        assert len(result) == 2

    def test_duplicate_marked_as_both(self, agent):
        semantic = [{"id": "a", "text": "A"}]
        keyword  = [{"id": "a", "text": "A"}]
        result = agent._merge_results(semantic, keyword)
        assert len(result) == 1
        assert result[0]["found_by"] == "both"

    def test_semantic_result_comes_first(self, agent):
        semantic = [{"id": "a", "text": "A"}]
        keyword  = [{"id": "b", "text": "B"}]
        result = agent._merge_results(semantic, keyword)
        assert result[0]["found_by"] == "semantic"
        assert result[1]["found_by"] == "keyword"

    def test_empty_inputs(self, agent):
        result = agent._merge_results([], [])
        assert result == []

    def test_keyword_only_results_included(self, agent):
        semantic = []
        keyword  = [{"id": "a", "text": "A"}]
        result = agent._merge_results(semantic, keyword)
        assert len(result) == 1
        assert result[0]["found_by"] == "keyword"
