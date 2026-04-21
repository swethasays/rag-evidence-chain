"""
tests/test_reasoning.py

Tests for Agent 2 — Reasoning.

Tests pure functions in isolation:
    - clamp()
    - extract_json()
    - resolve_chunk()
    - clean_title()
    - _get_cache_key()
"""

import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.reasoning import (
    clamp,
    extract_json,
    resolve_chunk,
    clean_title,
    _get_cache_key,
)


class TestClamp:

    def test_within_range(self):
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_below_min(self):
        assert clamp(-0.1, 0.0, 1.0) == 0.0

    def test_above_max(self):
        assert clamp(1.5, 0.0, 1.0) == 1.0

    def test_at_min(self):
        assert clamp(0.0, 0.0, 1.0) == 0.0

    def test_at_max(self):
        assert clamp(1.0, 0.0, 1.0) == 1.0


class TestExtractJson:

    def test_plain_json(self):
        raw = '{"answer_found": true, "overall_confidence": 0.9}'
        result = extract_json(raw)
        assert result == raw

    def test_json_with_fence(self):
        raw = '```json\n{"answer_found": true}\n```'
        result = extract_json(raw)
        assert '"answer_found": true' in result

    def test_json_with_preamble(self):
        raw = 'Here is the answer: {"answer_found": true}'
        result = extract_json(raw)
        assert '"answer_found": true' in result

    def test_nested_json(self):
        raw = '{"sentences": [{"text": "hello", "chunk_number": 1}]}'
        result = extract_json(raw)
        assert '"sentences"' in result
        assert '"text"' in result

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            extract_json("no json here at all")


class TestResolveChunk:

    def setup_method(self):
        self.chunk_map = {
            1: {"id": "chunk_1", "text": "text 1"},
            2: {"id": "chunk_2", "text": "text 2"},
        }

    def test_valid_chunk(self):
        chunk, hallucinated = resolve_chunk(1, self.chunk_map, "question")
        assert hallucinated is False
        assert chunk["id"] == "chunk_1"

    def test_hallucinated_chunk(self):
        chunk, hallucinated = resolve_chunk(99, self.chunk_map, "question")
        assert hallucinated is True
        assert chunk is None

    def test_boundary_chunk(self):
        chunk, hallucinated = resolve_chunk(2, self.chunk_map, "question")
        assert hallucinated is False
        assert chunk["id"] == "chunk_2"


class TestCleanTitle:

    def test_cuad_format_with_dash(self):
        title = "ADAMSGOLFINC_03_21_2005-EX-10.17-ENDORSEMENT AGREEMENT"
        result = clean_title(title)
        assert "ENDORSEMENT AGREEMENT" in result

    def test_format_with_space_dash(self):
        title = "Company Name - Employment Agreement"
        result = clean_title(title)
        assert result == "Employment Agreement"

    def test_short_title_unchanged(self):
        title = "Simple Agreement"
        result = clean_title(title)
        assert result == "Simple Agreement"


class TestGetCacheKey:

    def test_same_inputs_same_key(self):
        key1 = _get_cache_key("question", ["id1", "id2"])
        key2 = _get_cache_key("question", ["id1", "id2"])
        assert key1 == key2

    def test_different_questions_different_keys(self):
        key1 = _get_cache_key("question 1", ["id1"])
        key2 = _get_cache_key("question 2", ["id1"])
        assert key1 != key2

    def test_chunk_order_doesnt_matter(self):
        # Sorted internally — order shouldn't affect cache hit
        key1 = _get_cache_key("question", ["id1", "id2"])
        key2 = _get_cache_key("question", ["id2", "id1"])
        assert key1 == key2

    def test_key_has_prefix(self):
        key = _get_cache_key("question", ["id1"])
        assert key.startswith("reasoning:")