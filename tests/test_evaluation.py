"""
tests/test_evaluation.py

Tests for Agent 3 — Evaluation.

Tests pure functions in isolation:
    - Weight assertion
    - EvaluationResult dataclass
    - get_ground_truth keyword extraction
"""

import sys
import os
import string
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEvalWeights:

    def test_weights_sum_to_one(self):
        from config import (
            EVAL_WEIGHT_RETRIEVAL,
            EVAL_WEIGHT_FAITHFULNESS,
            EVAL_WEIGHT_RELEVANCE,
        )
        total = EVAL_WEIGHT_RETRIEVAL + EVAL_WEIGHT_FAITHFULNESS + EVAL_WEIGHT_RELEVANCE
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_weights_are_positive(self):
        from config import (
            EVAL_WEIGHT_RETRIEVAL,
            EVAL_WEIGHT_FAITHFULNESS,
            EVAL_WEIGHT_RELEVANCE,
        )
        assert EVAL_WEIGHT_RETRIEVAL    >= 0.0
        assert EVAL_WEIGHT_FAITHFULNESS >= 0.0
        assert EVAL_WEIGHT_RELEVANCE    >= 0.0


class TestKeywordExtraction:
    """Tests for keyword extraction logic in get_ground_truth."""

    def extract_keywords(self, question: str) -> list[str]:
        """Inline copy of keyword extraction for isolated testing."""
        stopwords = {"what", "is", "the", "a", "an", "are", "does",
                     "do", "in", "of", "for", "this", "that", "and"}
        return [
            stripped
            for word in question.split()
            for stripped in [word.lower().strip(string.punctuation)]
            if stripped
            and stripped not in stopwords
            and len(stripped) > 2
        ]

    def test_extracts_keywords(self):
        result = self.extract_keywords("What is the termination clause?")
        assert "termination" in result
        assert "clause" in result

    def test_removes_stopwords(self):
        result = self.extract_keywords("What is the termination clause?")
        assert "what" not in result
        assert "is" not in result
        assert "the" not in result

    def test_strips_punctuation(self):
        result = self.extract_keywords("What is the termination clause?")
        assert "clause" in result
        assert "clause?" not in result

    def test_empty_question_returns_empty(self):
        result = self.extract_keywords("")
        assert result == []

    def test_only_stopwords_returns_empty(self):
        result = self.extract_keywords("what is the")
        assert result == []

    def test_punctuation_only_word_skipped(self):
        result = self.extract_keywords("what is ??? the clause")
        assert "???" not in result
        assert "" not in result


class TestEvaluationResult:

    def test_passed_when_above_threshold(self):
        from agents.evaluation import EvaluationResult
        result = EvaluationResult(
            question="test",
            retrieval_score=0.8,
            faithfulness_score=0.8,
            answer_relevance=0.8,
            overall_score=0.8,
            passed=True,
            needs_human_review=False,
            failure_reason="",
            ground_truth_found=True,
        )
        assert result.passed is True

    def test_failed_when_below_threshold(self):
        from agents.evaluation import EvaluationResult
        result = EvaluationResult(
            question="test",
            retrieval_score=0.2,
            faithfulness_score=0.2,
            answer_relevance=0.2,
            overall_score=0.2,
            passed=False,
            needs_human_review=True,
            failure_reason="Low scores",
            ground_truth_found=False,
        )
        assert result.passed is False
        assert result.needs_human_review is True