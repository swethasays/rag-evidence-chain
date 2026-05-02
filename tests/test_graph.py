"""
tests/test_graph.py

Tests for the LangGraph pipeline routing logic.

Tests route_after_evaluation() in isolation — the function is pure
(no agents, no LLM, no DB) so it can be fully covered without mocks.
"""

import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.graph import route_after_evaluation, MAX_RETRIES, MIN_RETRIEVAL_SCORE
from agents.evaluation import EvaluationResult


def make_state(
    passed: bool,
    overall_score: float = 0.8,
    retrieval_score: float = 0.8,
    faithfulness_score: float = 0.8,
    answer_relevance: float = 0.8,
    needs_human_review: bool = False,
    failure_reason: str = "",
    ground_truth_found: bool = True,
    retry_count: int = 1,
) -> dict:
    """Build a minimal PipelineState for routing tests."""
    return {
        "eval_result": EvaluationResult(
            question="test question",
            retrieval_score=retrieval_score,
            faithfulness_score=faithfulness_score,
            answer_relevance=answer_relevance,
            overall_score=overall_score,
            passed=passed,
            needs_human_review=needs_human_review,
            failure_reason=failure_reason,
            ground_truth_found=ground_truth_found,
        ),
        "retry_count": retry_count,
    }


class TestRouteAfterEvaluation:

    def test_passed_routes_to_end(self):
        state = make_state(passed=True)
        assert route_after_evaluation(state) == "end"

    def test_low_retrieval_routes_to_retry(self):
        state = make_state(
            passed=False,
            retrieval_score=MIN_RETRIEVAL_SCORE - 0.01,
            retry_count=1,
        )
        assert route_after_evaluation(state) == "retry"

    def test_retry_exhausted_routes_to_human_review(self):
        # retry_count already equals MAX_RETRIES + 1 — no more retries allowed
        state = make_state(
            passed=False,
            retrieval_score=MIN_RETRIEVAL_SCORE - 0.01,
            retry_count=MAX_RETRIES + 1,
        )
        assert route_after_evaluation(state) == "human_review"

    def test_low_overall_routes_to_human_review(self):
        # Retrieval fine but answer quality low
        state = make_state(
            passed=False,
            overall_score=0.2,
            retrieval_score=MIN_RETRIEVAL_SCORE + 0.1,
            faithfulness_score=0.2,
            answer_relevance=0.2,
        )
        assert route_after_evaluation(state) == "human_review"

    def test_eval_failure_routes_to_human_review_not_retry(self):
        # All-zero scores from a crashed evaluation node should not trigger retry
        state = make_state(
            passed=False,
            overall_score=0.0,
            retrieval_score=0.0,
            faithfulness_score=0.0,
            answer_relevance=0.0,
            failure_reason="Evaluation failed: RuntimeError: connection refused",
            retry_count=1,
        )
        assert route_after_evaluation(state) == "human_review"

    def test_retry_at_exact_max_retries(self):
        # retry_count == MAX_RETRIES should still retry (boundary)
        state = make_state(
            passed=False,
            retrieval_score=MIN_RETRIEVAL_SCORE - 0.01,
            retry_count=MAX_RETRIES,
        )
        assert route_after_evaluation(state) == "retry"
