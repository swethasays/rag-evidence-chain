"""
tests/test_api.py

Tests for the FastAPI endpoints.

Uses FastAPI's TestClient. The RAGPipeline is mocked out so these
tests run without models, a database, or an NVIDIA API key.
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    """
    Return a TestClient with the RAGPipeline mocked.

    The fixture patches RAGPipeline before the app is imported so
    the lifespan handler never tries to load real models.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = {
        "question": "What is the termination clause?",
        "answer": "The agreement may be terminated with 30 days notice.",
        "sentences": [
            {
                "text": "The agreement may be terminated with 30 days notice.",
                "chunk_id": "chunk_1",
                "contract_title": "ENDORSEMENT AGREEMENT",
                "chunk_text": "Either party may terminate with 30 days written notice.",
                "confidence": 0.9,
            }
        ],
        "eval_scores": {
            "retrieval": 0.85,
            "faithfulness": 0.88,
            "relevance": 0.90,
            "overall": 0.88,
            "retrieval_score_available": True,
        },
        "needs_human_review": False,
        "passed": True,
        "chunks_used": ["chunk_1"],
    }

    with patch("agents.graph.RAGPipeline", return_value=mock_pipeline):
        # Import app inside patch context so lifespan uses the mock
        from fastapi.testclient import TestClient
        import importlib
        import api.main as main_module
        importlib.reload(main_module)
        main_module._pipeline = mock_pipeline
        yield TestClient(main_module.app, raise_server_exceptions=True), mock_pipeline


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        test_client, _ = client
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client):
        test_client, _ = client
        response = test_client.get("/health")
        assert response.json()["status"] == "ok"


class TestRootEndpoint:

    def test_root_returns_200(self, client):
        test_client, _ = client
        response = test_client.get("/")
        assert response.status_code == 200

    def test_root_lists_endpoints(self, client):
        test_client, _ = client
        data = test_client.get("/").json()
        assert "ask" in data["endpoints"]
        assert "health" in data["endpoints"]


class TestAskEndpoint:

    def test_ask_returns_200(self, client):
        test_client, _ = client
        response = test_client.post(
            "/ask", json={"question": "What is the termination clause?"}
        )
        assert response.status_code == 200

    def test_ask_returns_answer(self, client):
        test_client, _ = client
        response = test_client.post(
            "/ask", json={"question": "What is the termination clause?"}
        )
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_ask_returns_sentences(self, client):
        test_client, _ = client
        response = test_client.post(
            "/ask", json={"question": "What is the termination clause?"}
        )
        data = response.json()
        assert "sentences" in data
        assert isinstance(data["sentences"], list)

    def test_ask_returns_eval_scores(self, client):
        test_client, _ = client
        response = test_client.post(
            "/ask", json={"question": "What is the termination clause?"}
        )
        data = response.json()
        scores = data["eval_scores"]
        assert "retrieval" in scores
        assert "faithfulness" in scores
        assert "relevance" in scores
        assert "overall" in scores
        assert "retrieval_score_available" in scores

    def test_ask_strips_human_review_flag_from_answer(self, client):
        test_client, mock_pipeline = client
        from config import HUMAN_REVIEW_FLAG
        mock_pipeline.run.return_value["answer"] = (
            "Some answer.\n\n" + HUMAN_REVIEW_FLAG
        )
        response = test_client.post(
            "/ask", json={"question": "What is the payment clause?"}
        )
        assert HUMAN_REVIEW_FLAG not in response.json()["answer"]

    def test_ask_with_contract_filter_passes_filter_to_pipeline(self, client):
        test_client, mock_pipeline = client
        test_client.post(
            "/ask",
            json={
                "question": "What is the termination clause?",
                "contract_title": "ENDORSEMENT AGREEMENT",
            },
        )
        call_kwargs = mock_pipeline.run.call_args
        assert call_kwargs is not None
        filters = call_kwargs.kwargs.get("filters") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else None
        # Accept either positional or keyword
        if filters is None and call_kwargs.kwargs:
            filters = call_kwargs.kwargs.get("filters")
        assert filters is not None
        assert "contract_title" in filters

    def test_ask_empty_question_returns_422(self, client):
        test_client, _ = client
        response = test_client.post("/ask", json={"question": ""})
        assert response.status_code == 422
