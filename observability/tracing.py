"""
observability/tracing.py

LangSmith tracing for the RAG Evidence Chain pipeline.

LangSmith traces every pipeline run end-to-end so you can see:
    - What question went in
    - What chunks were retrieved
    - What the LLM received and returned
    - How long each step took
    - Where failures occurred
    - Token usage per run

This is the observability layer — it doesn't change pipeline
behaviour, only records what happened for debugging and monitoring.

Usage:
    from observability.tracing import setup_tracing, trace_pipeline_run
    setup_tracing()  # call once at startup
"""

import os
import sys
import time
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observability.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_tracing() -> bool:
    """
    Configure LangSmith tracing via environment variables.

    LangSmith tracing is enabled by setting three env vars:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=<your key>
        LANGCHAIN_PROJECT=rag-evidence-chain

    These are read from .env by config.py — we just need to ensure
    they're set in the environment before LangGraph runs.

    Returns:
        True if tracing is enabled, False if API key is missing
    """
    from config import LANGCHAIN_API_KEY, LANGCHAIN_PROJECT

    if not LANGCHAIN_API_KEY:
        logger.warning(
            "LANGCHAIN_API_KEY not set — LangSmith tracing disabled. "
            "Add it to .env to enable tracing."
        )
        return False

    # Set environment variables LangSmith reads automatically
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"]    = LANGCHAIN_PROJECT

    logger.info(
        "LangSmith tracing enabled. Project: %s",
        LANGCHAIN_PROJECT,
    )
    return True


# ---------------------------------------------------------------------------
# Pipeline run tracer
# ---------------------------------------------------------------------------

def trace_pipeline_run(
    question: str,
    result: dict,
    duration_ms: float,
) -> None:
    """
    Log a structured trace of a completed pipeline run.

    This is a lightweight local trace that runs regardless of
    whether LangSmith is configured. It logs:
        - Question asked
        - Number of sentences in answer
        - Evaluation scores
        - Pass/fail outcome
        - Total duration

    In production this feeds into LangSmith automatically via
    the LANGCHAIN_TRACING_V2 environment variable — LangGraph
    instruments itself without any code changes needed.

    Args:
        question:    The user's question
        result:      Pipeline result dict from RAGPipeline.run()
        duration_ms: Total pipeline duration in milliseconds
    """
    scores = result.get("eval_scores", {})

    logger.info(
        "Pipeline trace — question='%s' sentences=%d "
        "retrieval=%.2f faithfulness=%.2f relevance=%.2f "
        "overall=%.2f passed=%s duration_ms=%.0f",
        question[:60],
        len(result.get("sentences", [])),
        scores.get("retrieval", 0.0),
        scores.get("faithfulness", 0.0),
        scores.get("relevance", 0.0),
        scores.get("overall", 0.0),
        result.get("passed", False),
        duration_ms,
    )


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

class Timer:
    """
    Simple context manager for timing code blocks.

    Usage:
        with Timer() as t:
            result = pipeline.run(question)
        print(f"Took {t.elapsed_ms:.0f}ms")
    """

    def __enter__(self) -> "Timer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed_ms = (time.monotonic() - self._start) * 1000

    @property
    def elapsed_s(self) -> float:
        """Elapsed time in seconds."""
        return self.elapsed_ms / 1000