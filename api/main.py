"""
api/main.py

FastAPI backend for the RAG Evidence Chain system.

Exposes three endpoints:
    GET  /health      — liveness check
    GET  /contracts   — list available contracts
    POST /ask         — run the full RAG pipeline

The pipeline is loaded once at startup and shared across
all requests — avoids reloading models on every call.

Usage:
    uvicorn api.main:app --reload --port 8000
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager

import duckdb
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    QuestionRequest,
    PipelineResponse,
    CitedSentenceResponse,
    EvalScoresResponse,
    ContractListResponse,
    HealthResponse,
)
from api.middleware import add_middleware
from agents.graph import RAGPipeline
from config import API_KEY, DB_PATH, HUMAN_REVIEW_FLAG
from observability.logging import get_logger
from observability.tracing import setup_tracing

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str = Security(_api_key_header)) -> None:
    """
    Require X-API-Key header when API_KEY env var is set.

    In local dev (API_KEY not set) this is a no-op so you can call
    the API without a key. In production, set API_KEY and every
    protected endpoint will reject requests that omit or wrong the key.
    """
    if API_KEY is None:
        return  # dev mode — no key configured, skip check
    if key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Pass it as X-API-Key header.",
        )


# ---------------------------------------------------------------------------
# Pipeline — loaded once at startup
# ---------------------------------------------------------------------------

# Module-level pipeline instance
# Shared across all requests — avoids reloading models on every call
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Return the shared pipeline instance."""
    global _pipeline
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialised — server is starting up.",
        )
    return _pipeline


# ---------------------------------------------------------------------------
# Lifespan — startup and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the pipeline at startup, clean up at shutdown.

    Using lifespan instead of @app.on_event("startup") —
    the modern FastAPI pattern as of v0.93.
    """
    global _pipeline

    logger.info("Starting up RAG Evidence Chain API...")

    # Enable LangSmith tracing
    setup_tracing()

    # Load pipeline — this loads all models into memory
    # Takes ~30 seconds on first run (downloads models if needed)
    logger.info("Loading RAG pipeline...")
    _pipeline = RAGPipeline()
    logger.info("Pipeline ready. API accepting requests.")

    yield  # API is running

    # Shutdown
    logger.info("Shutting down RAG Evidence Chain API...")
    _pipeline = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Evidence Chain",
    description="Legal contract Q&A — every answer traced to its source",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS and rate limiting middleware
add_middleware(app)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Liveness check.

    Returns 200 if the API is running.
    Used by Docker health checks and load balancers.
    """
    return HealthResponse(status="ok")


@app.get("/", tags=["System"])
async def root():
    """API information and available endpoints."""
    return {
        "name": "RAG Evidence Chain",
        "description": "Legal contract Q&A - every answer traced to its source",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "contracts": "/contracts",
            "ask": "/ask",
            "docs": "/docs",
        },
    }


@app.get("/contracts", response_model=ContractListResponse, tags=["Contracts"],
         dependencies=[Depends(verify_api_key)])
async def list_contracts():
    """
    List all available contracts.

    Returns the titles of all contracts currently in the database.
    Used by the UI to populate the contract filter dropdown.
    """
    try:
        with duckdb.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT title FROM contracts ORDER BY title"
            ).fetchall()
        contracts = [row[0] for row in rows]
        return ContractListResponse(contracts=contracts, total=len(contracts))
    except Exception as e:
        logger.error("Failed to list contracts: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve contracts.")


@app.post("/ask", response_model=PipelineResponse, tags=["Pipeline"],
          dependencies=[Depends(verify_api_key)])
async def ask(request: QuestionRequest):
    """
    Run the full RAG pipeline for a question.

    Retrieves relevant chunks, forms a cited answer, and scores
    quality across retrieval, faithfulness, and relevance.

    Returns the answer with every sentence traced to its source chunk.
    """
    logger.info(
        "POST /ask — question='%s' contract='%s'",
        request.question[:60],
        request.contract_title or "all",
    )

    # Build filters from request
    filters = None
    if request.contract_title:
        filters = {"contract_title": request.contract_title}

    # Run pipeline
    try:
        pipeline = get_pipeline()
        result   = pipeline.run(request.question, filters=filters)
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {type(e).__name__}",
        )

    # Build response
    # Strip UI-specific human review text — API consumers use
    # needs_human_review field instead of embedded text
    clean_answer = result["answer"].replace(f"\n\n{HUMAN_REVIEW_FLAG}", "").strip()

    # Build response
    return PipelineResponse(
        question=result["question"],
        answer=clean_answer,
        sentences=[
            CitedSentenceResponse(**s)
            for s in result["sentences"]
        ],
        eval_scores=EvalScoresResponse(**result["eval_scores"]),
        needs_human_review=result["needs_human_review"],
        passed=result["passed"],
        chunks_used=result["chunks_used"],
    )