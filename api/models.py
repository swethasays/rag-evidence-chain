"""
api/models.py

Pydantic request and response models for the FastAPI backend.

Every API endpoint uses these models for:
    - Input validation — bad requests rejected before hitting the pipeline
    - Response serialization — consistent JSON structure every time
    - Auto-generated docs — FastAPI uses these for /docs swagger UI

Usage:
    from api.models import QuestionRequest, PipelineResponse
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    """
    Request body for POST /ask endpoint.

    Validates the question and optional contract filter
    before the pipeline runs.
    """
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The legal contract question to answer",
        examples=["What is the termination clause?"],
    )
    contract_title: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optional contract title to scope the search",
        examples=["WHITESMOKE,INC_11_08_2011-EX-10.26"],
    )

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        """Reject questions that are only whitespace."""
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace.")
        return v.strip()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class CitedSentenceResponse(BaseModel):
    """A single sentence in the answer with its source citation."""
    text:           str
    chunk_id:       str
    contract_title: str
    chunk_text:     str
    confidence:     float


class EvalScoresResponse(BaseModel):
    """Evaluation scores for the pipeline run."""
    retrieval:                 float
    faithfulness:              float
    relevance:                 float
    overall:                   float
    retrieval_score_available: bool = True  # False when no ground truth exists — retrieval score is 0.5 placeholder


class PipelineResponse(BaseModel):
    """
    Response body for POST /ask endpoint.

    Contains the answer, evidence chain, and evaluation scores.
    """
    question:           str
    answer:             str
    sentences:          list[CitedSentenceResponse]
    eval_scores:        EvalScoresResponse
    needs_human_review: bool
    passed:             bool
    chunks_used:        list[str]


class ContractListResponse(BaseModel):
    """Response body for GET /contracts endpoint."""
    contracts: list[str]
    total:     int


class HealthResponse(BaseModel):
    """Response body for GET /health endpoint."""
    status:  str
    version: str = "1.0.0"