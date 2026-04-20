"""
api/middleware.py

FastAPI middleware for the RAG Evidence Chain API.

Two layers of protection:
    1. CORS         — controls which domains can call the API
    2. Rate limiting — prevents abuse (10 requests/minute per IP)

Rate limiting uses slowapi which wraps the token bucket pattern
we already built in reasoning.py — but at the HTTP layer.

Usage:
    from api.middleware import add_middleware
    add_middleware(app)
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import RATE_LIMIT
from observability.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

# Keyed by IP address — each IP gets its own bucket
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Middleware setup
# ---------------------------------------------------------------------------

def add_middleware(app: FastAPI) -> None:
    """
    Add all middleware to the FastAPI app.

    Called once during app creation in main.py.
    Order matters — middleware is applied in reverse order.

    Args:
        app: The FastAPI application instance
    """
    # CORS — allow requests from UI and local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",   # Streamlit local
            "http://localhost:3000",   # React local (future)
            "https://*.hf.space",      # HuggingFace Spaces
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    logger.info("Middleware configured — CORS + rate limiting active.")