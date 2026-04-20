"""
observability/

Logging and tracing for the RAG Evidence Chain system.

Modules:
    logging  — centralised logger factory (get_logger)
    tracing  — LangSmith tracing setup and pipeline run tracer
"""

from observability.logging import get_logger
from observability.tracing import setup_tracing, trace_pipeline_run, Timer

__all__ = [
    "get_logger",
    "setup_tracing",
    "trace_pipeline_run",
    "Timer",
]