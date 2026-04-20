"""
observability/logging.py

Centralised logging configuration for the RAG Evidence Chain system.

Every agent imports get_logger() from here instead of calling
logging.basicConfig() independently. This ensures:
    - Consistent format across all modules
    - Single place to change log level or output
    - Structured JSON output in production
    - No duplicate handlers from multiple basicConfig() calls

Usage:
    from observability.logging import get_logger
    logger = get_logger(__name__)
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# Read log level from environment — defaults to INFO
# Set LOG_LEVEL=DEBUG in .env for verbose output during development
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Read log format from environment — "json" for production, "text" for dev
# Set LOG_FORMAT=json in .env for structured JSON output
LOG_FORMAT = os.getenv("LOG_FORMAT", "text").lower()


# ---------------------------------------------------------------------------
# JSON formatter — for production
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """
    Format log records as JSON for structured logging in production.

    JSON logs are machine-parseable — tools like Datadog, CloudWatch,
    and Loki can index and query them without regex parsing.

    Example output:
        {
            "timestamp": "2024-01-15T10:30:00Z",
            "level": "INFO",
            "logger": "agents.retrieval",
            "message": "Retrieved 5 chunks",
            "module": "retrieval",
            "line": 142
        }
    """

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record to a JSON string."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
            "module":    record.module,
            "line":      record.lineno,
        }

        # Include exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


# ---------------------------------------------------------------------------
# Text formatter — for development
# ---------------------------------------------------------------------------

TEXT_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s"
DATE_FORMAT = "%H:%M:%S"


# ---------------------------------------------------------------------------
# Handler setup — run once
# ---------------------------------------------------------------------------

_configured = False


def _setup_logging() -> None:
    """
    Configure the root logger once.

    Called automatically on first get_logger() call.
    Subsequent calls are no-ops — guards with _configured flag.

    Sets up:
        - StreamHandler to stdout
        - JSON or text formatter based on LOG_FORMAT env var
        - Log level from LOG_LEVEL env var
    """
    global _configured
    if _configured:
        return

    root_logger = logging.getLogger()

    # Remove any existing handlers — prevents duplicate output
    # from multiple basicConfig() calls across modules
    root_logger.handlers.clear()

    # Create stdout handler
    handler = logging.StreamHandler(sys.stdout)

    # Choose formatter based on environment
    if LOG_FORMAT == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(TEXT_FORMAT, datefmt=DATE_FORMAT)
        )

    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Suppress noisy LangSmith rate limit warnings
    logging.getLogger("langsmith.client").setLevel(logging.ERROR)
    
    _configured = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger, configuring the root logger on first call.

    This is the only function agents should import from this module.
    It replaces the logging.basicConfig() + logging.getLogger() pattern
    that was duplicated across every agent file.

    Args:
        name: Logger name — pass __name__ from the calling module.
              This gives loggers like "agents.retrieval", "agents.reasoning"
              which makes it easy to filter logs by component.

    Returns:
        Configured logger instance

    Usage:
        from observability.logging import get_logger
        logger = get_logger(__name__)
        logger.info("RetrievalAgent ready.")
    """
    _setup_logging()
    return logging.getLogger(name)