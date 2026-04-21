# docker/Dockerfile

# ── Base image ────────────────────────────────────────────
# Python 3.11 slim — smaller than full image, has what we need
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────
LABEL maintainer="rag-evidence-chain"
LABEL description="RAG Evidence Chain — legal contract Q&A"

# ── Environment ───────────────────────────────────────────
# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout/stderr
# Critical for seeing logs in real time in Docker
ENV PYTHONUNBUFFERED=1
# Set working directory
ENV WORKDIR=/app

# ── System dependencies ───────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────
WORKDIR $WORKDIR

# ── Python dependencies ───────────────────────────────────
# Copy requirements first — Docker layer caching
# If requirements.txt hasn't changed, this layer is cached
# and pip install is skipped on rebuild
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-docker.txt

# ── Application code ──────────────────────────────────────
COPY . .

# ── Data directory ────────────────────────────────────────
# Create data directory — DuckDB and FAISS index live here
# In production this would be a mounted volume
RUN mkdir -p data/contracts

# ── Ports ─────────────────────────────────────────────────
EXPOSE 7860
EXPOSE 8000

# ── Health check ──────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# ── Startup ───────────────────────────────────────────────
COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]