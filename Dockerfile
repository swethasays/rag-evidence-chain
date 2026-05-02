# docker/Dockerfile

# Base image
FROM python:3.11-slim

LABEL maintainer="rag-evidence-chain"
LABEL description="RAG Evidence Chain - legal contract Q&A"

# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout/stderr — critical for real-time logs in Docker
ENV PYTHONUNBUFFERED=1
ENV WORKDIR=/app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR $WORKDIR

# Copy requirements first so Docker can cache the install layer
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-docker.txt

COPY . .

# DuckDB and FAISS index live here; mount a volume in production
RUN mkdir -p data/contracts

EXPOSE 7860
EXPOSE 8000

# Check the FastAPI /health endpoint — it's the authoritative liveness signal.
# Streamlit (7860) is the UI layer; the API (8000) is what external callers use.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]
