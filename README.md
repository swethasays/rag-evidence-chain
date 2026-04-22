---
title: Rag Evidence Chain
emoji: 🐨
colorFrom: gray
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Legal contract Q&A — every answer traced to its source.
---

# 🔗 RAG Evidence Chain

> Production agentic RAG system for legal contracts — every answer traced to its source with a live evidence chain.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.0.55-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 The Problem

Legal professionals waste hours searching contracts for specific clauses.
Existing AI tools hallucinate answers with no way to verify them.

## 💡 The Solution

Ask any question about a legal contract and get:
- ✅ A precise answer
- ✅ Every sentence linked to its exact source chunk
- ✅ A live visual evidence graph (DAG)
- ✅ A confidence score
- ✅ Automatic flagging when the system is unsure

---

## 🏗️ Architecture

```
User Question
      ↓
Agent 1: Retrieval    → finds top 5 relevant chunks (FAISS + BM25)
      ↓
Agent 2: Reasoning    → forms answer + cites sources
      ↓
Agent 3: Evaluation   → scores answer quality
      ↓
Streamlit UI          → answer + clickable evidence chain DAG
```

---

## 🛠️ Stack

| Layer | Tools |
|---|---|
| Data | CUAD, DuckDB, Unstructured.io, HuggingFace |
| Agents | LangGraph, Groq, FAISS, BM25, Cross-encoder |
| Cache | Redis |
| Security | Pydantic, Rate limiting, dotenv |
| Observability | LangSmith, Weights & Biases |
| Backend | FastAPI, Docker |
| Frontend | Streamlit, Plotly |
| Deployment | HuggingFace Spaces, Railway, GitHub Actions |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/swethasays/rag-evidence-chain.git
cd rag-evidence-chain
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your keys
```bash
cp .env.example .env
# Fill in your API keys
```

### 4. Ingest the data
```bash
python data/ingest.py
```

### 5. Run the app
```bash
streamlit run ui/app.py
```

---

## 📊 Evaluation

The system measures:
- **Retrieval score** — did we find the right chunks?
- **Faithfulness score** — does the answer match the chunks?
- **Confidence score** — how certain is the system?

---

## ⚠️ Limitations

- FAISS doesn't scale beyond ~1M vectors (swap to Pinecone in production)
- Groq free tier has rate limits
- CUAD is English only
- Not a substitute for legal advice

---

## 📁 Project Structure

```
rag-evidence-chain/
│
├── data/
│   ├── storage/          # local → GCP → S3 (abstracted)
│   ├── vectorstore/      # FAISS → Pinecone (abstracted)
│   ├── ingest.py         # load CUAD dataset
│   └── chunker.py        # semantic chunking
│
├── agents/
│   ├── retrieval.py      # Agent 1 — find relevant chunks
│   ├── reasoning.py      # Agent 2 — form answer + cite sources
│   ├── evaluation.py     # Agent 3 — score answer quality
│   └── graph.py          # LangGraph wiring
│
├── api/
│   ├── main.py           # FastAPI
│   ├── models.py         # Pydantic models
│   └── middleware.py     # rate limiting + auth
│
├── ui/
│   └── app.py            # Streamlit + Plotly DAG
│
├── observability/
│   ├── tracing.py        # LangSmith
│   └── logging.py        # structured logs
│
├── tests/                # one test per agent
├── docker/               # Dockerfile + Compose
├── .github/workflows/    # CI/CD
├── config.py             # one line swaps
├── requirements.txt
└── .env                  # secrets (never pushed)
```

---

## 👩‍💻 Author

Built by [@swethasays](https://github.com/swethasays)