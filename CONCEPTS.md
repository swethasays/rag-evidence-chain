# Concepts Behind RAG Evidence Chain

This document explains the core concepts used in this project — what each technology is, why it exists, and how it is used here specifically. Written as a reference to revise from, not as a tutorial.

---

## 1. RAG — Retrieval-Augmented Generation

### What it is
RAG is a pattern for answering questions with an LLM where the model's context is dynamically populated with relevant documents retrieved from a knowledge base at query time, rather than relying on what was baked into the model during training.

The two steps are:
1. **Retrieve** — find the most relevant chunks from a document store based on the question
2. **Generate** — give those chunks to the LLM as context and ask it to answer using only that information

### Why not just fine-tune the model?
Fine-tuning bakes knowledge into model weights. That means:
- Knowledge becomes stale as documents change (contracts get updated, new ones are added)
- You need to re-train every time the document set changes — expensive and slow
- The model can't cite where its answer came from

RAG separates knowledge (stored in a vector database, updated independently) from reasoning (the LLM). Update the documents, re-index them, and the model immediately has access to the new information. No retraining.

### Why RAG for legal contracts?
Legal contracts are long, specific, and highly varied. A general-purpose LLM won't have memorised the specific clauses of a specific contract from 2003. RAG retrieves the relevant clause first, then the LLM reasons about it. Every answer can be traced back to the exact text it came from — which matters enormously in legal contexts.

### The key concept to remember
RAG = retrieve first, then generate. The model doesn't know the answer — it reads the answer from the documents you give it at runtime.

---

## 2. Vector Embeddings and Semantic Search

### What it is
A vector embedding is a list of numbers (a vector) that represents the meaning of a piece of text. Texts with similar meaning produce similar vectors. The similarity between two vectors can be measured with cosine similarity — the closer to 1.0, the more semantically similar the texts are.

An embedding model (here: NVIDIA nv-embedqa-e5-v5, 1024 dimensions) converts text into these vectors. Every chunk of every contract is embedded at ingestion time and stored in FAISS. At query time, the question is embedded and FAISS finds the most similar vectors in the database — those are the most semantically relevant chunks.

### Why 1024 dimensions?
More dimensions = more expressive representation = better at capturing nuanced meaning. 1024-dim embeddings are a good balance between quality and storage/compute cost. Older models used 384 or 768 dimensions; modern ones often use 1024 or more.

### What cosine similarity means
Given two vectors A and B:
- cosine similarity = 1.0 → identical direction → same meaning
- cosine similarity = 0.0 → perpendicular → no relationship
- cosine similarity = -1.0 → opposite direction → opposite meaning

In practice, retrieved chunks will have scores in the 0.3–0.8 range. A score of 0.4 against a ground truth answer is actually reasonable when comparing a short annotation to a long passage.

### The key concept to remember
Embeddings turn meaning into numbers. Similar meaning = similar numbers. FAISS finds the most similar numbers fast.

---

## 3. Hybrid Retrieval — FAISS + BM25 + CrossEncoder

### What each component does

**FAISS (dense retrieval)**
FAISS is Facebook AI Similarity Search — a library for fast nearest-neighbour search in high-dimensional vector spaces. Given a query vector, it returns the top-k most similar document vectors from the index. It is "dense" because every token contributes to every dimension of the vector.

Strength: captures semantic meaning. "Contract termination" and "agreement cancellation" are similar even though they share no words.
Weakness: can miss exact term matches. A query for "Section 12.3(b)" might not rank chunks containing that exact string highly if they are otherwise dissimilar.

**BM25 (sparse retrieval)**
BM25 (Best Match 25) is a classical information retrieval algorithm that scores documents based on term frequency and inverse document frequency. It is "sparse" because it only uses words that actually appear — most dimensions are zero.

Strength: exact term matching. Rare terms, codes, clause references, proper nouns — BM25 finds them reliably.
Weakness: no semantic understanding. "Termination" and "cancellation" are completely different words to BM25.

**Why combine them (hybrid retrieval)**
Dense retrieval finds semantically similar chunks. Sparse retrieval finds exact term matches. They fail in complementary ways — what one misses, the other often catches. Combining both and merging the results gives better coverage than either alone. This is called hybrid retrieval.

**CrossEncoder Reranker**
After FAISS returns top-20 and BM25 returns top-20, we have up to 40 candidate chunks (with duplicates removed). A CrossEncoder model then re-scores each candidate by looking at the (query, chunk) pair jointly — not just the chunk in isolation. This is much more accurate than the initial retrieval scores but too slow to run on the full corpus (hence the two-stage approach: fast retrieval to narrow to 40, then accurate reranking to narrow to 5).

### The key concept to remember
Use dense for semantic similarity, sparse for exact terms, reranker for precision. Each stage narrows the candidate set.

---

## 4. LangGraph and Stateful Pipeline Orchestration

### What it is
LangGraph is a framework for building stateful, multi-step AI pipelines as graphs. A graph has:
- **Nodes** — functions that do work (retrieve chunks, call an LLM, evaluate quality)
- **Edges** — connections between nodes (retrieve → reason → evaluate)
- **Conditional edges** — routing logic that sends the pipeline to different nodes based on the current state (if retrieval score is too low, go back to retrieve; if overall score is too low, go to human review)
- **State** — a shared data structure passed between every node (question, chunks, answer, scores, retry count)

### Why not a plain function chain?
A plain function chain works for a linear pipeline:
```python
chunks = retrieve(question)
answer = reason(question, chunks)
scores = evaluate(answer)
return answer
```

But production systems need:
- **Retries** — if retrieval quality is low, try again with wider search
- **Conditional routing** — different handling for different failure modes
- **Shared state** — retry count, original question, intermediate results all need to be accessible across steps
- **Looping** — the pipeline can go back to an earlier node

Implementing all of this in plain code produces deeply nested conditionals with state threaded through every function signature. LangGraph makes it explicit: every routing decision is a conditional edge, every piece of shared data is in `PipelineState`.

### The three routes in this project
After evaluation, the LangGraph router makes one of three decisions:
1. **PASS** (overall score ≥ 0.50) → return answer to user
2. **RETRY** (retrieval score < 0.05 and retry count ≤ 2) → go back to Agent 1 with expanded top-k
3. **HUMAN REVIEW** (everything else) → flag the answer with a warning

### The key concept to remember
LangGraph = state machine for LLM pipelines. Nodes do work. Edges define flow. Conditional edges define routing. State is shared.

---

## 5. Redis and Caching

### What Redis is
Redis (Remote Dictionary Server) is an in-memory key-value store. You store a value under a key, retrieve it by key, and optionally set an expiry (TTL — time to live) after which the key is automatically deleted.

It is in-memory, which makes reads and writes extremely fast (microseconds), but the data is lost if Redis restarts unless configured to persist to disk.

### Why cache LLM responses?
LLM API calls are:
- Slow (~9 seconds for LLaMA 3.3 70B)
- Expensive (charged per token)
- Deterministic for the same input (same question + same chunks = same answer)

If the same question is asked twice, there is no reason to call the LLM a second time. Cache the result under a key derived from the question and chunks. On the second request, return the cached result instantly.

### How cache keys work here
The cache key is a SHA256 hash of:
- The prompt version (so old cached answers are invalidated when the prompt changes)
- The question
- The chunk IDs in order (different chunks = different answer)

```python
content = f"{PROMPT_VERSION}:{question}:{'|'.join(chunk_ids)}"
key = f"reasoning:{hashlib.sha256(content.encode()).hexdigest()}"
```

This produces a fixed-length, collision-resistant key that uniquely identifies a (question, chunks, prompt version) triple.

### TTL (Time to Live)
Every cached result expires after 1 hour. This prevents stale answers being served indefinitely if the underlying data changes (new contracts ingested, index updated).

### Circuit breaker pattern
A circuit breaker prevents a failing dependency from being hammered on every request. Here:

1. Redis fails (connection refused, disk full, network issue)
2. Record `_redis_failure_time = time.monotonic()`
3. For the next 30 seconds, skip Redis entirely — don't even try to connect
4. After 30 seconds, try again

Without a circuit breaker, every request would attempt a Redis connection, wait for it to time out, then fall back to the LLM. With it, the failure is detected once and the system degrades gracefully until Redis recovers.

### The key concept to remember
Cache = don't repeat expensive work for the same input. TTL = don't serve stale results forever. Circuit breaker = fail fast instead of failing slow.

---

## 6. Observability — Logging, Tracing, and Metrics

### What observability means
Observability is the ability to understand what your system is doing from the outside, by looking at the data it produces. A system is observable if, when something goes wrong, you can diagnose the cause without adding new code.

The three pillars are:
- **Logs** — timestamped records of what happened ("cache HIT for question X", "LLM returned 312 tokens", "evaluation failed")
- **Traces** — end-to-end records of a single request passing through multiple services ("question X took 9.2s: retrieve 1.4s, reason 9.2s, evaluate 8.1s")
- **Metrics** — aggregated numerical measurements over time ("average latency last 5 minutes", "cache hit rate", "error rate")

### Logging in this project
Every module uses `get_logger(__name__)` from `observability/logging.py`. This gives each module a named logger (`agents.retrieval`, `agents.reasoning`, `agents.evaluation`) that makes it easy to filter logs by component.

Two formats:
- **Text** (`LOG_FORMAT=text`) — human-readable, for development
- **JSON** (`LOG_FORMAT=json`) — machine-parseable, for production tools like Datadog, CloudWatch, Loki

The JSON format produces structured records:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "logger": "agents.reasoning",
  "message": "Reasoning cache HIT for 'What is the termination...' — skipping LLM.",
  "module": "reasoning",
  "line": 792
}
```

Tools can index these and let you query: "show me all cache hits in the last hour" or "show me all requests where evaluation failed."

### LangSmith tracing
LangSmith is Anthropic/LangChain's observability platform for LLM applications. When `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set, LangGraph automatically sends a trace for every pipeline run — no code changes needed. Each trace shows:
- Which nodes ran and in what order
- What input each node received
- What output each node produced
- How long each node took
- Token usage per LLM call

This is invaluable for debugging. If a user reports a bad answer, you look up the trace for that request and see exactly what happened at every step.

### Per-agent timing
Even without LangSmith, every node logs its own duration:
```
Node: retrieve done in 1423ms — 5 chunks returned.
Node: reason done in 9210ms — answer_found=True.
Node: evaluate done in 7840ms — overall=0.78 passed=True.
```

When something is slow, you look at the logs and immediately know which node to investigate.

### The key concept to remember
Logs tell you what happened. Traces tell you what happened end-to-end for one request. Metrics tell you how the system is behaving over time. You need all three to operate a system you didn't just build.

---

## 7. Evaluation in LLM Systems

### Why evaluation is hard
Traditional software has deterministic outputs — given the same input, you get the same output, and you can write tests that check exact values. LLMs produce probabilistic, open-ended text. You can't write `assert answer == "The contract terminates with 30 days notice."` — the answer might be correct but worded differently.

### The three scores

**Retrieval score**
Measures whether Agent 1 found the right chunks. Compared against ground truth answers from DuckDB using embedding cosine similarity. A chunk is considered a hit if its embedding has cosine similarity ≥ 0.4 with a ground truth answer.

This is only meaningful when ground truth exists. For questions without a matching ground truth in the database, retrieval score defaults to 0.5 — not real signal, just a neutral placeholder. The `retrieval_score_available` flag in the response tells you which case you're in.

**Faithfulness score**
Measures whether the answer stays within what the source chunks say. The LLM judge is given each sentence of the answer alongside the chunk it cites and asked: "Is this claim supported by the source?" This catches hallucination — where the LLM adds information not present in the source material.

**Relevance score**
Measures whether the answer actually addresses the question. A technically faithful answer that doesn't answer what was asked still fails this check. The LLM judge is given the question and the full answer and asked to score how well the answer addresses the question.

### Why use an LLM as a judge?
Traditional metrics like BLEU and ROUGE compare text overlap between a generated answer and a reference answer. They are fast and deterministic but poor at capturing semantic quality — an answer can be completely correct but worded differently from the reference and score 0.

LLM-as-judge evaluates meaning, not surface form. "The agreement terminates with 30 days written notice" and "Either party may cancel the contract by providing 30 days notice in writing" would score 0 on BLEU/ROUGE against each other but near 1.0 from an LLM judge.

### Self-grading bias
If the same model generates the answer and evaluates it, it tends to rate its own outputs highly. This is analogous to a student grading their own exam — you are likely to interpret your own ambiguous answers charitably.

Using a different model family (Gemma 3 4B instead of LLaMA 3.3 70B) removes this bias. Gemma has different training data, different architecture, different reasoning patterns. What LLaMA confidently asserts, Gemma may question.

### The key concept to remember
Evaluate retrieval and generation separately so you know which one failed. Use a different model as the judge to avoid self-grading bias. LLM-as-judge captures meaning; token-overlap metrics do not.

---

## 8. API Design and Security Patterns

### FastAPI and Pydantic
FastAPI is a Python web framework that automatically generates OpenAPI documentation from your code. It uses Pydantic for request and response validation — if a request is missing a required field or has the wrong type, FastAPI rejects it with a clear error before your code runs.

This is important for reliability: you never deal with malformed requests inside your business logic. Validation happens at the boundary.

### API Key Authentication
The `/ask` and `/contracts` endpoints require an `X-API-Key` header when `API_KEY` is set in the environment. If `API_KEY` is not set, authentication is skipped (dev mode). This allows the same code to run locally without configuration and securely in production with a key set.

```
X-API-Key: your_key_here
```

The check is implemented as a FastAPI dependency:
```python
async def verify_api_key(key: str = Security(_api_key_header)) -> None:
    if API_KEY is None:
        return  # dev mode — no key required
    if key != API_KEY:
        raise HTTPException(status_code=401, ...)
```

A single shared key is the simplest form of authentication. It protects the endpoint from public access but doesn't give per-user control. The production upgrade path is OAuth 2.0 / JWT tokens with per-user keys.

### Rate Limiting
SlowAPI (a FastAPI wrapper around the `limits` library) limits how many requests a client can make per time window. Here: 10 requests per minute per IP. This prevents:
- Accidental loops from consuming the entire NVIDIA API quota
- Deliberate abuse from a single source exhausting the service

### CORS (Cross-Origin Resource Sharing)
Browsers block requests from one origin (e.g. `your-app.com`) to a different origin (e.g. `api.your-app.com`) unless the server explicitly allows it. CORS headers tell the browser which origins are permitted.

The middleware here allows requests from HuggingFace Spaces domains:
```python
allow_origin_regex=r"https://[a-zA-Z0-9-]+\.hf\.space"
```

Using a regex instead of a wildcard (`*`) means only HuggingFace Space domains are allowed, not arbitrary origins. A wildcard would allow any website to call the API from a user's browser.

### The key concept to remember
Validate at the boundary (Pydantic). Authenticate with keys (API key or JWT). Rate limit to prevent abuse. CORS controls which browser origins can call your API.

---

## 9. DuckDB and Ground Truth Storage

### What DuckDB is
DuckDB is an in-process analytical SQL database — it runs inside your Python process with no separate server, reads and writes to a single file, and is optimised for analytical queries (aggregations, joins, range scans) rather than high-concurrency transactional writes.

It is essentially SQLite but designed for analytics. You query it with standard SQL and get results as pandas DataFrames or Python lists.

### How it is used in this project
Three tables:

**`chunks`** — every text chunk from every contract, with metadata (contract title, position, source file). FAISS stores the vectors; DuckDB stores the text and metadata.

**`ground_truth`** — expert-annotated question-answer pairs from the CUAD dataset. Used by the evaluation agent to compute retrieval scores. ~22,000 rows covering 41 question types across 510 contracts.

**`evaluations`** — every EvaluationResult from every pipeline run, with timestamp. This is the evaluation history — you can query it to see how scores change over time, identify questions that consistently fail, or detect regressions after changing a prompt.

### Parameterized queries
Every SQL query that uses user input uses parameterized queries — placeholders (`?`) instead of string interpolation:

```python
# Safe — parameters passed separately
conn.execute("SELECT * FROM ground_truth WHERE LOWER(question) LIKE ?", [f"%{keyword}%"])

# Unsafe — never do this
conn.execute(f"SELECT * FROM ground_truth WHERE question LIKE '%{keyword}%'")
```

String interpolation is SQL injection. If `keyword` contains `%'; DROP TABLE ground_truth; --`, the unsafe version executes that. The parameterized version treats the entire string as data, not code.

### The key concept to remember
DuckDB = SQLite for analytics. No server, single file, full SQL. Parameterized queries prevent SQL injection. The evaluations table is how you track quality over time.

---

## 10. Docker and Containerisation

### What Docker is
Docker packages your application and all its dependencies (Python, libraries, config files) into a container image — a self-contained unit that runs identically everywhere. On your laptop, on a cloud server, on HuggingFace Spaces — same image, same behaviour.

Without Docker, deploying a Python application means installing the right Python version, the right library versions, the right system dependencies, and configuring the environment on every machine. With Docker, all of that is in the `Dockerfile` and reproducible with one command.

### How it works here
The `Dockerfile`:
1. Starts from a Python 3.11 base image
2. Installs system dependencies
3. Copies `requirements-docker.txt` and installs Python packages
4. Copies the application code
5. Exposes ports 7860 (Streamlit) and 8000 (FastAPI)
6. Runs `start.sh` which starts both services

`start.sh` waits up to 60 seconds for the FastAPI server to be ready (polls `/health`) before starting Streamlit. If FastAPI doesn't come up, it exits with an error rather than starting a UI that points at a broken API.

### HuggingFace Spaces deployment
HuggingFace Spaces with `sdk: docker` runs your Docker container directly. The `app_port: 7860` in the README frontmatter tells HuggingFace to expose that port as the public URL. The `deploy.yml` GitHub Action pushes to the `hf` remote on every merge to main, triggering a rebuild.

### The key concept to remember
Docker = reproducible environments. Same container image = same behaviour everywhere. `start.sh` is the entrypoint — it handles startup ordering so the UI never starts before the API is ready.

---

## Summary Table

| Concept | What it is | Why it's here |
|---|---|---|
| RAG | Retrieve relevant docs, then generate answer | Ground knowledge in source documents; enable citation |
| Vector embeddings | Text → numbers that represent meaning | Semantic search across 80k chunks |
| Hybrid retrieval | Dense + sparse + reranker | Better coverage than either alone |
| LangGraph | State machine for LLM pipelines | Conditional routing — retry, human review |
| Redis | In-memory key-value cache | Skip LLM calls for repeated questions |
| Circuit breaker | Fail fast after dependency failure | Prevent silent degradation when Redis is down |
| Observability | Logs + traces + metrics | Diagnose problems without guessing |
| LLM-as-judge | Model evaluates another model's output | Semantic quality measurement |
| Self-grading bias | Same model generating and evaluating | Different model families prevent this |
| DuckDB | In-process analytical SQL | Ground truth lookup, evaluation history |
| Parameterized queries | SQL with placeholders, not string concat | Prevent SQL injection |
| Docker | Reproducible containerised environment | Same behaviour everywhere, HuggingFace deployment |
| API key auth | X-API-Key header | Protect endpoint in production |
| Rate limiting | Max requests per minute per IP | Prevent quota exhaustion and abuse |
| CORS | Browser cross-origin policy | Allow only HuggingFace origins |
