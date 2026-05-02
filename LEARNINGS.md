# What I Learned Building RAG Evidence Chain

This document is a personal record of the real problems I hit, the decisions I made, and what each one taught me. Written so future-me can revise quickly without re-reading the code.

---

## 1. Measuring Latency Before Optimising

### What I built first
The evaluation agent called two LLM judge functions — one for faithfulness, one for relevance — sequentially, with a `time.sleep()` between them to avoid rate limits.

### What went wrong
End-to-end query time was ~42 seconds. That felt wrong. I assumed the LLM itself was slow.

### How I investigated
I added timing logs around each step and ran a query. The LLM calls themselves were ~9s each. But because they ran one after the other, the total was 9 + 9 = 18s just for evaluation — plus the sleep on top. The two judge calls are completely independent (faithfulness doesn't need relevance's result), so there was no reason to run them sequentially.

### What I changed
Replaced the sequential calls with `concurrent.futures.ThreadPoolExecutor(max_workers=2)` — both judge calls submit at the same time and the code waits for both to finish. Total evaluation time dropped from ~18s to ~9s (the slower of the two calls).

Also simplified the faithfulness prompt — it was sending full chunk text (~780 tokens) when 120 characters was enough for the judge to make a decision. That alone cut tokens by ~65%.

### What I learned
- **Profile before optimising.** I assumed the model was the bottleneck. It wasn't — the bottleneck was sequential independent calls.
- **Parallelise independent work.** If two things don't depend on each other's output, they should run at the same time.
- **Prompt length affects latency.** Shorter prompts with the same signal are strictly better. The judge doesn't need the full chunk — it needs enough to verify the claim.

---

## 2. Silent Cache Failures Are Worse Than Loud Ones

### What I built
Redis caching for both reasoning (Agent 2) and evaluation (Agent 3). Same question + same chunks = skip the LLM entirely. TTL 1 hour.

### What went wrong
Cache was configured, the code was correct, but repeat queries were still taking ~20 seconds. No errors appeared anywhere in the logs.

### How I investigated
I checked whether Redis was actually writing. It wasn't. Dug further — the disk was full (`/tmp` had filled up). Redis couldn't write to disk, so every `SET` was silently failing. The app kept falling through to the LLM on every request without any indication something was wrong.

### What I changed
Two things:

1. **Freed disk space** (deleted unused apps, cleared `/tmp`).
2. **Added a circuit breaker** — after a Redis failure, the code records `_redis_failure_time` and skips Redis for 30 seconds before retrying. If Redis is down, it logs a clear warning instead of silently degrading. This pattern is called a circuit breaker — it prevents hammering a broken service on every request.

Also added explicit cache hit logging so you can see in the logs whether a response came from Redis or the LLM:
```
Reasoning cache HIT for 'What is the termination...' — skipping LLM.
```

### What I learned
- **Silent failures are the hardest bugs.** The system looked healthy — no exceptions, responses returned — but was silently bypassing its most important optimisation.
- **Always log cache hits and misses explicitly.** Without that log line, I had no way to tell whether caching was working from the outside.
- **Infrastructure can fail for non-code reasons.** Full disk, network partition, Redis restart — none of these are bugs in your code but all of them break your system. Defensive fallbacks (circuit breaker, graceful degradation) are part of the code.

---

## 3. Why Two Model Families for Generate and Evaluate

### The design decision
Used LLaMA 3.3 70B (NVIDIA NIM) to generate answers and Gemma 3 4B (different architecture, different training data) to evaluate them.

### Why not use the same model for both?
If LLaMA 3.3 generates an answer and LLaMA 3.3 evaluates it, the judge is likely to confirm its own outputs. Models have consistent reasoning patterns and consistent blind spots. If the generator made a subtly wrong inference, the same model evaluating it would often make the same inference and score it as correct.

Using Gemma 3 4B — a completely different architecture trained on different data — breaks that correlation. The judge has no "memory" of generating the answer and no systematic tendency to agree with LLaMA's reasoning style.

### What I learned
- **Self-evaluation is unreliable.** This applies to LLMs the same way it applies to humans — you miss your own mistakes more than someone else does.
- **Diversity in evaluation systems has real value.** Different model families bring different failure modes. What one hallucinates, another may catch.
- **Small judge models work well for binary/scalar evaluation.** Gemma 3 4B is much cheaper and faster than LLaMA 3.3 70B. For the task "is this sentence supported by the source?" you don't need a 70B model.

---

## 4. Splitting Evaluation Into Three Independent Scores

### The design decision
Rather than a single "quality score," the evaluation agent produces three separate scores:
- **Retrieval score** — did Agent 1 find the right chunks? (measured against DuckDB ground truth using embedding cosine similarity)
- **Faithfulness score** — does the answer stay within what the chunks say? (LLM judge)
- **Relevance score** — does the answer actually address the question? (LLM judge)

### Why this matters
A bad answer could be a retrieval failure or a generation failure. Without separating them, you just know "bad answer" with no direction to fix it.

A low retrieval score tells you Agent 1 didn't find relevant chunks — fix: expand top-k, tune the reranker, or improve the query.
A low faithfulness score tells you Agent 2 hallucinated — fix: tighten the prompt, reduce temperature, or add a stronger instruction to only use source material.
A low relevance score tells you the answer technically came from the chunks but didn't address the question — fix: adjust the prompt to focus on the specific question.

Each score points to a different part of the system.

### What I learned
- **Diagnostic scores are more useful than aggregate scores.** A single number hides where the problem is.
- **Retrieval and generation fail independently.** Good retrieval with bad generation, and bad retrieval with good generation, are both possible. You need to measure them separately.
- **Ground truth data is rare and valuable.** The CUAD dataset has expert-annotated answers, which made it possible to compute a real retrieval score. Most production RAG systems don't have this — they default to 0.5 and rely only on faithfulness and relevance. Having ground truth is a significant advantage.

---

## 5. LangGraph Earns Its Complexity Through Routing

### What I tried first (mentally)
Three agents in sequence could be a simple function chain:
```python
chunks = retrieval.search(question)
answer = reasoning.reason(question, chunks)
scores = evaluation.evaluate(answer)
return answer
```

### Why that breaks
Production systems need conditional logic: retry if retrieval score is too low, flag for human review if overall confidence is below threshold, cap retries to prevent infinite loops. In plain code this becomes deeply nested if/else with shared state threaded through every function.

LangGraph models the pipeline as a state graph — each agent is a node, each routing decision is a conditional edge. The state (question, chunks, retry count, scores) is shared cleanly. Adding a new route (e.g. "escalate if contract is high value") is adding one edge, not refactoring five functions.

### What I learned
- **Orchestration frameworks add value at complexity boundaries.** For a simple linear chain, they're overhead. For anything with branching, retries, or shared state, they pay for themselves.
- **Making state explicit forces you to think about what data flows where.** `PipelineState` as a TypedDict means every agent declares what it reads and writes. That's documentation and type-checking at the same time.
- **Conditional routing is a first-class concern in production RAG.** "What do you do when the answer is bad?" is not an edge case — it's the core of a reliable system.

---

## 6. The Gap Between "It Works" and "You Can Operate It"

### What surprised me
The core RAG pipeline — retrieve, reason, evaluate — was working relatively early. But there was a large gap between "working locally" and "something you'd trust in production." That gap was filled by:

- **Redis circuit breaker** — so cache failures don't silently degrade the system
- **API key authentication** — so the endpoint isn't open to anyone
- **Structured JSON logging** (`LOG_FORMAT=json`) — so logs are machine-parseable by tools like Datadog or CloudWatch
- **LangSmith tracing** — so every pipeline run is visible end-to-end: which chunks were retrieved, what the LLM received, how long each step took
- **Per-agent timing logs** — so you can see "retrieve: 1.4s, reason: 9.2s, evaluate: 8.1s" and know where to look when something is slow
- **Graceful startup** — `start.sh` waits up to 60 seconds for the API to be ready before starting the UI, and exits cleanly if it doesn't come up
- **`answer_found` guard** — if the LLM claims it found an answer but all cited sentences reference hallucinated chunk numbers, the answer is forced to `False`

None of these change what the app does for the user. All of them change whether you can run it reliably and debug it when something goes wrong.

### What I learned
- **Observability is not optional.** Without logs, tracing, and timing, debugging production issues means guessing.
- **Defensive coding is different from error handling.** Error handling catches exceptions. Defensive coding handles the cases where nothing throws but the system is still wrong — like a cache that silently stops writing.
- **"Production-ready" is a spectrum.** There is no checkbox. Each layer of resilience (caching, auth, retries, monitoring) moves the system further along the spectrum. The gaps table in the README is an honest acknowledgement of where this project sits.

---

## 7. Retrieval Is Harder Than It Looks

### What I built
Hybrid retrieval: FAISS dense search (semantic, using NVIDIA nv-embedqa-e5-v5 1024-dim embeddings) + BM25 sparse search (keyword matching) merged and reranked by a CrossEncoder.

### Why hybrid
Dense search (FAISS) finds semantically similar chunks but misses exact term matches. If someone asks about "Section 12.3(b)" and that exact string is in a chunk, BM25 finds it instantly while FAISS might not rank it highly. BM25 handles exact terms and rare words well but has no semantic understanding. Combining both and reranking with a CrossEncoder — which scores (query, chunk) pairs jointly — gives better top-5 results than either alone.

### What was surprising
Even with good retrieval, the CUAD ground truth scores stayed in the 0.1–0.3 range for most queries. This isn't a failure — CUAD ground truth answers are very specific clause-level annotations ("The agreement terminates upon 30 days written notice"), while retrieved chunks are broader passages. Cosine similarity between a short ground truth and a long passage will naturally be lower than similarity between two passages of similar length. The retrieval score threshold for triggering a retry had to be set very low (0.05) — only retry when chunks are essentially absent, not just imperfectly matched.

### What I learned
- **Hybrid retrieval consistently outperforms single-method retrieval.** The engineering cost is low (BM25 is fast and cheap), the benefit is real.
- **Evaluation metrics must match what you're measuring.** Comparing a 10-word ground truth answer to a 200-word retrieved passage using cosine similarity will produce low scores even when retrieval is working. The metric was measuring mismatch in passage length as much as retrieval quality.
- **Ground truth data shapes what you can measure.** Having CUAD annotations gave a real retrieval signal. Without them, the system falls back to 0.5 and relies entirely on the LLM judges — which is the reality for most RAG deployments.

---

## Summary

| Lesson | Short version |
|---|---|
| Measure before optimising | Profile first — the bottleneck is rarely where you think |
| Silent failures are dangerous | Log cache hits/misses, add circuit breakers |
| Separate generator from judge | Same model evaluating its own output introduces bias |
| Diagnostic scores > aggregate scores | Three scores tell you where to fix; one score just tells you it's broken |
| Orchestration earns its place at routing boundaries | Simple chains don't need LangGraph; conditional retry/review logic does |
| Operational concerns are code concerns | Auth, logging, tracing, graceful startup — these are part of the system |
| Hybrid retrieval beats single-method | BM25 + FAISS + CrossEncoder is worth the extra complexity |
