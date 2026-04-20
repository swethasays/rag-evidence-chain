"""
agents/reasoning.py

Agent 2 — Reasoning

Takes the top-k chunks from Agent 1 and uses a Groq LLM to form
a complete, cited answer. Every sentence in the answer is linked
back to the exact chunk it came from — this is the evidence chain.

The LLM is prompted to respond in structured JSON so citations
are generated at the same time as the answer, not matched after.

Flow:
    chunks (from Agent 1)
        → build prompt with numbered chunks
        → call Groq LLM (with retry on failure)
        → extract JSON robustly
        → parse structured JSON response
        → return answer + evidence chain
"""

import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field

import redis

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CACHE_TTL,
    GROQ_API_KEY,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MIN_CONFIDENCE_SCORE,
    REDIS_URL,
)

from groq import Groq, BadRequestError, AuthenticationError
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt version — increment when build_prompt() changes
# ---------------------------------------------------------------------------

# Versioning ensures we can trace which prompt produced which answer.
# If you change build_prompt(), bump this version so old results
# stored in cache or logs are not confused with new ones.
PROMPT_VERSION = "v1.0"

# ---------------------------------------------------------------------------
# Groq client — lazy singleton
# ---------------------------------------------------------------------------

# Not created at module load — that would crash before API key validation.
# Created on first call to call_groq() and reused after that.
_groq_client: Groq | None = None


def get_groq_client() -> Groq:
    """
    Return the Groq client, creating it on first call.

    Lazy initialization means the module loads safely even if
    GROQ_API_KEY is not set yet. The error surfaces at call time
    with a clear message, not at import time with a cryptic crash.

    Returns:
        Shared Groq client instance

    Raises:
        ValueError: If GROQ_API_KEY is not set
    """
    global _groq_client

    if _groq_client is None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file."
            )
        _groq_client = Groq(api_key=GROQ_API_KEY)

    return _groq_client


# ---------------------------------------------------------------------------
# Redis client — lazy singleton
# ---------------------------------------------------------------------------

# Not created at module load — created on first cache access.
# Same pattern as Groq client — one connection, reused everywhere.
_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis | None:
    """
    Return the Redis client, creating and verifying it on first call.

    If Redis is unavailable, returns None instead of a broken client.
    On every call where client is None, retries the connection —
    so if Redis comes back online, the system reconnects automatically.

    Returns:
        Redis client if connected, None if unavailable

    Note:
        Callers must handle None — cache is disabled when Redis is down.
    """
    global _redis_client

    # If we have a working client, return it
    if _redis_client is not None:
        return _redis_client

    # Try to connect and verify
    try:
        client = redis.from_url(REDIS_URL)
        client.ping()
        _redis_client = client
        logger.info("Redis connection established at %s", REDIS_URL)
        return _redis_client
    except redis.RedisError as e:
        # Don't store broken client — next call will retry
        logger.warning(
            "Redis unavailable at %s — caching disabled: %s",
            REDIS_URL, e,
        )
        return None


def reset_redis_client() -> None:
    """Reset the Redis client so next call to get_redis_client() retries."""
    global _redis_client
    _redis_client = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CitedSentence:
    """
    A single sentence in the answer with its source citation.

    Every claim the LLM makes is wrapped in this structure so
    the UI can render a clickable evidence chain.
    """
    text: str            # the sentence itself
    chunk_id: str        # which chunk this came from
    contract_title: str  # which contract that chunk belongs to
    chunk_text: str      # the source chunk text (shown on click)
    confidence: float    # how confident the LLM is (0.0 - 1.0)


@dataclass
class ReasoningResult:
    question: str
    sentences: list[CitedSentence]
    overall_confidence: float
    answer_found: bool
    raw_answer: str
    chunks_used: list[str]
    prompt_version: str = field(default_factory=lambda: PROMPT_VERSION)
    low_confidence: bool = field(init=False)

    def __post_init__(self):
        # Auto-flag if overall confidence is below threshold
        # Agent 3 uses this to route answer to human review
        self.low_confidence = self.overall_confidence < MIN_CONFIDENCE_SCORE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a float between min_val and max_val.

    Used to sanitize LLM confidence scores — the model might return
    values like 1.5 or -0.1 which would break downstream logic.

    Args:
        value:   The value to clamp
        min_val: Lower bound (inclusive)
        max_val: Upper bound (inclusive)

    Returns:
        value clamped to [min_val, max_val]
    """
    return max(min_val, min(max_val, value))

def clean_title(title: str) -> str:
    """
    Extract a readable contract name from a raw CUAD filename.

    CUAD titles look like:
        ADAMSGOLFINC_03_21_2005-EX-10.17-ENDORSEMENT AGREEMENT
    We want:
        ENDORSEMENT AGREEMENT

    Args:
        title: Raw CUAD contract title

    Returns:
        Clean readable contract name
    """
    if " - " in title:
        return title.split(" - ")[-1].strip()
    parts = title.split("_")
    return " ".join(parts[-3:]).strip() if len(parts) > 3 else title


def extract_json(raw: str) -> str:
    """
    Extract a JSON object from an LLM response robustly.

    All three strategies use brace counting to handle nested objects
    correctly. Non-greedy regex (.*?) would stop at the first closing
    brace — wrong for nested JSON like {"sentences": [{"text": "..."}]}.

    Strategies tried in order:
        1. Find the { after ```json fence, count braces to closing }
        2. Find the { after ``` fence (no language tag), count braces
        3. Find the first { anywhere in the response, count braces

    Args:
        raw: Raw string returned by the LLM

    Returns:
        Clean JSON string ready for json.loads()

    Raises:
        ValueError: If no valid balanced JSON object can be found
    """

    def count_braces(text: str, start: int) -> str | None:
        """
        Extract a balanced JSON object starting at index `start`.

        Walks forward from `start`, counting { and }.
        Returns the complete object when depth returns to 0.
        Returns None if braces are unbalanced.

        Args:
            text:  Full string to search
            start: Index of the opening { to start from

        Returns:
            Balanced JSON string, or None if unbalanced
        """
        depth = 0
        for i, char in enumerate(text[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1].strip()
        return None  # unbalanced — no matching closing brace

    # Strategy 1 — find { after ```json fence
    fence1 = re.search(r"```json\s*(\{)", raw, re.DOTALL)
    if fence1:
        result = count_braces(raw, fence1.start(1))
        if result:
            return result

    # Strategy 2 — find { after ``` fence (no language tag)
    fence2 = re.search(r"```\s*(\{)", raw, re.DOTALL)
    if fence2:
        result = count_braces(raw, fence2.start(1))
        if result:
            return result

    # Strategy 3 — find the first { anywhere in the response
    start = raw.find("{")
    if start != -1:
        result = count_braces(raw, start)
        if result:
            return result

    # Nothing worked — raise so caller can handle gracefully
    raise ValueError(
        f"No valid balanced JSON object found in LLM response. "
        f"First 200 chars: {raw[:200]}"
    )

def resolve_chunk(
    chunk_num: int,
    chunk_map: dict,
    question: str,
) -> tuple:
    """
    Resolve a 1-based chunk number to a chunk dict.

    LLMs occasionally hallucinate chunk numbers that don't exist.
    Rather than silently falling back to the wrong chunk, we warn
    and return None so the caller can skip the sentence entirely.

    Args:
        chunk_num: 1-based chunk number from the LLM response
        chunk_map: Mapping of {chunk_number: chunk dict}
        question:  Original question (included in warning for context)

    Returns:
        (chunk dict, hallucinated: bool)
        If hallucinated is True, chunk is None — caller must skip.
    """
    chunk = chunk_map.get(chunk_num)

    if chunk is None:
        logger.warning(
            "LLM cited chunk number %d which does not exist "
            "(only %d chunks provided). Skipping sentence. "
            "Question was: '%s'",
            chunk_num,
            len(chunk_map),
            question[:80],
        )
        return None, True

    return chunk, False

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _get_cache_key(question: str, chunk_ids: list[str]) -> str:
    """
    Build a stable cache key from question + chunk IDs + prompt version.

    Including chunk IDs means the cache invalidates automatically
    if retrieval results change after re-ingestion.
    Including prompt version means old cached answers don't get
    served after build_prompt() changes.

    Args:
        question:  The user's question
        chunk_ids: IDs of chunks passed to the LLM

    Returns:
        SHA256 hex string safe for use as a Redis key
    """
    # Sort chunk_ids for consistency — order shouldn't affect cache hit
    content = f"{PROMPT_VERSION}:{question}:{'|'.join(sorted(chunk_ids))}"
    return f"reasoning:{hashlib.sha256(content.encode()).hexdigest()}"

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build the prompt sent to the LLM.

    We number each chunk so the LLM can reference them by number
    in its citations. The prompt strictly instructs the LLM to:
      1. Only use information from the provided chunks
      2. Cite which chunk each sentence came from
      3. Respond in JSON format only — no markdown, no preamble

    Args:
        question: The user's question
        chunks:   Top-k chunks from Agent 1, each with id and text

    Returns:
        Complete prompt string
    """
    # Format each chunk with a number for easy LLM reference
    chunk_context = ""
    for i, chunk in enumerate(chunks, 1):
        chunk_context += f"""
CHUNK {i} (ID: {chunk['id']})
Contract: {clean_title(chunk['contract_title'])}
Text: {chunk['text']}
{"─" * 40}
"""

    prompt = f"""You are a legal contract analysis assistant.
Your job is to answer questions about legal contracts accurately and cite your sources.

CONTEXT — use ONLY the chunks below to answer:
{chunk_context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer using ONLY information from the chunks above
2. Break your answer into individual sentences
3. For each sentence, cite which CHUNK NUMBER it came from
4. If a sentence uses information from multiple chunks, cite the most relevant one
5. If the answer cannot be found in the chunks, set "answer_found" to false
6. Be concise — one sentence per key point

RESPOND IN THIS EXACT JSON FORMAT — no preamble, no markdown, just JSON:
{{
  "answer_found": true,
  "overall_confidence": 0.85,
  "sentences": [
    {{
      "text": "Your sentence here.",
      "chunk_number": 1,
      "confidence": 0.9
    }},
    {{
      "text": "Another sentence here.",
      "chunk_number": 2,
      "confidence": 0.8
    }}
  ]
}}

If the answer is not in the chunks:
{{
  "answer_found": false,
  "overall_confidence": 0.0,
  "sentences": []
}}"""

    return prompt

# ---------------------------------------------------------------------------
# Rate limiter — token bucket
# ---------------------------------------------------------------------------

class TokenBucket:
    """
    Simple token bucket rate limiter for LLM API calls.

    Limits how many Groq calls happen per second across all threads.
    When the bucket is empty, calls wait until tokens refill —
    preventing rate limit errors under concurrent load.

    Args:
        rate:     Tokens added per second (controls sustained rate)
        capacity: Maximum tokens (controls burst size)
    """

    def __init__(self, rate: float = 2.0, capacity: float = 5.0):
        self._rate = rate            # tokens added per second
        self._capacity = capacity    # maximum tokens in bucket
        self._tokens = capacity      # start full
        self._lock = threading.Lock()
        self._last_refill = time.monotonic()

    def acquire(self) -> None:
        """
        Wait until a token is available, then consume one.

        Blocks the calling thread if the bucket is empty.
        Safe to call from multiple threads simultaneously.
        """
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
            # Bucket empty — wait briefly before retrying
            time.sleep(0.1)

    def _refill(self) -> None:
        """Add tokens proportional to elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._capacity,
            self._tokens + elapsed * self._rate,
        )
        self._last_refill = now


# Shared rate limiter — 2 calls/second sustained, burst up to 5
_rate_limiter = TokenBucket(rate=2.0, capacity=5.0)

# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    # Never retry permanent errors — only transient ones
    retry=retry_if_not_exception_type((BadRequestError, AuthenticationError)),
    before_sleep=lambda retry_state: logger.warning(
        "Groq API call failed. Retrying attempt %d/3...",
        retry_state.attempt_number,
    ),
)

def call_groq(prompt: str, model: str = LLM_MODEL) -> str:
    """
    Send a prompt to Groq and return the raw response text.

    Uses a module-level client created once at startup — not on
    every call. Retries up to 3 times on transient errors only.
    Permanent errors (bad model, bad API key) raise immediately.

    Args:
        prompt: The complete prompt string
        model:  Groq model name from config

    Returns:
        Raw response string from the LLM

    Raises:
        BadRequestError:      Immediately — permanent error, no retry
        AuthenticationError:  Immediately — permanent error, no retry
        Exception:            After 3 failed attempts
    """
    # Acquire rate limit token before calling API
    _rate_limiter.acquire()

    logger.info("Calling Groq LLM (model=%s)...", model)

    response = get_groq_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise legal contract analyst. "
                    "Always respond in valid JSON only. "
                    "Never include markdown, preamble, or explanation outside the JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        # Low temperature = deterministic, consistent JSON output
        # High temperature would risk creative but malformed responses
        temperature=LLM_TEMPERATURE,
        # Enough for a detailed cited answer
        # In production, monitor for truncation and increase if needed
        max_tokens=LLM_MAX_TOKENS,
    )

    raw = response.choices[0].message.content

    # Log token usage — every token costs money in production
    usage = response.usage
    logger.info(
        "Groq token usage — prompt: %d, completion: %d, total: %d",
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
    )

    return raw


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_response(
    raw: str,
    question: str,
    chunks: list[dict],
) -> ReasoningResult:
    """
    Parse the LLM's JSON response into a ReasoningResult.

    Handles all failure modes defensively:
        - Malformed JSON         → logs error, returns empty result
        - Extra text around JSON → extract_json() strips it
        - Hallucinated chunk num → resolve_chunk() warns and skips
        - Confidence out of range → clamp() fixes silently
        - Empty chunks list      → guards at top

    Args:
        raw:      Raw string from Groq
        question: Original question
        chunks:   Chunks passed to the LLM (used to resolve citations)

    Returns:
        ReasoningResult with cited sentences — never raises
    """
    # Guard — empty chunks would crash chunk_map lookup below
    if not chunks:
        logger.error("parse_response called with empty chunks list.")
        return ReasoningResult(
            question=question,
            sentences=[],
            overall_confidence=0.0,
            answer_found=False,
            raw_answer=raw,
            chunks_used=[],
        )

    # Extract JSON robustly — handles fences, extra text, etc.
    try:
        cleaned = extract_json(raw)
    except ValueError as e:
        logger.error("Could not extract JSON from LLM response: %s", e)
        logger.warning(
            "User will see empty answer. "
            "Consider checking the system prompt. Raw: %s", raw[:300]
        )
        return ReasoningResult(
            question=question,
            sentences=[],
            overall_confidence=0.0,
            answer_found=False,
            raw_answer=raw,
            chunks_used=[],
        )

    # Parse extracted JSON string
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("JSON decode failed after extraction: %s", e)
        logger.warning("Cleaned JSON was: %s", cleaned[:300])
        return ReasoningResult(
            question=question,
            sentences=[],
            overall_confidence=0.0,
            answer_found=False,
            raw_answer=raw,
            chunks_used=[],
        )

    # Build 1-based chunk number → chunk dict lookup
    # LLM uses 1-based numbers, Python lists are 0-based
    chunk_map = {i + 1: chunk for i, chunk in enumerate(chunks)}

    cited_sentences = []
    chunks_used = []        # list preserves citation order
    chunks_used_set = set() # set for O(1) deduplication check

    for s in data.get("sentences", []):
        chunk_num = s.get("chunk_number", 1)

        # Resolve chunk — skip sentence if LLM hallucinated chunk number 
        chunk, hallucinated = resolve_chunk(chunk_num, chunk_map, question) 
        if hallucinated:
            continue

        # Skip empty sentences — LLM occasionally returns blank text 
        sentence_text = s.get("text", "").strip()
        if not sentence_text:
            logger.warning("Skipping empty sentence from LLM response.")
            continue

        cited_sentences.append(CitedSentence(
            text=sentence_text,
            chunk_id=chunk["id"],
            contract_title=chunk["contract_title"],
            chunk_text=chunk["text"],
            # Clamp to [0.0, 1.0] — LLM might return out-of-range values
            confidence=clamp(s.get("confidence", 0.5), 0.0, 1.0),
        ))

        # Track cited chunks in order — skip duplicates
        if chunk["id"] not in chunks_used_set:
            chunks_used.append(chunk["id"])
            chunks_used_set.add(chunk["id"])

    # Build plain text answer by joining all sentences
    raw_answer = " ".join(s.text for s in cited_sentences)

    return ReasoningResult(
        question=question,
        sentences=cited_sentences,
        overall_confidence=clamp(
            data.get("overall_confidence", 0.0), 0.0, 1.0
        ),
        answer_found=data.get("answer_found", False),
        raw_answer=raw_answer,
        chunks_used=chunks_used,
    )


# ---------------------------------------------------------------------------
# Reasoning Agent
# ---------------------------------------------------------------------------

class ReasoningAgent:
    """
    Agent 2 — reads retrieved chunks and forms a cited answer.

    Takes the top-k chunks from Agent 1, sends them to Groq with
    a structured prompt, and parses the response into a ReasoningResult
    where every sentence is linked to its source chunk.

    Usage:
        agent = ReasoningAgent()
        result = agent.reason(question, chunks)
    """

    def __init__(self):
        logger.info("Initialising ReasoningAgent...")
        # API key validated lazily on first call to call_groq()
        logger.info("ReasoningAgent ready.")
        

    def reason(
        self,
        question: str,
        chunks: list[dict],
    ) -> ReasoningResult:
        """
        Form a cited answer from retrieved chunks.

        Pipeline:
            1. Build prompt with numbered chunks
            2. Call Groq LLM (retries on failure)
            3. Extract JSON robustly
            4. Parse into ReasoningResult with cited sentences
            5. Flag low confidence answers for human review

        Args:
            question: The user's natural language question
            chunks:   Top-k chunks from Agent 1

        Returns:
            ReasoningResult with cited sentences and confidence scores
        """
        logger.info(
            "Reasoning about: '%s' (prompt_version=%s)",
            question[:80],
            PROMPT_VERSION,
        )
        logger.info("Using %d chunks as context.", len(chunks))

        # Guard — no chunks means nothing to reason about
        if not chunks:
            logger.warning("No chunks provided — cannot form an answer.")
            return ReasoningResult(
                question=question,
                sentences=[],
                overall_confidence=0.0,
                answer_found=False,
                raw_answer="No relevant chunks were retrieved.",
                chunks_used=[],
            )

        # Check Redis cache — same question + chunks = same answer
        cache_key = _get_cache_key(question, [c["id"] for c in chunks])
        cached = self._cache_get(cache_key)
        if cached:
            logger.info("Cache HIT — returning cached answer instantly.")
            return cached

        # Build the prompt with numbered chunks
        prompt = build_prompt(question, chunks)

        # Call the LLM — retries automatically on transient failures
        raw = call_groq(prompt)

        # Parse into structured result with citations
        result = parse_response(raw, question, chunks)

        # Cache result — next identical question skips the LLM entirely
        self._cache_set(cache_key, result)

        # Log the outcome clearly
        if result.answer_found:
            logger.info(
                "Answer formed: %d sentences, confidence=%.2f, low_confidence=%s",
                len(result.sentences),
                result.overall_confidence,
                result.low_confidence,
            )
        else:
            logger.warning(
                "LLM could not find answer in provided chunks. "
                "Consider expanding retrieval top-k."
            )

        return result
    

    def _cache_get(self, key: str) -> "ReasoningResult | None":
        """
        Retrieve a cached ReasoningResult from Redis.

        Returns None on cache miss, Redis unavailable, or connection error.
        Redis failures are non-fatal — system falls back to LLM.
        """
        r = get_redis_client()
        if r is None:
            return None  # Redis down — skip cache, call LLM

        try:
            data = r.get(key)
            if data:
                logger.info("Cache HIT for key: %s...", key[:20])
                return self._deserialize(data)
        except redis.RedisError as e:
            # Connection dropped mid-session — reset so next call retries
            reset_redis_client()
            logger.warning("Redis read failed, resetting connection: %s", e)
        return None

    def _cache_set(self, key: str, result: "ReasoningResult") -> None:
        """
        Store a ReasoningResult in Redis with TTL expiry.

        Redis failures are non-fatal — answer still returned to user.
        """
        r = get_redis_client()
        if r is None:
            return  # Redis down — skip cache silently

        try:
            r.setex(key, CACHE_TTL, self._serialize(result))
            logger.info("Result cached with TTL=%ds.", CACHE_TTL)
        except redis.RedisError as e:
            # Connection dropped mid-session — reset so next call retries
            reset_redis_client()
            logger.warning("Redis write failed, resetting connection: %s", e)

    def _serialize(self, result: "ReasoningResult") -> str:
        """Serialize ReasoningResult to JSON string for Redis storage."""
        return json.dumps({
            "question": result.question,
            "overall_confidence": result.overall_confidence,
            "answer_found": result.answer_found,
            "raw_answer": result.raw_answer,
            "chunks_used": result.chunks_used,
            "prompt_version": result.prompt_version,
            "sentences": [
                {
                    "text": s.text,
                    "chunk_id": s.chunk_id,
                    "contract_title": s.contract_title,
                    "chunk_text": s.chunk_text,
                    "confidence": s.confidence,
                }
                for s in result.sentences
            ],
        })

    def _deserialize(self, data: bytes) -> "ReasoningResult":
        """Deserialize JSON bytes from Redis back into ReasoningResult."""
        d = json.loads(data)
        return ReasoningResult(
            question=d["question"],
            sentences=[
                CitedSentence(
                    text=s["text"],
                    chunk_id=s["chunk_id"],
                    contract_title=s["contract_title"],
                    chunk_text=s["chunk_text"],
                    confidence=s["confidence"],
                )
                for s in d["sentences"]
            ],
            overall_confidence=d["overall_confidence"],
            answer_found=d["answer_found"],
            raw_answer=d["raw_answer"],
            chunks_used=d["chunks_used"],
            prompt_version=d.get("prompt_version", "unknown"),
        )

# ---------------------------------------------------------------------------
# Entry point — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from agents.retrieval import RetrievalAgent

    # Step 1 — retrieve relevant chunks using Agent 1
    retrieval = RetrievalAgent()
    question  = "What is the termination clause?"
    chunks    = retrieval.search(question)

    # Step 2 — reason over those chunks using Agent 2
    reasoning = ReasoningAgent()
    result    = reasoning.reason(question, chunks)

    # Step 3 — print the full evidence chain
    print(f"\n{'═' * 60}")
    print(f"QUESTION: {question}")
    print(f"{'═' * 60}")
    print(f"Answer found    : {result.answer_found}")
    print(f"Confidence      : {result.overall_confidence:.2f}")
    print(f"Low confidence  : {result.low_confidence}")
    print(f"Chunks used     : {len(result.chunks_used)}")
    print(f"\n{'─' * 60}")
    print("EVIDENCE CHAIN:")
    print(f"{'─' * 60}\n")

    for i, sentence in enumerate(result.sentences, 1):
        print(f"Sentence {i}: {sentence.text}")
        print(f"  ← Source    : {sentence.contract_title[:60]}")
        print(f"  ← Chunk     : {sentence.chunk_id[:40]}...")
        print(f"  ← Confidence: {sentence.confidence:.2f}")
        print()

    print(f"{'─' * 60}")
    print(f"FULL ANSWER:\n{result.raw_answer}")