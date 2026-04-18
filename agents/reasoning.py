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

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field

from groq import Groq
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GROQ_API_KEY, LLM_MODEL, MIN_CONFIDENCE_SCORE

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
    """
    The complete output of Agent 2.

    Contains the full answer as a list of cited sentences,
    plus metadata about the overall response quality.
    """
    question: str                        # original question
    sentences: list[CitedSentence]       # answer broken into cited sentences
    overall_confidence: float            # average confidence across sentences
    answer_found: bool                   # did LLM find a relevant answer?
    raw_answer: str                      # full answer as plain text
    chunks_used: list[str]               # chunk IDs cited, in order of use
    low_confidence: bool = field(init=False)  # auto-computed flag

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


def extract_json(raw: str) -> str:
    """
    Extract a JSON object from an LLM response robustly.

    LLMs are unpredictable — they may wrap JSON in code fences,
    add explanation text before or after, or mix prose and JSON.
    This function handles all common cases.

    Strategies tried in order:
        1. JSON inside ```json ... ``` fences
        2. JSON inside ``` ... ``` fences (no language tag)
        3. Outermost { ... } block in the response

    Args:
        raw: Raw string returned by the LLM

    Returns:
        Clean JSON string ready for json.loads()

    Raises:
        ValueError: If no JSON object can be found anywhere in the response
    """
    # Strategy 1 — JSON inside ```json ... ``` fences
    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Strategy 2 — JSON inside ``` ... ``` fences (no language tag)
    fence_match = re.search(r"```\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Strategy 3 — find the outermost { ... } block
    # Handles extra text before or after the JSON
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        return brace_match.group(0).strip()

    # Nothing worked — raise so caller can handle gracefully
    raise ValueError(
        f"No JSON object found in LLM response. "
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
Contract: {chunk['contract_title']}
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
# LLM caller
# ---------------------------------------------------------------------------

from groq import BadRequestError, AuthenticationError
from tenacity import retry_if_not_exception_type

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

    Decorated with @retry so transient API failures (network errors,
    rate limits, timeouts) are automatically retried up to 3 times
    with exponential backoff before raising to the caller.

    Groq is used because it's the fastest free LLM inference
    available — responses in ~1 second vs 10+ for others.

    Args:
        prompt: The complete prompt string
        model:  Groq model name from config

    Returns:
        Raw response string from the LLM

    Raises:
        Exception: After 3 failed attempts
    """
    client = Groq(api_key=GROQ_API_KEY)

    logger.info("Calling Groq LLM (model=%s)...", model)

    response = client.chat.completions.create(
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
        temperature=0.1,
        # 1500 tokens is enough for a detailed cited answer
        # In production, monitor for truncation and increase if needed
        max_tokens=1500,
    )

    raw = response.choices[0].message.content
    logger.info("Groq response received (%d chars).", len(raw))
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

        cited_sentences.append(CitedSentence(
            text=s.get("text", ""),
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

        # Validate API key exists before making any calls
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file."
            )

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
        logger.info("Reasoning about: '%s'", question[:80])
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

        # Build the prompt with numbered chunks
        prompt = build_prompt(question, chunks)

        # Call the LLM — retries automatically on failure
        raw = call_groq(prompt)

        # Parse into structured result with citations
        result = parse_response(raw, question, chunks)

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