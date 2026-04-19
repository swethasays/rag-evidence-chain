"""
agents/evaluation.py

Agent 3 — Evaluation

Scores the quality of answers produced by Agent 2.
Measures retrieval and generation quality separately so we know
exactly where failures come from — a retrieval problem or a
generation problem.

Three scores:
    retrieval_score    — did Agent 1 find the right chunks?
    faithfulness_score — does the answer stay within the chunks?
    answer_relevance   — does the answer address the question?

Flow:
    ReasoningResult (from Agent 2)
        → check ground truth in DuckDB
        → score retrieval quality
        → score faithfulness via LLM judge
        → score answer relevance via LLM judge
        → return EvaluationResult
"""

import hashlib
import json
import logging
import os
import string
import sys
from dataclasses import dataclass, field
import uuid

import duckdb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DB_PATH,
    EMBEDDING_MODEL,
    EVAL_WEIGHT_FAITHFULNESS,
    EVAL_WEIGHT_RELEVANCE,
    EVAL_WEIGHT_RETRIEVAL,
    LLM_JUDGE_MODEL,
    MIN_CONFIDENCE_SCORE,
)
from agents.reasoning import (
    ReasoningResult,
    call_groq,
    clamp,
    extract_json,
)

# ---------------------------------------------------------------------------
# Sanity checks — catch misconfiguration at startup, not at runtime
# ---------------------------------------------------------------------------

# Weights must sum to exactly 1.0 — enforced here so misconfiguration
# surfaces immediately on import, not silently during scoring
assert abs(
    EVAL_WEIGHT_RETRIEVAL +
    EVAL_WEIGHT_FAITHFULNESS +
    EVAL_WEIGHT_RELEVANCE - 1.0
) < 1e-9, (
    f"Evaluation weights must sum to 1.0, got "
    f"{EVAL_WEIGHT_RETRIEVAL + EVAL_WEIGHT_FAITHFULNESS + EVAL_WEIGHT_RELEVANCE:.4f}. "
    f"Check EVAL_WEIGHT_* in config.py."
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
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Complete evaluation scores for one question-answer pair.

    Contains three independent scores so we can diagnose
    exactly where the pipeline is failing.
    """
    question: str              # original question
    retrieval_score: float     # 0.0-1.0 — did we retrieve right chunks?
    faithfulness_score: float  # 0.0-1.0 — is answer faithful to chunks?
    answer_relevance: float    # 0.0-1.0 — does answer address question?
    overall_score: float       # weighted average of all three
    passed: bool               # overall_score >= MIN_CONFIDENCE_SCORE
    needs_human_review: bool   # flagged for human review
    failure_reason: str        # why it failed (if it did)
    ground_truth_found: bool   # was ground truth available in DuckDB?


# ---------------------------------------------------------------------------
# Ground truth lookup
# ---------------------------------------------------------------------------

def get_ground_truth(
    question: str,
    db_path: str = DB_PATH,
) -> list[dict]:
    """
    Look up ground truth answers using keyword matching.

    CUAD questions are long and specific — exact matching rarely works.
    We extract keywords from the question and search for ground truth
    answers whose questions contain those keywords.

    Uses parameterized queries throughout — never interpolates user
    input directly into SQL strings.

    Args:
        question: The user's question
        db_path:  Path to DuckDB database

    Returns:
        List of ground truth dicts
    """
    conn = duckdb.connect(db_path)

    # Extract meaningful keywords — skip common words
    stopwords = {"what", "is", "the", "a", "an", "are", "does",
                 "do", "in", "of", "for", "this", "that", "and"}

    keywords = [
        stripped
        for word in question.split()
        # Strip punctuation first, then apply all filters
        for stripped in [word.lower().strip(string.punctuation)]
        if stripped                        # not empty string after stripping
        and stripped not in stopwords      # not a stopword
        and len(stripped) > 2              # at least 3 characters
    ]

    if not keywords:
        conn.close()
        return []

    # Build parameterized query — one ? per keyword
    # Never interpolate user input directly into SQL
    placeholders = " OR ".join([
        "LOWER(question) LIKE ?"
        for _ in keywords
    ])

    # Wrap each keyword in % for LIKE matching
    params = [f"%{kw}%" for kw in keywords]

    rows = conn.execute(f"""
        SELECT
            answer,
            answer_start,
            contract_id
        FROM ground_truth
        WHERE {placeholders}
        LIMIT 10
    """, params).fetchall()

    conn.close()

    logger.info(
        "Ground truth search: keywords=%s, found=%d matches",
        keywords, len(rows),
    )

    return [
        {
            "answer": row[0],
            "answer_start": row[1],
            "contract_id": row[2],
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Retrieval scorer
# ---------------------------------------------------------------------------

def score_retrieval(
    reasoning_result: ReasoningResult,
    ground_truths: list[dict],
    embed_model=None,
) -> float:
    """
    Score retrieval quality using embedding cosine similarity.

    Compares each ground truth answer against all cited chunks
    using embedding similarity — much more accurate than keyword
    overlap which fails on paraphrasing and synonyms.

    A ground truth is considered found if any cited chunk has
    cosine similarity >= 0.5 with the ground truth answer.

    Args:
        reasoning_result: Output from Agent 2
        ground_truths:    Expert answers from DuckDB
        embed_model:      SentenceTransformer (loaded lazily if None)

    Returns:
        Float 0.0-1.0
    """
    if not ground_truths:
        logger.info("No ground truth — retrieval score defaulting to 0.5")
        return 0.5

    if not reasoning_result.sentences:
        return 0.0


    # Load embedding model lazily — reuse if already loaded
    if embed_model is None:
        # This should never happen in normal use — EvaluationAgent.__init__
        # always passes self.embed_model. If hit, it means the function was
        # called directly without an agent instance — warn and load fresh.
        logger.warning(
            "embed_model not provided — loading fresh (slow path). "
            "Pass embed_model=self.embed_model for production use."
        )
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Collect cited chunk texts
    cited_chunks = [s.chunk_text for s in reasoning_result.sentences]

    # Embed all cited chunks at once — efficient batch operation
    chunk_embeddings = embed_model.encode(
        cited_chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    hits = 0
    for gt in ground_truths:
        gt_answer = gt["answer"].strip()
        if not gt_answer or len(gt_answer) < 5:
            continue

        # Embed ground truth answer
        gt_embedding = embed_model.encode(
            [gt_answer],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        # Cosine similarity — vectors are normalized so dot product = cosine sim
        similarities = chunk_embeddings @ gt_embedding

        # Ground truth found if any chunk is similar enough
        max_similarity = float(similarities.max())
        if max_similarity >= 0.5:
            hits += 1

    score = hits / len(ground_truths)
    logger.info(
        "Retrieval score: %.2f (%d/%d ground truths found via embedding similarity)",
        score, hits, len(ground_truths),
    )
    return clamp(score, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Faithfulness scorer
# ---------------------------------------------------------------------------

def score_faithfulness(
    question: str,
    reasoning_result: ReasoningResult,
) -> float:
    """
    Score faithfulness — does the answer stay within what the chunks say?

    Uses an LLM judge to evaluate whether each sentence in the answer
    is supported by the source chunk it cites. This catches hallucination
    where the LLM adds information not in the source material.

    Args:
        question:         Original question
        reasoning_result: Output from Agent 2 with cited sentences

    Returns:
        Float 0.0-1.0
        1.0 = every sentence fully supported by its cited chunk
        0.0 = answer contains claims not in the source chunks
    """
    if not reasoning_result.sentences:
        return 0.0

    # Build faithfulness evaluation prompt
    sentences_text = ""
    for i, s in enumerate(reasoning_result.sentences, 1):
        sentences_text += f"""
Sentence {i}: "{s.text}"
Source chunk: "{s.chunk_text[:300]}..."
---"""

    prompt = f"""You are evaluating whether an AI answer is faithful to its source documents.

QUESTION: {question}

ANSWER SENTENCES AND THEIR SOURCE CHUNKS:
{sentences_text}

For each sentence, judge whether it is FULLY SUPPORTED by its source chunk.
A sentence is faithful if every claim in it can be directly verified from the chunk.
A sentence is NOT faithful if it adds information, makes assumptions, or contradicts the chunk.

RESPOND IN THIS EXACT JSON FORMAT — no preamble, no markdown:
{{
  "sentences": [
    {{
      "sentence_number": 1,
      "faithful": true,
      "reason": "Direct quote from chunk"
    }}
  ],
  "overall_faithfulness": 0.95
}}"""

    try:
        raw = call_groq(prompt, model=LLM_JUDGE_MODEL)
        cleaned = extract_json(raw)
        data = json.loads(cleaned)

        score = clamp(
            data.get("overall_faithfulness", 0.5), 0.0, 1.0
        )
        logger.info("Faithfulness score: %.2f", score)
        return score

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # Expected failures — LLM returned unparseable response
        logger.warning("Faithfulness scoring failed: %s", e)
        return 0.5
    except Exception as e:
        # Unexpected failure — log as error, not warning
        # This is likely a programming bug, not an LLM issue
        logger.error("Unexpected error in faithfulness scoring: %s", e)
        raise


# ---------------------------------------------------------------------------
# Answer relevance scorer
# ---------------------------------------------------------------------------

def score_answer_relevance(
    question: str,
    reasoning_result: ReasoningResult,
) -> float:
    """
    Score answer relevance — does the answer actually address the question?

    Sometimes the LLM returns an answer that is technically faithful to
    the chunks but doesn't answer what was asked. This catches that.

    Args:
        question:         Original question
        reasoning_result: Output from Agent 2

    Returns:
        Float 0.0-1.0
        1.0 = answer directly addresses the question
        0.0 = answer is off-topic or misses the question entirely
    """
    if not reasoning_result.answer_found or not reasoning_result.raw_answer:
        return 0.0

    prompt = f"""You are evaluating whether an AI answer addresses the question asked.

QUESTION: {question}

ANSWER: {reasoning_result.raw_answer}

Score how well the answer addresses the question on a scale of 0.0 to 1.0:
1.0 = directly and completely answers the question
0.5 = partially answers the question
0.0 = does not answer the question at all

RESPOND IN THIS EXACT JSON FORMAT — no preamble, no markdown:
{{
  "relevance_score": 0.9,
  "reason": "The answer directly addresses the termination conditions asked about"
}}"""

    try:
        raw = call_groq(prompt, model=LLM_JUDGE_MODEL)
        cleaned = extract_json(raw)
        data = json.loads(cleaned)

        score = clamp(
            data.get("relevance_score", 0.5), 0.0, 1.0
        )
        logger.info("Answer relevance score: %.2f", score)
        return score

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # Expected failures — LLM returned unparseable response
        logger.warning("Answer relevance scoring failed: %s", e)
        return 0.5
    except Exception as e:
        # Unexpected failure — re-raise so bugs surface immediately
        logger.error("Unexpected error in answer relevance scoring: %s", e)
        raise


# ---------------------------------------------------------------------------
# Evaluation Agent
# ---------------------------------------------------------------------------

class EvaluationAgent:
    """
    Agent 3 — scores the quality of answers from Agent 2.

    Measures three things independently:
        retrieval_score    — did Agent 1 find the right chunks?
        faithfulness_score — does the answer stay within the chunks?
        answer_relevance   — does the answer address the question?

    This separation is critical — a bad answer could be a retrieval
    failure OR a generation failure. This agent tells you which.

    Usage:
        agent = EvaluationAgent()
        result = agent.evaluate(reasoning_result)
    """

    def __init__(self, db_path: str = DB_PATH):
        logger.info("Initialising EvaluationAgent...")

        self.db_path = db_path

        # Load embedding model once — reused for retrieval scoring
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model for retrieval scoring...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

        # Create evaluations table if it doesn't exist
        # Stores every result so we can track quality over time
        self._setup_storage()

        logger.info("EvaluationAgent ready.")

    def _setup_storage(self) -> None:
        """
        Create the evaluations table in DuckDB.

        Stores every EvaluationResult so quality can be tracked
        over time. Week-over-week score changes reveal regressions
        or improvements in the pipeline.
        """
        conn = duckdb.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id                 VARCHAR PRIMARY KEY,
                question           VARCHAR,
                retrieval_score    FLOAT,
                faithfulness_score FLOAT,
                answer_relevance   FLOAT,
                overall_score      FLOAT,
                passed             BOOLEAN,
                needs_human_review BOOLEAN,
                failure_reason     VARCHAR,
                ground_truth_found BOOLEAN,
                prompt_version     VARCHAR,
                created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.close()
        logger.info("Evaluation storage ready.")

    def evaluate(
        self,
        reasoning_result: ReasoningResult,
    ) -> EvaluationResult:
        """
        Score a ReasoningResult across all three dimensions.

        Pipeline:
            1. Look up ground truth from DuckDB
            2. Score retrieval — did we find the right chunks?
            3. Score faithfulness — is answer faithful to chunks?
            4. Score relevance — does answer address the question?
            5. Compute overall score and pass/fail
            6. Store result in DuckDB for trend tracking

        Args:
            reasoning_result: Output from Agent 2

        Returns:
            EvaluationResult with all three scores and pass/fail
        """
        question = reasoning_result.question
        logger.info("Evaluating answer for: '%s'", question[:80])

        # Guard — nothing to evaluate
        if not reasoning_result.answer_found:
            logger.warning("No answer to evaluate — returning zero scores.")
            return EvaluationResult(
                question=question,
                retrieval_score=0.0,
                faithfulness_score=0.0,
                answer_relevance=0.0,
                overall_score=0.0,
                passed=False,
                needs_human_review=True,
                failure_reason="Agent 2 could not find an answer in the retrieved chunks.",
                ground_truth_found=False,
            )

        # Step 1 — look up ground truth from DuckDB
        ground_truths = get_ground_truth(question)
        ground_truth_found = len(ground_truths) > 0
        logger.info(
            "Ground truth: %d answers found for this question.",
            len(ground_truths),
        )

        # Step 2 — score retrieval quality
        retrieval_score = score_retrieval(
            reasoning_result,
            ground_truths,
            embed_model=self.embed_model,  # reuse loaded model
        )

        # Step 3 — score faithfulness (LLM judge)
        faithfulness_score = score_faithfulness(question, reasoning_result)

        # Step 4 — score answer relevance (LLM judge)
        answer_relevance = score_answer_relevance(question, reasoning_result)

        # Step 5 — compute weighted overall score
        # Weights live in config.py — tune without touching code
        overall_score = clamp(
            (retrieval_score    * EVAL_WEIGHT_RETRIEVAL) +
            (faithfulness_score * EVAL_WEIGHT_FAITHFULNESS) +
            (answer_relevance   * EVAL_WEIGHT_RELEVANCE),
            0.0, 1.0,
        )

        # Determine pass/fail and failure reason
        passed = overall_score >= MIN_CONFIDENCE_SCORE
        needs_human_review = not passed or reasoning_result.low_confidence

        failure_reason = ""
        if not passed:
            # Identify the weakest dimension to guide improvement
            scores = {
                "retrieval": retrieval_score,
                "faithfulness": faithfulness_score,
                "relevance": answer_relevance,
            }
            weakest = min(scores, key=scores.get)
            failure_reason = (
                f"Primary failure in {weakest} "
                f"(score={scores[weakest]:.2f}). "
                f"Overall={overall_score:.2f} below threshold={MIN_CONFIDENCE_SCORE}."
            )

        logger.info(
            "Evaluation complete — retrieval=%.2f, faithfulness=%.2f, "
            "relevance=%.2f, overall=%.2f, passed=%s",
            retrieval_score, faithfulness_score,
            answer_relevance, overall_score, passed,
        )

        eval_result = EvaluationResult(
            question=question,
            retrieval_score=retrieval_score,
            faithfulness_score=faithfulness_score,
            answer_relevance=answer_relevance,
            overall_score=overall_score,
            passed=passed,
            needs_human_review=needs_human_review,
            failure_reason=failure_reason,
            ground_truth_found=ground_truth_found,
        )

        # Store result — prompt version from reasoning result for traceability
        self._store_result(eval_result, prompt_version=reasoning_result.prompt_version)

        return eval_result

    def _store_result(self, result: EvaluationResult, prompt_version: str) -> None:
        """
        Persist an EvaluationResult to DuckDB.

        Every evaluation is stored with a timestamp so we can
        query score trends over time and detect regressions.

        Args:
            result:         The evaluation result to store
            prompt_version: Which prompt version produced the answer
        """

        # UUID — every evaluation gets a unique ID
        # Never overwrites — full history preserved for trend tracking
        eval_id = str(uuid.uuid4())

        conn = duckdb.connect(self.db_path)
        conn.execute("""
            INSERT INTO evaluations (
                id, question, retrieval_score, faithfulness_score,
                answer_relevance, overall_score, passed,
                needs_human_review, failure_reason,
                ground_truth_found, prompt_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            eval_id,
            result.question,
            result.retrieval_score,
            result.faithfulness_score,
            result.answer_relevance,
            result.overall_score,
            result.passed,
            result.needs_human_review,
            result.failure_reason,
            result.ground_truth_found,
            prompt_version,
        ])
        conn.close()
        logger.info("Evaluation result stored (id=%s).", eval_id)


# ---------------------------------------------------------------------------
# Entry point — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from agents.retrieval import RetrievalAgent
    from agents.reasoning import ReasoningAgent

    # Step 1 — retrieve
    retrieval = RetrievalAgent()
    question  = "What is the termination clause?"
    chunks    = retrieval.search(question)

    # Step 2 — reason
    reasoning = ReasoningAgent()
    result    = reasoning.reason(question, chunks)

    # Step 3 — evaluate
    evaluation = EvaluationAgent()
    eval_result = evaluation.evaluate(result)

    # Step 4 — print full pipeline output
    print(f"\n{'═' * 60}")
    print(f"QUESTION: {question}")
    print(f"{'═' * 60}")
    print(f"\nANSWER:")
    print(result.raw_answer)
    print(f"\n{'─' * 60}")
    print(f"EVALUATION SCORES:")
    print(f"{'─' * 60}")
    print(f"Retrieval score    : {eval_result.retrieval_score:.2f}")
    print(f"Faithfulness score : {eval_result.faithfulness_score:.2f}")
    print(f"Answer relevance   : {eval_result.answer_relevance:.2f}")
    print(f"Overall score      : {eval_result.overall_score:.2f}")
    print(f"Passed             : {eval_result.passed}")
    print(f"Needs human review : {eval_result.needs_human_review}")
    print(f"Ground truth found : {eval_result.ground_truth_found}")
    if eval_result.failure_reason:
        print(f"Failure reason     : {eval_result.failure_reason}")

    # Step 5 — verify storage
    print(f"\n{'─' * 60}")
    print("STORED EVALUATIONS:")
    print(f"{'─' * 60}")
    import duckdb as _duckdb
    conn = _duckdb.connect("data/contracts.db")
    print(conn.execute("""
        SELECT question[:50], overall_score, passed, created_at
        FROM evaluations
        ORDER BY created_at DESC
        LIMIT 5
    """).fetchdf())
    conn.close()