"""
agents/graph.py

LangGraph Pipeline — wires Agent 1, 2, and 3 into a stateful
graph with conditional routing.

The graph runs each agent as a node and routes based on evaluation
scores — passing answers through, flagging low-confidence ones for
human review, and retrying retrieval when scores are too low.

Flow:
    question
        → [retrieve]   Agent 1 — find relevant chunks
        → [reason]     Agent 2 — form cited answer
        → [evaluate]   Agent 3 — score quality
        → [route]      conditional — pass / review / retry
"""

import logging
import os
import sys
import time
from typing import TypedDict

from langgraph.graph import StateGraph, END

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HUMAN_REVIEW_FLAG, MIN_CONFIDENCE_SCORE, TOP_K_RETRIEVAL
from agents.retrieval import RetrievalAgent
from agents.reasoning import ReasoningAgent, ReasoningResult
from agents.evaluation import EvaluationAgent, EvaluationResult

from observability.logging import get_logger
from observability.tracing import setup_tracing

logger = get_logger(__name__)

# Enable LangSmith tracing if API key is configured
setup_tracing()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of retrieval retries before giving up
# Prevents infinite loops if retrieval consistently fails
MAX_RETRIES = 2

# Minimum retrieval score to accept without retrying
# Below this — try a different retrieval strategy
# CUAD ground truth answers are very specific clause-level annotations;
# retrieved chunks are broader passages, so similarity naturally lands in
# the 0.1-0.3 range even for good retrievals. Only retry when chunks are
# essentially absent (score near 0), not just below 0.3.
MIN_RETRIEVAL_SCORE = 0.05


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    """
    Shared state passed between every node in the graph.

    Each node reads what it needs and writes its output back.
    LangGraph merges updates automatically — nodes only need to
    return the keys they changed.
    """
    question: str                           # original user question
    chunks: list[dict]                      # retrieved chunks from Agent 1
    reasoning_result: ReasoningResult       # cited answer from Agent 2
    eval_result: EvaluationResult           # scores from Agent 3
    final_answer: str                       # plain text answer for the user
    needs_human_review: bool                # flagged for human review
    retry_count: int                        # how many times retrieval retried
    error: str                              # error message if pipeline fails
    filters: dict | None

# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def retrieve_node(state: PipelineState, retrieval_agent: RetrievalAgent) -> dict:
    """
    Node 1 — retrieve relevant chunks using Agent 1.

    On first attempt uses standard TOP_K_RETRIEVAL.
    On retry, expands top_k to search wider — giving the reasoning
    agent more context to work with when initial retrieval scores low.
    Applies contract filters before reranking if provided.

    Args:
        state:           Current pipeline state
        retrieval_agent: Shared RetrievalAgent instance

    Returns:
        Updated state keys: chunks, retry_count
    """
    question    = state["question"]
    retry_count = state.get("retry_count", 0)
    filters     = state.get("filters", None)  # ← get filters from state

    # Expand the candidate pool on each retry — more context for the LLM
    expanded_top_k = TOP_K_RETRIEVAL * (1 + retry_count)

    logger.info(
        "Node: retrieve (attempt %d/%d, top_k=%d) — '%s'",
        retry_count + 1, MAX_RETRIES + 1,
        expanded_top_k,
        question[:60],
    )

    t0 = time.monotonic()
    reranked = retrieval_agent.search(
        question,
        filters=filters,
        top_k=expanded_top_k,
    )
    elapsed_ms = (time.monotonic() - t0) * 1000

    logger.info(
        "Node: retrieve done in %.0fms — %d chunks returned.",
        elapsed_ms, len(reranked),
    )

    return {
        "chunks": reranked,
        "retry_count": retry_count + 1,
    }

def reason_node(state: PipelineState, reasoning_agent: ReasoningAgent) -> dict:
    """
    Node 2 — form a cited answer using Agent 2.

    Sends retrieved chunks to the NVIDIA LLM with a structured prompt.
    Returns answer with every sentence linked to its source chunk.

    Args:
        state:           Current pipeline state
        reasoning_agent: Shared ReasoningAgent instance

    Returns:
        Updated state keys: reasoning_result, final_answer
    """
    question = state["question"]
    chunks   = state["chunks"]

    logger.info("Node: reason — forming answer from %d chunks.", len(chunks))

    t0 = time.monotonic()
    reasoning_result = reasoning_agent.reason(question, chunks)
    elapsed_ms = (time.monotonic() - t0) * 1000

    logger.info("Node: reason done in %.0fms — answer_found=%s.", elapsed_ms, reasoning_result.answer_found)

    return {
        "reasoning_result": reasoning_result,
        "final_answer": reasoning_result.raw_answer,
    }


def evaluate_node(state: PipelineState, evaluation_agent: EvaluationAgent) -> dict:
    """
    Node 3 — score answer quality using Agent 3.

    Measures retrieval, faithfulness, and relevance independently.
    Stores result in DuckDB for trend tracking.

    Catches all exceptions so the graph always completes — an eval
    failure returns a safe fallback result that flags for human review
    rather than crashing the pipeline and leaving final_state unset.

    Args:
        state:            Current pipeline state
        evaluation_agent: Shared EvaluationAgent instance

    Returns:
        Updated state keys: eval_result, needs_human_review
    """
    reasoning_result = state["reasoning_result"]

    logger.info("Node: evaluate — scoring answer quality.")

    t0 = time.monotonic()
    try:
        eval_result = evaluation_agent.evaluate(reasoning_result)
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info("Node: evaluate done in %.0fms — overall=%.2f passed=%s.", elapsed_ms, eval_result.overall_score, eval_result.passed)

    except Exception as e:
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.error(
            "Node: evaluate failed after %.0fms: %s. "
            "Returning fallback result and flagging for human review.",
            elapsed_ms, e,
        )

        eval_result = EvaluationResult(
            question=reasoning_result.question,
            retrieval_score=0.0,
            faithfulness_score=0.0,
            answer_relevance=0.0,
            overall_score=0.0,
            passed=False,
            needs_human_review=True,
            failure_reason=f"Evaluation failed: {type(e).__name__}: {e}",
            ground_truth_found=False,
            retrieval_score_available=False,
        )

    return {
        "eval_result": eval_result,
        "needs_human_review": eval_result.needs_human_review,
    }


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def route_after_evaluation(state: PipelineState) -> str:
    """
    Conditional edge — decide what happens after evaluation.

    Three possible routes:
        "end"           → answer passed, return to user
        "human_review"  → low confidence, flag for human
        "retry"         → retrieval score too low, try again

    This is the key production insight — we can diagnose failures
    and respond differently based on which score is lowest.

    Args:
        state: Current pipeline state

    Returns:
        String key for next node
    """
    eval_result  = state["eval_result"]
    retry_count  = state.get("retry_count", 0)

    # If answer passed evaluation — we're done
    if eval_result.passed:
        logger.info("Route: PASS → returning answer to user.")
        return "end"

    # Don't retry if evaluation itself failed — scores are meaningless zeros
    # not genuine retrieval failures
    eval_failed = (
        eval_result.retrieval_score == 0.0
        and eval_result.faithfulness_score == 0.0
        and eval_result.answer_relevance == 0.0
        and "Evaluation failed" in (eval_result.failure_reason or "")
    )

    if (
        not eval_failed
        and eval_result.retrieval_score < MIN_RETRIEVAL_SCORE
        # retry_count is incremented inside retrieve_node before routing
        # so we use <= to get MAX_RETRIES actual retries (not MAX_RETRIES - 1)
        and retry_count <= MAX_RETRIES
    ):
        logger.info(
            "Route: RETRY → retrieval score %.2f below %.2f (attempt %d/%d).",
            eval_result.retrieval_score, MIN_RETRIEVAL_SCORE,
            retry_count, MAX_RETRIES + 1,
        )
        return "retry"

    # Otherwise — flag for human review
    logger.info(
        "Route: HUMAN REVIEW → overall=%.2f, retrieval=%.2f, "
        "faithfulness=%.2f, relevance=%.2f.",
        eval_result.overall_score,
        eval_result.retrieval_score,
        eval_result.faithfulness_score,
        eval_result.answer_relevance,
    )
    return "human_review"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG pipeline using LangGraph.

    Wires Agent 1 (Retrieval), Agent 2 (Reasoning), and Agent 3
    (Evaluation) into a stateful graph with conditional routing.

    The graph handles retries, human review flagging, and error
    recovery — production concerns that a simple function chain
    cannot express.

    Usage:
        pipeline = RAGPipeline()
        result = pipeline.run("What is the termination clause?")
    """

    def __init__(self):
        logger.info("Initialising RAGPipeline...")

        # Initialise all three agents — shared across graph runs
        self.retrieval_agent  = RetrievalAgent()
        self.reasoning_agent  = ReasoningAgent()
        self.evaluation_agent = EvaluationAgent()

        # Build and compile the LangGraph graph
        self.graph = self._build_graph()

        logger.info("RAGPipeline ready.")

    def _build_graph(self):
        """
        Build the LangGraph state graph.

        Nodes:
            retrieve     → Agent 1
            reason       → Agent 2
            evaluate     → Agent 3
            human_review → flag and return

        Edges:
            retrieve → reason → evaluate → [conditional]
            conditional → end | human_review | retrieve (retry)

        Returns:
            Compiled LangGraph graph
        """
        # Create the graph with our state schema
        graph = StateGraph(PipelineState)

        # Add nodes — wrap agent calls in lambdas to inject agent instances
        graph.add_node(
            "retrieve",
            lambda state: retrieve_node(state, self.retrieval_agent)
        )
        graph.add_node(
            "reason",
            lambda state: reason_node(state, self.reasoning_agent)
        )
        graph.add_node(
            "evaluate",
            lambda state: evaluate_node(state, self.evaluation_agent)
        )
        graph.add_node(
            "human_review",
            lambda state: {
                "needs_human_review": True,
                "final_answer": (
                    # Guard — reasoning_result may be None if pipeline failed
                    # before Agent 2 completed. Fallback to empty string.
                    (getattr(state.get("reasoning_result"), "raw_answer", "") or "") +
                    f"\n\n{HUMAN_REVIEW_FLAG}"
                ),
            }
        )

        # Add edges — linear flow
        graph.add_edge("retrieve", "reason")
        graph.add_edge("reason", "evaluate")

        # Conditional edge after evaluation
        graph.add_conditional_edges(
            "evaluate",
            route_after_evaluation,
            {
                "end":          END,
                "human_review": "human_review",
                "retry":        "retrieve",   # loop back to retrieval
            }
        )

        # Human review always ends after flagging
        graph.add_edge("human_review", END)

        # Entry point
        graph.set_entry_point("retrieve")

        return graph.compile()

    def run(
        self,
        question: str,
        filters: dict = None,
    ) -> dict:
        """
        Run the full RAG pipeline for a question.

        Args:
            question: The user's natural language question
            filters:  Optional metadata filters for retrieval

        Returns:
            Dict with:
                answer            — plain text answer
                sentences         — cited sentences with sources
                eval_scores       — retrieval, faithfulness, relevance
                needs_human_review — flagged for review
                passed            — overall pass/fail
        """
        logger.info("Pipeline run: '%s'", question[:80])

        # Initial state
        initial_state: PipelineState = {
            "question": question,
            "chunks": [],
            "reasoning_result": None,
            "eval_result": None,
            "final_answer": "",
            "needs_human_review": False,
            "retry_count": 0,
            "error": "",
            "filters": filters,
        }

        # Run the graph — timed for tracing
        from observability.tracing import Timer, trace_pipeline_run

        with Timer() as t:
            final_state = self.graph.invoke(initial_state)


        # Package result for the API/UI layer
        # Guard — reasoning or eval may be None if pipeline failed mid-run
        reasoning_result = final_state.get("reasoning_result")
        eval_result      = final_state.get("eval_result")

        if reasoning_result is None or eval_result is None:
            logger.error("Pipeline failed mid-run — reasoning or eval result is None.")
            return {
                "question":           question,
                "answer":             "Pipeline failed — please try again.",
                "sentences":          [],
                "eval_scores":        {
                    "retrieval": 0.0, "faithfulness": 0.0,
                    "relevance": 0.0, "overall": 0.0,
                },
                "needs_human_review": True,
                "passed":             False,
                "chunks_used":        [],
            }

        # Package result for the API/UI layer
        result = {
            "question":           question,
            "answer":             final_state["final_answer"],
            "sentences":          [
                {
                    "text":           s.text,
                    "chunk_id":       s.chunk_id,
                    "contract_title": s.contract_title,
                    "chunk_text":     s.chunk_text,
                    "confidence":     s.confidence,
                }
                for s in reasoning_result.sentences
            ],
            "eval_scores": {
                "retrieval":                 eval_result.retrieval_score,
                "faithfulness":              eval_result.faithfulness_score,
                "relevance":                 eval_result.answer_relevance,
                "overall":                   eval_result.overall_score,
                "retrieval_score_available": eval_result.retrieval_score_available,
            },
            "needs_human_review": final_state["needs_human_review"],
            "passed":             eval_result.passed,
            "chunks_used":        reasoning_result.chunks_used,
        }

        # Trace the completed run
        trace_pipeline_run(question, result, t.elapsed_ms)

        return result


# ---------------------------------------------------------------------------
# Entry point — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Build and run the full pipeline
    pipeline = RAGPipeline()

    question = "What is the termination clause?"
    result   = pipeline.run(question)

    print(f"\n{'═' * 60}")
    print(f"QUESTION: {question}")
    print(f"{'═' * 60}")
    print(f"\nANSWER:")
    print(result["answer"])
    print(f"\n{'─' * 60}")
    print("EVIDENCE CHAIN:")
    print(f"{'─' * 60}")
    for i, s in enumerate(result["sentences"], 1):
        print(f"\nSentence {i}: {s['text']}")
        print(f"  ← Source    : {s['contract_title'][:60]}")
        print(f"  ← Confidence: {s['confidence']:.2f}")

    print(f"\n{'─' * 60}")
    print("EVALUATION:")
    print(f"{'─' * 60}")
    print(f"Retrieval    : {result['eval_scores']['retrieval']:.2f}")
    print(f"Faithfulness : {result['eval_scores']['faithfulness']:.2f}")
    print(f"Relevance    : {result['eval_scores']['relevance']:.2f}")
    print(f"Overall      : {result['eval_scores']['overall']:.2f}")
    print(f"Passed       : {result['passed']}")
    print(f"Human review : {result['needs_human_review']}")