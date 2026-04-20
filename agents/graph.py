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
from typing import TypedDict

from langgraph.graph import StateGraph, END

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MIN_CONFIDENCE_SCORE, TOP_K_RETRIEVAL
from agents.retrieval import RetrievalAgent
from agents.reasoning import ReasoningAgent, ReasoningResult
from agents.evaluation import EvaluationAgent, EvaluationResult

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
# Constants
# ---------------------------------------------------------------------------

# Maximum number of retrieval retries before giving up
# Prevents infinite loops if retrieval consistently fails
MAX_RETRIES = 2

# Minimum retrieval score to accept without retrying
# Below this — try a different retrieval strategy
MIN_RETRIEVAL_SCORE = 0.3


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
    """..."""
    question    = state["question"]
    retry_count = state.get("retry_count", 0)
    filters     = state.get("filters", None)  # ← get filters from state

    expanded_top_k = TOP_K_RETRIEVAL * (1 + retry_count)

    logger.info(
        "Node: retrieve (attempt %d/%d, top_k=%d) — '%s'",
        retry_count + 1, MAX_RETRIES + 1,
        expanded_top_k,
        question[:60],
    )

    chunks         = retrieval_agent._semantic_search(question, top_k=expanded_top_k)
    keyword_chunks = retrieval_agent._keyword_search(question, top_k=expanded_top_k)
    merged         = retrieval_agent._merge_results(chunks, keyword_chunks)

    # Apply filters before reranking — not after
    if filters:
        merged = retrieval_agent._apply_filters(merged, filters)

    reranked = retrieval_agent._rerank(question, merged)

    logger.info(
        "Retrieved %d chunks (expanded top_k=%d on retry %d).",
        len(reranked), expanded_top_k, retry_count,
    )

    return {
        "chunks": reranked,
        "retry_count": retry_count + 1,
    }

def reason_node(state: PipelineState, reasoning_agent: ReasoningAgent) -> dict:
    """
    Node 2 — form a cited answer using Agent 2.

    Sends retrieved chunks to Groq LLM with structured prompt.
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

    reasoning_result = reasoning_agent.reason(question, chunks)

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

    try:
        eval_result = evaluation_agent.evaluate(reasoning_result)

    except Exception as e:
        # Evaluation failed — log the error but don't crash the graph
        # Return a safe fallback that routes to human review
        # The None-guard in run() handles missing eval_result,
        # but a fallback here is cleaner — graph still completes normally
        logger.error(
            "Evaluation node failed unexpectedly: %s. "
            "Returning fallback result and flagging for human review.", e,
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

    # retry_count is incremented inside retrieve_node before routing
    # so we use <= to get MAX_RETRIES actual retries (not MAX_RETRIES - 1)
    if eval_result.retrieval_score < MIN_RETRIEVAL_SCORE and retry_count <= MAX_RETRIES:
        logger.info(
            "Route: RETRY → retrieval score %.2f below %.2f (attempt %d/%d).",
            eval_result.retrieval_score, MIN_RETRIEVAL_SCORE,
            retry_count + 1, MAX_RETRIES,
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

    def _build_graph(self) -> any:
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
                    "\n\n[⚠️ This answer has been flagged for human review "
                    "due to low confidence scores.]"
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

        # Run the graph
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
        return {
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
                "retrieval":    eval_result.retrieval_score,
                "faithfulness": eval_result.faithfulness_score,
                "relevance":    eval_result.answer_relevance,
                "overall":      eval_result.overall_score,
            },
            "needs_human_review": final_state["needs_human_review"],
            "passed":             eval_result.passed,
            "chunks_used":        reasoning_result.chunks_used,
        }


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