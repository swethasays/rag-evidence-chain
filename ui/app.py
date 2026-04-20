"""
ui/app.py

Streamlit web interface for the RAG Evidence Chain system.

Features:
    - Contract selection dropdown (filters retrieval to one contract)
    - Question input with search button
    - Answer display with evidence chain
    - Plotly DAG visualization of evidence
    - Evaluation score gauges
    - Human review flag display
    - Feedback buttons (thumbs up/down)

Usage:
    streamlit run ui/app.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import duckdb

from config import DB_PATH
from agents.graph import RAGPipeline

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Evidence Chain",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark legal aesthetic
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&display=swap');

    /* Base */
    .stApp {
        background: #0d0f12;
        color: #e8e4dc;
    }

    /* Typography */
    h1, h2, h3 {
        font-family: 'DM Serif Display', serif !important;
        color: #e8e4dc !important;
    }

    /* Input fields */
    .stTextInput input, .stSelectbox select {
        background: #161920 !important;
        border: 1px solid #2a2d35 !important;
        color: #e8e4dc !important;
        border-radius: 4px !important;
        font-family: 'DM Mono', monospace !important;
    }

    /* Buttons */
    .stButton button {
        background: #c9a84c !important;
        color: #0d0f12 !important;
        border: none !important;
        font-family: 'DM Mono', monospace !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em !important;
        border-radius: 3px !important;
    }

    .stButton button:hover {
        background: #e8c46a !important;
    }

    /* Sentence cards */
    .sentence-card {
        background: #161920;
        border-left: 3px solid #c9a84c;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 4px 4px 0;
        font-family: 'DM Mono', monospace;
        font-size: 0.85rem;
    }

    .sentence-text {
        color: #e8e4dc;
        margin-bottom: 6px;
        line-height: 1.6;
    }

    .sentence-meta {
        color: #6b7280;
        font-size: 0.75rem;
    }

    .confidence-high { color: #4ade80; }
    .confidence-mid  { color: #facc15; }
    .confidence-low  { color: #f87171; }

    /* Score bars */
    .score-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 6px 0;
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
    }

    .score-label {
        width: 100px;
        color: #9ca3af;
        text-align: right;
    }

    .score-bar-bg {
        flex: 1;
        height: 6px;
        background: #2a2d35;
        border-radius: 3px;
        overflow: hidden;
    }

    .score-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.8s ease;
    }

    .score-value {
        width: 36px;
        color: #e8e4dc;
        text-align: right;
    }

    /* Review badge */
    .review-badge {
        background: #451a03;
        border: 1px solid #92400e;
        color: #fbbf24;
        padding: 8px 14px;
        border-radius: 4px;
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
        display: inline-block;
        margin: 8px 0;
    }

    .pass-badge {
        background: #052e16;
        border: 1px solid #166534;
        color: #4ade80;
        padding: 8px 14px;
        border-radius: 4px;
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
        display: inline-block;
        margin: 8px 0;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #2a2d35 !important;
        margin: 24px 0 !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load pipeline — cached so it only initialises once
# ---------------------------------------------------------------------------

@st.cache_resource
def load_pipeline():
    """
    Load the RAG pipeline once and cache it across sessions.

    @st.cache_resource means this runs once when the app starts
    and the same pipeline instance is reused for every user request.
    Avoids reloading all models on every page refresh.
    """
    return RAGPipeline()


@st.cache_data
def load_contracts():
    """
    Load contract titles from DuckDB for the dropdown.

    @st.cache_data caches the result so we don't query DuckDB
    on every user interaction — only when the app first loads.
    """
    with duckdb.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT title
            FROM contracts
            ORDER BY title
        """).fetchall()
    return [row[0] for row in rows]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("## 🔗 RAG Evidence Chain")
st.markdown(
    "<p style='color:#6b7280; font-family:DM Mono,monospace; font-size:0.85rem;'>"
    "Legal contract Q&A — every answer traced to its source</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar — about
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    **RAG Evidence Chain** answers questions about legal contracts
    and shows exactly which clause each sentence came from.

    **Stack**
    - LangGraph agents
    - Groq LLM
    - FAISS + BM25 hybrid search
    - Cross-encoder re-ranking
    - DuckDB + Redis

    **Eval metrics**
    - Retrieval quality
    - Faithfulness
    - Answer relevance
    """)

# ---------------------------------------------------------------------------
# Search form
# ---------------------------------------------------------------------------

contracts = load_contracts()
contract_options = ["All contracts"] + contracts

col1, col2 = st.columns([1, 2])

with col1:
    selected_contract = st.selectbox(
        "Filter by contract (optional)",
        contract_options,
        help="Scope the search to a single contract for higher precision",
    )

with col2:
    question = st.text_input(
        "Your question",
        placeholder="What is the termination clause?",
    )

search_clicked = st.button("🔍 Search", use_container_width=False)

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

if search_clicked and question.strip():

    # Build filter if contract selected
    filters = None
    if selected_contract != "All contracts":
        filters = {"contract_title": selected_contract}

    with st.spinner("Searching contracts..."):
        pipeline = load_pipeline()
        result   = pipeline.run(question.strip(), filters=filters)

    st.markdown("---")

    # ── Answer ──────────────────────────────────────────────────────────────
    st.markdown("### Answer")

    # Show pass/fail badge
    if result["passed"]:
        st.markdown(
            '<span class="pass-badge">✓ Passed evaluation</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="review-badge">⚠ Flagged for human review</span>',
            unsafe_allow_html=True,
        )

    # Show plain text answer
    answer_text = result["answer"].replace(
        "\n\n[⚠️ This answer has been flagged for human review due to low confidence scores.]",
        ""
    )
    st.markdown(
        f"<p style='font-size:1rem; line-height:1.8; color:#e8e4dc;'>{answer_text}</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Evidence chain ───────────────────────────────────────────────────────
    col_evidence, col_scores = st.columns([3, 2])

    with col_evidence:
        st.markdown("### Evidence Chain")
        st.markdown(
            "<p style='color:#6b7280; font-size:0.78rem; font-family:DM Mono,monospace;'>"
            "Every sentence traced to its source chunk</p>",
            unsafe_allow_html=True,
        )

        for i, s in enumerate(result["sentences"], 1):
            conf = s["confidence"]
            if conf >= 0.8:
                conf_class = "confidence-high"
                conf_label = "high"
            elif conf >= 0.6:
                conf_class = "confidence-mid"
                conf_label = "mid"
            else:
                conf_class = "confidence-low"
                conf_label = "low"

            st.markdown(f"""
            <div class="sentence-card">
                <div class="sentence-text">{i}. {s['text']}</div>
                <div class="sentence-meta">
                    ← <b>{s['contract_title'][:55]}</b>
                    &nbsp;|&nbsp;
                    confidence: <span class="{conf_class}">{conf:.2f} ({conf_label})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Expandable source chunk
            with st.expander(f"View source chunk {i}"):
                st.markdown(
                    f"<p style='font-family:DM Mono,monospace; font-size:0.8rem; "
                    f"color:#9ca3af; line-height:1.6;'>{s['chunk_text'][:600]}...</p>",
                    unsafe_allow_html=True,
                )

    # ── Evaluation scores ────────────────────────────────────────────────────
    with col_scores:
        st.markdown("### Evaluation Scores")

        scores = result["eval_scores"]

        def score_bar(label: str, value: float) -> str:
            """Render a score row with animated bar."""
            pct   = int(value * 100)
            color = "#4ade80" if value >= 0.7 else "#facc15" if value >= 0.5 else "#f87171"
            return f"""
            <div class="score-row">
                <span class="score-label">{label}</span>
                <div class="score-bar-bg">
                    <div class="score-bar-fill"
                         style="width:{pct}%; background:{color};"></div>
                </div>
                <span class="score-value">{value:.2f}</span>
            </div>"""

        st.markdown(
            score_bar("Retrieval",    scores["retrieval"]) +
            score_bar("Faithfulness", scores["faithfulness"]) +
            score_bar("Relevance",    scores["relevance"]) +
            "<hr style='margin:12px 0;'/>" +
            score_bar("Overall",      scores["overall"]),
            unsafe_allow_html=True,
        )

        # Threshold indicator
        st.markdown(
            f"<p style='color:#6b7280; font-size:0.72rem; font-family:DM Mono,monospace;"
            f" margin-top:8px;'>Pass threshold: 0.70</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── DAG visualization ────────────────────────────────────────────────────
    st.markdown("### Evidence Graph")
    st.markdown(
        "<p style='color:#6b7280; font-size:0.78rem; font-family:DM Mono,monospace;'>"
        "Question → sentences → source chunks</p>",
        unsafe_allow_html=True,
    )

    sentences = result["sentences"]
    n = len(sentences)

    # Node positions
    # Question at top, sentences in middle, chunks at bottom
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    edge_x, edge_y = [], []

    # Question node
    node_x.append(0.5)
    node_y.append(1.0)
    node_text.append(f"Q: {question[:40]}...")
    node_color.append("#c9a84c")
    node_size.append(20)

    # Sentence nodes + edges from question
    for i in range(n):
        sx = (i + 1) / (n + 1)
        node_x.append(sx)
        node_y.append(0.5)
        node_text.append(f"S{i+1}: {sentences[i]['text'][:40]}...")
        node_color.append("#4a6fa5")
        node_size.append(16)

        # Edge question → sentence
        edge_x += [0.5, sx, None]
        edge_y += [1.0, 0.5, None]

    # Chunk nodes + edges from sentences
    seen_chunks = {}
    for i, s in enumerate(sentences):
        cid = s["chunk_id"]
        if cid not in seen_chunks:
            cx = (i + 1) / (n + 1)
            seen_chunks[cid] = (cx, 0.0)
            node_x.append(cx)
            node_y.append(0.0)
            node_text.append(f"{s['contract_title'][:30]}...")
            node_color.append("#2d6a4f")
            node_size.append(14)

        sx = (i + 1) / (n + 1)
        cx, cy = seen_chunks[cid]

        # Edge sentence → chunk
        edge_x += [sx, cx, None]
        edge_y += [0.5, 0.0, None]

    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#2a2d35", width=2),
        hoverinfo="none",
    ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(color="#0d0f12", width=2),
        ),
        text=node_text,
        textposition="bottom center",
        textfont=dict(
            family="DM Mono, monospace",
            size=9,
            color="#9ca3af",
        ),
        hoverinfo="text",
    ))

    fig.update_layout(
        showlegend=False,
        paper_bgcolor="#0d0f12",
        plot_bgcolor="#0d0f12",
        margin=dict(l=20, r=20, t=20, b=60),
        height=320,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Feedback ─────────────────────────────────────────────────────────────
    st.markdown("### Was this answer helpful?")
    fb_col1, fb_col2, _ = st.columns([1, 1, 6])

    with fb_col1:
        if st.button("👍 Yes"):
            st.success("Thank you for your feedback!")

    with fb_col2:
        if st.button("👎 No"):
            st.error("Thank you — we'll use this to improve.")

elif search_clicked and not question.strip():
    st.warning("Please enter a question.")