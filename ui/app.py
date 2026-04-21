"""
ui/app.py

Streamlit web interface for the RAG Evidence Chain system.

Usage:
    streamlit run ui/app.py
"""

import hashlib
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import duckdb

from config import DB_PATH
from agents.graph import RAGPipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Evidence Chain",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;0,600;1,300&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap');

    :root {
        --bg:        #0a0b0d;
        --surface:   #14171f;
        --surface2:  #1c2030;
        --border:    #252b38;
        --border2:   #313a4d;
        --gold:      #d4a853;
        --gold-dim:  #a07838;
        --gold-text: #e8c47a;
        --text:      #f0ebe2;
        --text-dim:  #a0a8b8;
        --text-faint:#5a6070;
        --green:     #4caf7d;
        --yellow:    #e8b84b;
        --red:       #e06060;
        --blue:      #5b8dee;
    }

    /* ── Reset ── */
    .stApp { background: var(--bg) !important; }
    .block-container { padding: 2.5rem 3rem 4rem !important; max-width: 1100px !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Typography ── */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif !important;
        color: var(--text) !important;
    }
    h1, h2, h3, h4 {
        font-family: 'Fraunces', serif !important;
        font-weight: 300 !important;
        letter-spacing: -0.02em !important;
    }

    /* ── App header ── */
    .app-title {
        font-family: 'Fraunces', serif;
        font-size: 2rem;
        font-weight: 300;
        color: var(--text);
        letter-spacing: -0.03em;
        line-height: 1;
        margin-bottom: 6px;
    }
    .app-title span { color: var(--gold); font-style: italic; }
    .app-subtitle {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        color: var(--text-faint);
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 28px;
        margin-top: 4px;
    }

    /* ── Divider ── */
    .divider { height: 1px; background: var(--border); margin: 24px 0; }

    /* ── Search labels ── */
    .search-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.66rem;
        color: var(--text-dim);
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* ── Input ── */
    .stTextInput > div > div > input {
        background: var(--surface) !important;
        border: 1px solid var(--border2) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 13px 18px !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
        height: 48px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 3px rgba(212,168,83,0.12) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: var(--text-faint) !important;
        font-style: italic !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background: var(--surface) !important;
        border: 1px solid var(--border2) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        min-height: 48px !important;
        height: 48px !important;
    }
    .stSelectbox > div > div > div {
        padding-top: 10px !important;
        padding-bottom: 10px !important;
        line-height: 1.4 !important;
    }
            
    /* ── Selectbox dropdown ── */
    [data-baseweb="select"] input {
        color: var(--text) !important;
        caret-color: var(--gold) !important;
    }

    /* Dropdown menu background */
    [data-baseweb="popover"] {
        background: var(--surface) !important;
    }

    /* Dropdown options */
    [data-baseweb="menu"] ul li {
        background: var(--surface) !important;
        color: var(--text-dim) !important;
    }

    [data-baseweb="menu"] ul li:hover {
        background: var(--surface2) !important;
        color: var(--text) !important;
    }

    /* "Press Enter to apply" hint (Streamlit selectbox helper text) */
    .stSelectbox + div > div {
        color: #8b93a7 !important;  /* medium muted color */
        font-size: 0.72rem !important;
        opacity: 1 !important;
    }
            

    /* ── Form — remove default chrome ── */
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }

    /* ── Submit button (→ arrow) ── */
    div[data-testid="stFormSubmitButton"] > button {
        background: var(--gold) !important;
        color: #0a0b0d !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        height: 48px !important;
        min-height: 48px !important;
        max-width: 52px !important;
        padding: 0 !important;
        width: 100% !important;
        transition: background 0.15s, transform 0.1s !important;
    }
            
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #e8c06a !important;
        transform: translateY(-1px) !important;
    }

    /* ── All other buttons ── */
    div[data-testid="stButton"] > button {
        background: var(--surface2) !important;
        color: var(--text-dim) !important;
        border: 1px solid var(--border2) !important;
        border-radius: 6px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
        font-weight: 400 !important;
        transition: background 0.15s !important;
        width: 100% !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: var(--border2) !important;
        color: var(--text) !important;
        transform: none !important;
    }

    /* ── Feedback buttons ── */
    .fb-wrap div[data-testid="stButton"] > button {
        padding: 8px 20px !important;
        font-size: 0.82rem !important;
        letter-spacing: 0 !important;
    }

    /* ── Badges ── */
    .badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 12px; border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem; font-weight: 500;
        letter-spacing: 0.04em; margin-bottom: 14px;
    }
    .badge-pass   { background: rgba(76,175,125,0.12); border: 1px solid rgba(76,175,125,0.35); color: var(--green); }
    .badge-review { background: rgba(232,184,75,0.1);  border: 1px solid rgba(232,184,75,0.3);  color: var(--yellow); }

    /* ── Answer ── */
    .answer-text {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.97rem; line-height: 1.9;
        color: var(--text); padding: 16px 0 4px;
    }

    /* ── Section label ── */
    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.63rem; color: var(--text-faint);
        letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 10px;
    }

    /* ── Evidence card ── */
    .ev-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 3px solid var(--gold-dim);
        border-radius: 0 8px 8px 0;
        padding: 14px 18px; margin-bottom: 10px;
        transition: border-left-color 0.2s, background 0.2s;
    }
    .ev-card:hover { border-left-color: var(--gold); background: var(--surface2); }
    .ev-num    { font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; color: var(--gold-dim); margin-bottom: 6px; letter-spacing: 0.1em; }
    .ev-text   { font-size: 0.9rem; line-height: 1.65; color: var(--text); margin-bottom: 10px; }
    .ev-source { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--text-faint); display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .ev-conf-high { color: var(--green); }
    .ev-conf-mid  { color: var(--yellow); }
    .ev-conf-low  { color: var(--red); }

    /* ── Score card ── */
    .score-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 10px; padding: 22px;
    }
    .score-card-title { font-family: 'Fraunces', serif; font-size: 1rem; font-weight: 300; color: var(--text); margin-bottom: 20px; }
    .score-row-item   { margin-bottom: 16px; }
    .score-row-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
    .score-name { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--text-dim); display: flex; align-items: center; gap: 6px; }
    .score-val  { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight: 500; }
    .score-track { height: 5px; background: var(--border2); border-radius: 3px; overflow: hidden; }
    .score-fill  { height: 100%; border-radius: 3px; }
    .score-divider   { height: 1px; background: var(--border); margin: 16px 0; }
    .score-threshold { font-family: 'IBM Plex Mono', monospace; font-size: 0.63rem; color: var(--text-faint); margin-top: 12px; }

    /* ── Tooltip ── */
    .tip-wrap { position: relative; display: inline-flex; align-items: center; cursor: help; }
    .tip-icon {
        font-size: 0.58rem; color: var(--text-faint);
        border: 1px solid var(--border2); border-radius: 50%;
        width: 13px; height: 13px;
        display: inline-flex; align-items: center; justify-content: center;
        font-family: 'IBM Plex Mono', monospace;
    }
    .tip-box {
        visibility: hidden; opacity: 0;
        background: var(--surface2); border: 1px solid var(--border2);
        color: var(--text-dim); font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.72rem; line-height: 1.5; border-radius: 6px;
        padding: 8px 12px; width: 210px;
        position: absolute; left: 20px; top: -4px; z-index: 100;
        transition: opacity 0.15s; pointer-events: none;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }
    .tip-wrap:hover .tip-box { visibility: visible; opacity: 1; }
            
    /* Prevent tooltip overflow on narrow screens */
    .score-name .tip-box {
        left: auto !important;
        right: 0 !important;
        top: 20px !important;
    }

    /* ── Empty state ── */
    .empty-state {
        background: var(--surface); border: 1px dashed var(--border2);
        border-radius: 10px; padding: 44px 32px; text-align: center; margin: 12px 0;
    }
    .empty-icon  { font-size: 1.8rem; margin-bottom: 14px; }
    .empty-title { font-family: 'Fraunces', serif; font-size: 1.05rem; font-weight: 300; color: var(--text); margin-bottom: 8px; }
    .empty-body  { font-family: 'IBM Plex Sans', sans-serif; font-size: 0.82rem; color: var(--text-dim); line-height: 1.65; max-width: 340px; margin: 0 auto; }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: var(--surface2) !important; border: 1px solid var(--border) !important;
        border-radius: 4px !important; font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.7rem !important; color: var(--text-faint) !important;
    }
    .streamlit-expanderContent {
        background: var(--surface2) !important; border: 1px solid var(--border) !important;
        border-top: none !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] { background: var(--surface) !important; }
    section[data-testid="stSidebar"] p { font-size: 0.81rem !important; color: var(--text-dim) !important; line-height: 1.6 !important; }

    /* ── Suggestion pills ── */
    .sug-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 16px; }
    .sug-pill {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.74rem;
        color: var(--gold-text);
        background: rgba(212,168,83,0.06);
        border: 1px solid var(--gold-dim);
        border-radius: 20px;
        padding: 4px 12px;
        white-space: nowrap;
        opacity: 0.85;
        cursor: default !important;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: var(--gold) !important; }
    
    div[data-testid="stFormSubmitButton"] {
        margin-top: auto !important;
        display: flex !important;
        align-items: flex-end !important;}
    
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_contract_name(raw: str) -> str:
    ex_match = re.search(r'EX-[\d.]+[-_](.+)$', raw, re.IGNORECASE)
    if ex_match:
        desc = ex_match.group(1).replace('_', ' ').replace('-', ' ').strip()
        desc = ' '.join(w.capitalize() for w in desc.split())
        year_m = re.search(r'(\d{4})', raw)
        year = year_m.group(1) if year_m else None
        co_raw = re.split(r'_\d{2}_\d{2}_\d{4}', raw)[0]
        co = re.sub(r'([a-z])([A-Z])', r'\1 \2', co_raw).strip()
        co = ' '.join(w.capitalize() for w in co.split())
        return f"{co} — {desc} ({year})" if year else f"{co} — {desc}"
    return ' '.join(w.capitalize() for w in raw.replace('_', ' ').replace('-', ' ').split())[:80]

def make_unique_display_names(contracts: list[str]) -> dict:
    """
    Map raw contract titles to unique display names.

    If two contracts produce the same clean name, append a short
    hash to distinguish them — prevents silent collision in dropdown.

    Returns:
        {raw_title: display_name}
    """
    seen   = {}   # clean_name → first raw title that produced it
    result = {}   # raw_title  → display name

    for raw in contracts:
        name = clean_contract_name(raw)
        if name in seen:
            # Collision — append short hash to distinguish
            short       = hashlib.sha256(raw.encode()).hexdigest()[:4]
            result[raw] = f"{name} [{short}]"
            # Fix the first entry too if not already fixed
            first_raw = seen[name]
            if not result[first_raw].endswith("]"):
                first_short         = hashlib.sha256(first_raw.encode()).hexdigest()[:4]
                result[first_raw]   = f"{name} [{first_short}]"
        else:
            seen[name]  = raw
            result[raw] = name

    return result

def score_color(v: float) -> str:
    return "#4caf7d" if v >= 0.7 else "#e8b84b" if v >= 0.5 else "#e06060"


def score_bar_html(label: str, value: float, tooltip: str) -> str:
    pct   = int(value * 100)
    color = score_color(value)
    return f"""
    <div class="score-row-item">
        <div class="score-row-header">
            <span class="score-name">
                {label}
                <span class="tip-wrap">
                    <span class="tip-icon">?</span>
                    <span class="tip-box">{tooltip}</span>
                </span>
            </span>
            <span class="score-val" style="color:{color}">{value:.2f}</span>
        </div>
        <div class="score-track">
            <div class="score-fill" style="width:{pct}%; background:{color};"></div>
        </div>
    </div>"""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUGGESTED_QUESTIONS = [
    "What is the termination clause?",
    "What is the governing law?",
    "Are there non-compete clauses?",
    "Is there a confidentiality clause?",
    "What happens if a party breaches?",
]

COVERED_TOPICS = [
    "Termination conditions",
    "Governing law & jurisdiction",
    "Payment terms",
    "Confidentiality obligations",
    "Non-compete & non-solicitation",
    "Intellectual property ownership",
    "Liability & indemnification",
    "Renewal & expiry",
    "Breach & remedies",
    "Assignment of rights",
]

METRIC_TOOLTIPS = {
    "Faithfulness": "Is every sentence grounded in the source text? High score = no hallucination.",
    "Relevance":    "Does the answer address what you asked? High score = directly on-topic.",
    "Overall":      "Combined score. 0.70 or above passes quality checks.",
}

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

@st.cache_data
def load_contracts():
    with duckdb.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT title FROM contracts ORDER BY title").fetchall()
    return [row[0] for row in rows]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### RAG Evidence Chain")
    st.markdown("Answers questions about legal contracts and traces every sentence back to its source clause.")
    st.markdown("---")
    st.markdown("**Stack**")
    for item in ["LangGraph agents", "Groq LLM (Llama 3.3-70b)",
                 "FAISS + BM25 hybrid search", "Cross-encoder re-ranking", "DuckDB + Redis"]:
        st.markdown(f"· {item}")
    st.markdown("---")
    st.markdown("**Topics covered**")
    for t in COVERED_TOPICS:
        st.markdown(
            f"<p style='font-size:0.77rem; color:#6b7280; margin:2px 0; "
            f"font-family:IBM Plex Mono,monospace;'>· {t}</p>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<p style='font-size:0.7rem; color:#4a505c; margin-top:14px; line-height:1.5;'>"
        "💡 Select a specific contract for more focused answers.</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="app-title">🔗 RAG <span>Evidence</span> Chain</div>
<div class="app-subtitle">Legal contract Q&amp;A · Every answer traced to its source</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "trigger_search" not in st.session_state:
    st.session_state.trigger_search = False

# ---------------------------------------------------------------------------
# Search form
# ---------------------------------------------------------------------------

contracts      = load_contracts()
clean_names    = make_unique_display_names(contracts)
display_opts   = ["All contracts"] + [clean_names[c] for c in contracts]
display_to_raw = {clean_names[r]: r for r in contracts}


col_contract, col_question = st.columns([1.4, 3.4])

with col_contract:
    st.markdown('<div class="search-label">Contract</div>', unsafe_allow_html=True)

    contract_search = st.text_input(
        "Search contracts",
        placeholder="Type to filter...",
        label_visibility="collapsed",
        key="contract_search",
    )

    filtered_opts = [
        opt for opt in display_opts
        if contract_search.lower() in opt.lower()
    ] if contract_search.strip() else display_opts

    if not filtered_opts:
        filtered_opts = ["All contracts"]

    selected_display = st.selectbox(
        "Contract",
        filtered_opts,
        label_visibility="collapsed",
    )
    selected_contract = (
        display_to_raw.get(selected_display)
        if selected_display != "All contracts" else "All contracts"
    )

with col_question:
    st.markdown('<div class="search-label">Your question</div>', unsafe_allow_html=True)
    with st.form("search_form", clear_on_submit=False):
        q_col, btn_col = st.columns([5, 1])
        with q_col:
            question = st.text_input(
                "question",
                value=st.session_state.question_input,
                placeholder="e.g. What is the termination clause?",
                label_visibility="collapsed",
                key="main_question",
            )
            st.session_state.question_input = question
        with btn_col:
            search_clicked = st.form_submit_button("→", use_container_width=True)

# Handle trigger from suggestion pills
if st.session_state.trigger_search:
    search_clicked = True
    st.session_state.trigger_search = False

# ---------------------------------------------------------------------------
# Suggestion pills — hidden once user starts typing
# ---------------------------------------------------------------------------

if not question.strip():
    st.markdown(
        "<p style='font-family:IBM Plex Mono,monospace; font-size:0.65rem; color:#4a505c; "
        "letter-spacing:0.1em; text-transform:uppercase; margin:12px 0 6px;'>Try asking</p>",
        unsafe_allow_html=True,
    )
    suggestions_html = "".join([
        f'<span class="sug-pill">{sug}</span>'
        for sug in SUGGESTED_QUESTIONS
    ])
    st.markdown(f'<div class="sug-row">{suggestions_html}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

active_question = question.strip()
run_search = search_clicked and active_question

if run_search:
    filters = {"contract_title": selected_contract} if selected_contract != "All contracts" else None

    with st.spinner("Searching contracts..."):
        pipeline = load_pipeline()
        result   = pipeline.run(active_question, filters=filters)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Empty state ──────────────────────────────────────────────────────────
    no_answer = (
        not result["sentences"] or
        not result["answer"].strip() or
        result["answer"].strip() == "No relevant chunks were retrieved."
    )

    if no_answer:
        scope = (f'in <b>{clean_contract_name(selected_contract)}</b>'
                 if selected_contract != "All contracts" else "across all contracts")
        st.markdown(f"""
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <div class="empty-title">No relevant clauses found</div>
            <div class="empty-body">
                Nothing matched <i>"{active_question}"</i> {scope}.<br><br>
                Try rephrasing, selecting a specific contract, or choosing
                a topic from the sidebar list.
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        # ── Badge ────────────────────────────────────────────────────────────
        if result["passed"]:
            st.markdown('<span class="badge badge-pass">✓ Passed evaluation</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-review">⚠ Low confidence — review recommended</span>',
                        unsafe_allow_html=True)

        # ── Answer ───────────────────────────────────────────────────────────
        answer_text = result["answer"].replace(
            "\n\n[⚠️ This answer has been flagged for human review due to low confidence scores.]", "")
        st.markdown(f'<div class="answer-text">{answer_text}</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Evidence + Scores ─────────────────────────────────────────────────
        col_ev, col_sc = st.columns([3, 2], gap="large")

        with col_ev:
            st.markdown('<div class="section-label">Evidence Chain</div>', unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:0.78rem; color:#4a505c; "
                "font-family:IBM Plex Sans,sans-serif; margin-bottom:12px;'>"
                "Every sentence traced to its source clause.</p>",
                unsafe_allow_html=True,
            )

            for i, s in enumerate(result["sentences"], 1):
                conf = s["confidence"]
                conf_cls   = "ev-conf-high" if conf >= 0.8 else "ev-conf-mid" if conf >= 0.6 else "ev-conf-low"
                conf_label = "high" if conf >= 0.8 else "mid" if conf >= 0.6 else "low"
                display_title = clean_contract_name(s["contract_title"])

                st.markdown(f"""
                <div class="ev-card">
                    <div class="ev-num">SENTENCE {i}</div>
                    <div class="ev-text">{s['text']}</div>
                    <div class="ev-source">
                        <span>← {display_title[:60]}</span>
                        <span style="color:#2a3040">·</span>
                        <span class="{conf_cls}">confidence {conf:.2f} ({conf_label})</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                with st.expander(f"View source · chunk {i}"):
                    st.markdown(
                        f"<p style='font-family:IBM Plex Mono,monospace; font-size:0.75rem; "
                        f"color:#6b7280; line-height:1.65;'>{s['chunk_text'][:600]}…</p>",
                        unsafe_allow_html=True,
                    )

        with col_sc:
            scores = result["eval_scores"]
            st.markdown(f"""
            <div class="score-card">
                <div class="score-card-title">Evaluation Scores</div>
                {score_bar_html("Faithfulness", scores["faithfulness"], METRIC_TOOLTIPS["Faithfulness"])}
                {score_bar_html("Relevance",    scores["relevance"],    METRIC_TOOLTIPS["Relevance"])}
                <div class="score-divider"></div>
                {score_bar_html("Overall",      scores["overall"],      METRIC_TOOLTIPS["Overall"])}
                <div class="score-threshold">Pass threshold · 0.70</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Evidence Graph ────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Evidence Graph</div>', unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.78rem; color:#4a505c; font-family:IBM Plex Sans,sans-serif; "
            "margin-bottom:10px;'>Question → answer sentences → source chunks</p>",
            unsafe_allow_html=True,
        )

        sentences = result["sentences"]
        n = len(sentences)
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        edge_x, edge_y = [], []

        node_x.append(0.5); node_y.append(1.0)
        node_text.append(f"Q: {active_question[:45]}...")
        node_color.append("#c8a96e"); node_size.append(18)

        for i in range(n):
            sx = (i + 1) / (n + 1)
            node_x.append(sx); node_y.append(0.5)
            node_text.append(f"S{i+1}: {sentences[i]['text'][:35]}...")
            node_color.append("#5b8dee"); node_size.append(14)
            edge_x += [0.5, sx, None]; edge_y += [1.0, 0.5, None]

        seen_chunks = {}
        for i, s in enumerate(sentences):
            cid = s["chunk_id"]
            if cid not in seen_chunks:
                cx = (i + 1) / (n + 1)
                seen_chunks[cid] = (cx, 0.0)
                node_x.append(cx); node_y.append(0.0)
                node_text.append(f"{clean_contract_name(s['contract_title'])[:28]}...")
                node_color.append("#3a7d5a"); node_size.append(12)
            sx = (i + 1) / (n + 1)
            cx, _ = seen_chunks[cid]
            edge_x += [sx, cx, None]; edge_y += [0.5, 0.0, None]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                  line=dict(color="#1f2430", width=1.5), hoverinfo="none"))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                  marker=dict(color=node_color, size=node_size,
                                              line=dict(color="#0a0b0d", width=2)),
                                  text=node_text, textposition="bottom center",
                                  textfont=dict(family="IBM Plex Mono, monospace",
                                                size=8, color="#4a505c"),
                                  hoverinfo="text"))
        fig.update_layout(showlegend=False, paper_bgcolor="#0a0b0d", plot_bgcolor="#0a0b0d",
                          margin=dict(l=10, r=10, t=10, b=50), height=280,
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Feedback ──────────────────────────────────────────────────────────
        st.markdown(
            "<p style='font-family:IBM Plex Mono,monospace; font-size:0.65rem; color:#4a505c; "
            "letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px;'>"
            "Was this helpful?</p>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="fb-wrap">', unsafe_allow_html=True)
        fb1, fb2, _ = st.columns([0.7, 0.7, 5])
        with fb1:
            if st.button("👍 Yes", key="fb_yes"):
                st.success("Thanks for the feedback!")
        with fb2:
            if st.button("👎 No", key="fb_no"):
                st.info("Thanks — we'll use this to improve.")
        st.markdown('</div>', unsafe_allow_html=True)

elif search_clicked and not active_question:
    st.markdown(
        "<p style='color:#e06060; font-family:IBM Plex Mono,monospace; "
        "font-size:0.8rem; margin-top:8px;'>Please enter a question first.</p>",
        unsafe_allow_html=True,
    )
