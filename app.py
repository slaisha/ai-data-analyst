"""AI Data Analyst — a local-first tool for exploring any dataset.

Upload a CSV/Excel file and get an executive summary plus natural-language Q&A.
Runs entirely on your machine: uses a local Ollama model when available, and
falls back to rule-based analysis (pandas) when it is not.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

import analysis
import llm

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="◆",
    layout="wide",
)

# ── Typography + light styling (matches portfolio fonts) ──────────────────────
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

      html, body, [class*="css"], .stMarkdown, .stButton button {
        font-family: 'Plus Jakarta Sans', sans-serif;
      }
      h1, h2, h3 { letter-spacing: -0.5px; font-weight: 800; }
      code, .mono { font-family: 'DM Mono', monospace; }

      .app-eyebrow {
        font-family: 'DM Mono', monospace;
        font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
        color: #0F766E;
      }
      .status-pill {
        font-family: 'DM Mono', monospace; font-size: 11px;
        padding: 4px 12px; border-radius: 20px; display: inline-block;
      }
      .status-on  { background: #DCF5EE; color: #0F766E; }
      .status-off { background: #F3E8E8; color: #9A3B3B; }
      .summary-card {
        background: #F1EFEA; border-left: 3px solid #0F766E;
        padding: 20px 24px; border-radius: 8px; line-height: 1.7;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_dataframe(uploaded) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file into a DataFrame."""
    name = uploaded.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    return pd.read_csv(uploaded)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="app-eyebrow">Configuration</div>', unsafe_allow_html=True)
    st.markdown("### Model settings")

    running = llm.is_ollama_running()
    available = llm.list_models() if running else []

    if running:
        st.markdown(
            '<span class="status-pill status-on">● Ollama connected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill status-off">● Ollama offline — using fallback</span>',
            unsafe_allow_html=True,
        )

    model_name = st.text_input("Model name", value=llm.DEFAULT_MODEL)
    if available:
        st.caption("Installed models: " + ", ".join(available))

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.divider()
    st.caption(
        "Local-first: when Ollama is unavailable, summaries are generated from "
        "pandas analysis instead. Only summarized insights — never your full "
        "dataset — are sent to the model."
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="app-eyebrow">Local-first analytics</div>', unsafe_allow_html=True)
st.title("AI Data Analyst")
st.write(
    "Upload a dataset to get an executive summary and ask questions in plain "
    "English. Runs entirely on your machine."
)

uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info("Upload a file to begin, or try the sample dataset in `sample_data/`.")
    st.stop()

try:
    df = load_dataframe(uploaded)
except Exception as exc:  # noqa: BLE001 — surface a friendly message at this boundary
    st.error(f"Could not read that file: {exc}")
    st.stop()

if df.empty:
    st.warning("That file loaded but contains no rows.")
    st.stop()

profile = analysis.profile_dataframe(df)
insights_text = analysis.insights_to_text(profile)

# ── Overview metrics ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{profile['n_rows']:,}")
c2.metric("Columns", profile["n_cols"])
c3.metric("Numeric", len(profile["numeric_cols"]))
c4.metric("Categorical", len(profile["categorical_cols"]))

st.markdown("#### Data preview")
st.dataframe(df.head(50), use_container_width=True)

# ── Executive summary ─────────────────────────────────────────────────────────
st.markdown("#### Executive summary")

if running:
    with st.spinner("Generating summary with the local model…"):
        ok, text = llm.generate(
            llm.build_summary_prompt(insights_text),
            model=model_name,
            temperature=temperature,
        )
    if ok:
        st.markdown(f'<div class="summary-card">{text}</div>', unsafe_allow_html=True)
    else:
        st.warning(text)
        st.markdown(
            f'<div class="summary-card">{analysis.rule_based_summary(profile)}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Shown above: rule-based fallback summary.")
else:
    st.markdown(
        f'<div class="summary-card">{analysis.rule_based_summary(profile)}</div>',
        unsafe_allow_html=True,
    )
    st.caption("Rule-based summary (start Ollama for AI-generated summaries).")

# ── Q&A ───────────────────────────────────────────────────────────────────────
st.markdown("#### Ask a question")
question = st.text_input(
    "e.g. Which category has the highest total? What stands out in this data?"
)

if question:
    if running:
        with st.spinner("Thinking…"):
            ok, text = llm.generate(
                llm.build_question_prompt(insights_text, question),
                model=model_name,
                temperature=temperature,
            )
        if ok:
            st.markdown(text)
        else:
            st.warning(text)
            st.markdown(analysis.answer_question_rule_based(profile, question))
    else:
        st.markdown(analysis.answer_question_rule_based(profile, question))

# ── What gets sent to the model ───────────────────────────────────────────────
with st.expander("See exactly what is sent to the model"):
    st.caption("Only this summarized profile leaves your machine — never raw rows.")
    st.code(insights_text, language="text")
