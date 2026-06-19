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

      :root {
        --plum: #5A3050;
        --plum-soft: #D8B4CC;
        --teal: #0F766E;
        --ink: #2A1A1C;
      }

      html, body, [class*="css"], .stMarkdown, .stButton button {
        font-family: 'Plus Jakarta Sans', sans-serif;
      }
      h1, h2, h3 { letter-spacing: -0.5px; font-weight: 800; }
      code, .mono { font-family: 'DM Mono', monospace; }

      /* App background: soft animated gradient wash */
      .stApp {
        background:
          radial-gradient(1200px 600px at 10% -10%, #F3E7F0 0%, rgba(243,231,240,0) 55%),
          radial-gradient(1000px 500px at 110% 10%, #E3F1EE 0%, rgba(227,241,238,0) 50%),
          #FAF7F4;
      }

      @keyframes rise {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
      }

      /* Hero banner */
      .hero {
        position: relative;
        border-radius: 22px;
        padding: 40px 44px;
        margin: 6px 0 26px 0;
        color: #fff;
        background: linear-gradient(120deg, #5A3050 0%, #3C2A4D 45%, #0F766E 110%);
        box-shadow: 0 18px 40px -18px rgba(90,48,80,0.55);
        overflow: hidden;
        animation: rise 0.6s ease both;
      }
      .hero::after {
        content: "";
        position: absolute;
        top: -60px; right: -40px;
        width: 280px; height: 280px;
        background: radial-gradient(circle, rgba(216,180,204,0.45) 0%, rgba(216,180,204,0) 70%);
        filter: blur(8px);
      }
      .hero-eyebrow {
        font-family: 'DM Mono', monospace;
        font-size: 11px; letter-spacing: 3px; text-transform: uppercase;
        color: #E9D2E2; margin-bottom: 10px;
      }
      .hero h1 {
        font-size: 42px; line-height: 1.05; margin: 0 0 12px 0; color: #fff;
      }
      .hero p { font-size: 15px; max-width: 560px; color: #F1E6EE; margin: 0; line-height: 1.6; }

      .app-eyebrow {
        font-family: 'DM Mono', monospace;
        font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
        color: var(--plum);
      }

      .status-pill {
        font-family: 'DM Mono', monospace; font-size: 11px; font-weight: 500;
        padding: 5px 13px; border-radius: 20px; display: inline-block;
      }
      .status-on  { background: #DCF5EE; color: #0F766E; }
      .status-off { background: #F6E9E3; color: #9A5B3B; }

      .summary-card {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(90,48,80,0.12);
        border-left: 4px solid var(--plum);
        padding: 22px 26px; border-radius: 14px; line-height: 1.75;
        box-shadow: 0 12px 30px -22px rgba(42,26,28,0.5);
        animation: rise 0.5s ease both;
      }

      /* Glassmorphism metric cards */
      div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.65);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(90,48,80,0.10);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 10px 26px -20px rgba(42,26,28,0.55);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        animation: rise 0.5s ease both;
      }
      div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 18px 34px -20px rgba(90,48,80,0.55);
      }
      div[data-testid="stMetricValue"] {
        font-weight: 800; color: var(--plum);
      }
      div[data-testid="stMetricLabel"] {
        font-family: 'DM Mono', monospace;
        text-transform: uppercase; letter-spacing: 1.5px; font-size: 11px;
      }

      /* Section headings with accent tick */
      .stMarkdown h4 {
        position: relative; padding-left: 14px; margin-top: 8px;
      }
      .stMarkdown h4::before {
        content: ""; position: absolute; left: 0; top: 4px; bottom: 4px;
        width: 4px; border-radius: 4px;
        background: linear-gradient(180deg, var(--plum), var(--teal));
      }

      /* Buttons */
      .stButton button, .stDownloadButton button {
        border-radius: 10px; font-weight: 700; border: none;
        background: linear-gradient(120deg, var(--plum), #3C2A4D);
        color: #fff; transition: transform 0.15s ease, box-shadow 0.15s ease;
      }
      .stButton button:hover, .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 22px -12px rgba(90,48,80,0.6);
        color: #fff;
      }

      /* Upload dropzone */
      section[data-testid="stFileUploaderDropzone"] {
        border: 2px dashed rgba(90,48,80,0.35);
        border-radius: 16px;
        background: rgba(255,255,255,0.5);
      }

      /* Dataframe rounding */
      div[data-testid="stDataFrame"] {
        border-radius: 14px; overflow: hidden;
        box-shadow: 0 10px 26px -22px rgba(42,26,28,0.55);
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
st.markdown(
    """
    <div class="hero">
      <div class="hero-eyebrow">Local-first analytics</div>
      <h1>AI Data Analyst</h1>
      <p>Upload a dataset to get an executive summary, interactive charts, and
      plain-English answers — running entirely on your machine.</p>
    </div>
    """,
    unsafe_allow_html=True,
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

# ── Charts ────────────────────────────────────────────────────────────────────
numeric_cols = profile["numeric_cols"]
categorical_cols = profile["categorical_cols"]

if numeric_cols:
    st.markdown("#### Charts")
    chart_cols = st.columns(2)

    # Top categories by a numeric measure (bar chart)
    with chart_cols[0]:
        if categorical_cols:
            cat_col = st.selectbox("Group by", categorical_cols, key="cat_col")
            measure = st.selectbox("Measure", numeric_cols, key="bar_measure")
            grouped = (
                df.groupby(cat_col, dropna=False)[measure]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )
            st.caption(f"Total {measure} by {cat_col} (top 10)")
            st.bar_chart(grouped)
        else:
            st.caption("No categorical columns to group by.")

    # Relationship between two numeric columns (scatter)
    with chart_cols[1]:
        if len(numeric_cols) >= 2:
            default_y = numeric_cols[1]
            x_col = st.selectbox("X axis", numeric_cols, index=0, key="scatter_x")
            y_col = st.selectbox(
                "Y axis", numeric_cols, index=numeric_cols.index(default_y), key="scatter_y"
            )
            st.caption(f"{x_col} vs {y_col}")
            st.scatter_chart(df[[x_col, y_col]], x=x_col, y=y_col)
        else:
            st.caption("Need at least two numeric columns for a scatter plot.")

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
