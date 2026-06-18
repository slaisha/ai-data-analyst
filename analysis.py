"""Data profiling and rule-based fallback utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _to_float(value: Any) -> float | None:
    """Convert numeric-like values to plain Python float."""
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact, JSON-serializable profile of a DataFrame."""
    work = df.copy()

    numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in work.columns if c not in numeric_cols]

    numeric_stats: dict[str, dict[str, float | None]] = {}
    for col in numeric_cols:
        series = work[col].dropna()
        if series.empty:
            numeric_stats[col] = {
                "min": None,
                "mean": None,
                "median": None,
                "max": None,
                "sum": None,
            }
            continue
        numeric_stats[col] = {
            "min": _to_float(series.min()),
            "mean": _to_float(series.mean()),
            "median": _to_float(series.median()),
            "max": _to_float(series.max()),
            "sum": _to_float(series.sum()),
        }

    categorical_stats: dict[str, dict[str, Any]] = {}
    for col in categorical_cols:
        counts = work[col].astype("string").fillna("<missing>").value_counts()
        top_values = []
        for idx, val in counts.head(5).items():
            top_values.append({"value": str(idx), "count": int(val)})
        categorical_stats[col] = {
            "unique": int(work[col].nunique(dropna=True)),
            "top_values": top_values,
        }

    missing_values = {col: int(work[col].isna().sum()) for col in work.columns}

    strong_correlations: list[dict[str, Any]] = []
    if len(numeric_cols) >= 2:
        corr = work[numeric_cols].corr(numeric_only=True)
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1 :]:
                value = corr.loc[col1, col2]
                if pd.notna(value) and abs(float(value)) >= 0.6:
                    strong_correlations.append(
                        {"col1": col1, "col2": col2, "corr": float(value)}
                    )
        strong_correlations.sort(key=lambda x: abs(x["corr"]), reverse=True)

    return {
        "n_rows": int(len(work)),
        "n_cols": int(len(work.columns)),
        "columns": [str(c) for c in work.columns],
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "missing_values": missing_values,
        "strong_correlations": strong_correlations,
    }


def _fmt_number(v: float | None) -> str:
    if v is None:
        return "n/a"
    if abs(v) >= 1000:
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"{v:.2f}".rstrip("0").rstrip(".")


def insights_to_text(profile: dict[str, Any]) -> str:
    """Render profile into concise text for LLM prompts."""
    lines: list[str] = []
    lines.append(f"Rows: {profile['n_rows']}, Columns: {profile['n_cols']}")
    lines.append("Column names: " + ", ".join(profile["columns"]))
    lines.append("")

    lines.append("Numeric columns (min / mean / median / max / sum):")
    if profile["numeric_cols"]:
        for col in profile["numeric_cols"]:
            s = profile["numeric_stats"][col]
            lines.append(
                "  - "
                + f"{col}: min={_fmt_number(s['min'])}, mean={_fmt_number(s['mean'])}, "
                + f"median={_fmt_number(s['median'])}, max={_fmt_number(s['max'])}, "
                + f"sum={_fmt_number(s['sum'])}"
            )
    else:
        lines.append("  - none")
    lines.append("")

    lines.append("Categorical columns (unique count, top values):")
    if profile["categorical_cols"]:
        for col in profile["categorical_cols"]:
            c = profile["categorical_stats"][col]
            top = ", ".join(
                [f"{item['value']} ({item['count']})" for item in c["top_values"][:3]]
            )
            lines.append(f"  - {col}: {c['unique']} unique | top: {top if top else 'n/a'}")
    else:
        lines.append("  - none")
    lines.append("")

    missing = [f"{k}: {v}" for k, v in profile["missing_values"].items() if v > 0]
    if missing:
        lines.append("Missing values: " + ", ".join(missing))
        lines.append("")

    lines.append("Strong correlations:")
    if profile["strong_correlations"]:
        for item in profile["strong_correlations"][:8]:
            lines.append(
                f"  - {item['col1']} vs {item['col2']}: {item['corr']:.2f}"
            )
    else:
        lines.append("  - none")

    return "\n".join(lines)


def rule_based_summary(profile: dict[str, Any]) -> str:
    """Generate a deterministic executive summary when LLM is unavailable."""
    lines: list[str] = []
    n_numeric = len(profile["numeric_cols"])
    n_categorical = len(profile["categorical_cols"])

    lines.append(
        f"This dataset contains **{profile['n_rows']:,} rows** across **{profile['n_cols']} columns** "
        f"({n_numeric} numeric, {n_categorical} categorical)."
    )

    if profile["numeric_cols"]:
        best_col = None
        best_sum = None
        for col in profile["numeric_cols"]:
            s = profile["numeric_stats"][col]["sum"]
            if s is None:
                continue
            if best_sum is None or abs(s) > abs(best_sum):
                best_col = col
                best_sum = s
        if best_col is not None:
            stat = profile["numeric_stats"][best_col]
            lines.append(
                f"\nThe largest numeric measure is **{best_col}**, totaling {_fmt_number(best_sum)} "
                f"with an average of {_fmt_number(stat['mean'])} "
                f"(ranging {_fmt_number(stat['min'])} to {_fmt_number(stat['max'])})."
            )

    if profile["categorical_cols"]:
        col = profile["categorical_cols"][0]
        c = profile["categorical_stats"][col]
        if c["top_values"]:
            top = c["top_values"][0]
            pct = (top["count"] / max(profile["n_rows"], 1)) * 100
            lines.append(
                f"\nIn **{col}**, the most common value is **{top['value']}** "
                f"({top['count']} rows, {pct:.1f}% of the data), out of {c['unique']} distinct values."
            )

    if profile["strong_correlations"]:
        top_corr = profile["strong_correlations"][0]
        direction = "positively" if top_corr["corr"] > 0 else "negatively"
        lines.append(
            f"\n**{top_corr['col1']}** and **{top_corr['col2']}** are strongly {direction} "
            f"correlated ({top_corr['corr']:.2f}), worth investigating."
        )

    total_missing = sum(profile["missing_values"].values())
    if total_missing:
        lines.append(f"\nThere are **{total_missing:,} missing values** across the dataset.")
    else:
        lines.append("\nNo missing values were detected — the dataset is complete.")

    return "\n".join(lines)


def answer_question_rule_based(profile: dict[str, Any], question: str) -> str:
    """Best-effort fallback answer for user questions."""
    q = question.lower().strip()

    mentioned_numeric = [
        col for col in profile["numeric_cols"] if col.lower() in q
    ]

    if not mentioned_numeric and profile["numeric_cols"]:
        mentioned_numeric = profile["numeric_cols"][:2]

    answers: list[str] = []

    for col in mentioned_numeric:
        s = profile["numeric_stats"][col]
        parts = []
        if any(k in q for k in ["total", "sum"]):
            parts.append(f"total {_fmt_number(s['sum'])}")
        if any(k in q for k in ["average", "mean", "avg"]):
            parts.append(f"average {_fmt_number(s['mean'])}")
        if any(k in q for k in ["min", "minimum", "lowest"]):
            parts.append(f"minimum {_fmt_number(s['min'])}")
        if any(k in q for k in ["max", "maximum", "highest"]):
            parts.append(f"maximum {_fmt_number(s['max'])}")

        if not parts:
            parts = [
                f"total {_fmt_number(s['sum'])}",
                f"average {_fmt_number(s['mean'])}",
                f"range {_fmt_number(s['min'])} to {_fmt_number(s['max'])}",
            ]

        answers.append(f"**{col}** — " + ", ".join(parts) + ".")

    if answers:
        return "Here is what the data shows for the columns you mentioned:\n\n" + "\n\n".join(answers)

    if "missing" in q or "null" in q:
        total_missing = sum(profile["missing_values"].values())
        return f"There are {total_missing:,} missing values in total."

    return (
        "I could not match that question to a specific column in fallback mode. "
        "Try mentioning a column name (for example: 'What is the total revenue?')."
    )
