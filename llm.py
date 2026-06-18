"""Helpers for connecting to Ollama and generating text responses."""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def is_ollama_running(timeout: float = 1.5) -> bool:
    """Return True when Ollama API is reachable."""
    try:
        res = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        return res.status_code == 200
    except requests.RequestException:
        return False


def list_models(timeout: float = 3.0) -> list[str]:
    """List installed Ollama models."""
    try:
        res = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        res.raise_for_status()
        data = res.json()
        models = data.get("models", [])
        names = [m.get("name", "") for m in models if m.get("name")]
        return names
    except requests.RequestException:
        return []


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    timeout: float = 120.0,
) -> tuple[bool, str]:
    """Call Ollama generate endpoint and return (success, text)."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": float(temperature)},
    }

    try:
        res = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=timeout,
        )
        if res.status_code != 200:
            return (
                False,
                f"Model request failed ({res.status_code}). Using fallback summary instead.",
            )

        data = res.json()
        text = (data.get("response") or "").strip()
        if not text:
            return False, "Model returned an empty response. Using fallback summary instead."

        return True, text

    except requests.RequestException:
        return (
            False,
            "Could not reach Ollama. Make sure `ollama serve` is running.",
        )


def build_summary_prompt(insights_text: str) -> str:
    """Prompt template for executive summary generation."""
    return (
        "You are a senior analytics consultant. "
        "Write a concise executive summary for business stakeholders.\n\n"
        "Requirements:\n"
        "- 4 to 6 bullet points\n"
        "- Highlight key patterns, risks, and opportunities\n"
        "- Use plain business language\n"
        "- If data quality issues appear, mention them clearly\n"
        "- Do not invent numbers\n\n"
        "Dataset insights:\n"
        f"{insights_text}\n"
    )


def build_question_prompt(insights_text: str, question: str) -> str:
    """Prompt template for question answering from summarized insights."""
    return (
        "Answer the user question using only the dataset insights below.\n"
        "If information is missing, say that clearly instead of guessing.\n\n"
        "Dataset insights:\n"
        f"{insights_text}\n\n"
        f"User question: {question}\n"
    )
