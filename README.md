# AI Data Analyst

A local-first Streamlit app that turns CSV/Excel files into executive summaries and quick Q&A.

**[▶ Live demo](https://slaisha-ai-data-analyst-app-fqsp4g.streamlit.app/)**

![Demo](demo.gif)

> Privacy-first: your data stays on your machine. When a local Ollama model is
> available it generates narrative summaries; otherwise the app falls back to
> deterministic pandas analysis — so it never breaks in a demo.

## Features

- Upload CSV, XLSX, or XLS files
- Automatic pandas profiling (shape, stats, missing values, correlations)
- Executive summary generation
- Natural-language question answering
- Ollama integration for local LLM responses
- Rule-based fallback when Ollama is unavailable

## Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py --server.port 8502
```

Open: http://localhost:8502

## Optional: enable local model

```bash
ollama pull llama3.1:8b
ollama serve
```

When Ollama is online, the sidebar will show "Ollama connected".
