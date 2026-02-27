# Summarization Pipeline

Streamlit web app for converting PDF documents using [Docling](https://docling-project.github.io/docling/) and summarizing text using [bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) by Facebook. Long documents are chunked with Docling's `HybridChunker` before summarization.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Usage

1. Upload a PDF file
2. Click **Summarize**
3. View the generated summary and metrics
4. Download results as JSON

## Development

```bash
uv run ruff check .        # lint
uv run ruff format .       # format
uv run ty check            # typecheck
uv run pytest              # test
```

Configuration is in `pyproject.toml`.
