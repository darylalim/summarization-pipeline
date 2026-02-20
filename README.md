# Summarization Pipeline

Streamlit web app that converts PDF documents to Markdown using [Docling](https://docling-project.github.io/docling/) and summarizes text using [Falconsai/text_summarization](https://huggingface.co/Falconsai/text_summarization).

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
uv run pyright             # typecheck
uv run pytest              # test
```

Configuration is in `pyproject.toml`.
