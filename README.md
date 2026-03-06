# News Article Summarizer

Streamlit web app for summarizing news articles using [bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) by Facebook. Articles are extracted from URLs using [newspaper4k](https://github.com/AndyTheFactory/newspaper4k). Long articles are split into token-aware chunks before summarization.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Usage

1. Paste a news article URL
2. Click **Summarize**
3. View article metadata, generated summary, and metrics
4. Download results as JSON

## Development

```bash
uv run ruff check .        # lint
uv run ruff format .       # format
uv run ty check            # typecheck
uv run pytest              # test
```

Configuration is in `pyproject.toml`.
