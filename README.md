# News Article Summarizer

Streamlit web app for summarizing news articles using [granite-4.0-h-1b](https://huggingface.co/ibm-granite/granite-4.0-h-1b) by IBM via `mlx-lm`. Articles are extracted from URLs using [newspaper4k](https://github.com/AndyTheFactory/newspaper4k). Long articles are split into token-aware chunks before summarization. Summaries accumulate into a session collection that can be reordered, removed, and exported as JSON or CSV.

Requires Apple Silicon (M-series Mac).

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

The model (~2.9 GB) downloads automatically on first run.

## Usage

1. Paste a news article URL and click **Summarize**
2. The article is extracted, chunked, and summarized, then added to the session collection
3. View each article's original text and summary side by side
4. Review metrics: word counts, compression ratio, duration, token counts
5. Reorder or remove articles from the collection
6. Adjust generation parameters in the sidebar
7. Export the full collection as JSON or CSV

## Features

- **Session collection** — summaries accumulate across the session
- **Side-by-side display** — original article text and generated summary in two columns
- **Generation controls** — sidebar sliders for max_tokens, temp, top_p, and repetition_penalty, with a reset-to-defaults button
- **Article metadata** — title, authors, publish date, keywords, and source URL
- **Word count metrics** — original word count, summary word count, and compression ratio
- **Collection management** — reorder (Up/Down) and remove articles
- **JSON/CSV export** — download the full collection with metadata, metrics, and generation parameters

## Development

```bash
uv run ruff check .    # lint
uv run ruff format .   # format
uv run ty check        # typecheck
uv run pytest          # test
```

Configuration is in `pyproject.toml`.
