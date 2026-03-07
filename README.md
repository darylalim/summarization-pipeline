# News Article Summarizer

Streamlit web app for summarizing news articles using [bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) by Facebook. Articles are extracted from URLs using [newspaper4k](https://github.com/AndyTheFactory/newspaper4k). Long articles are split into token-aware chunks before summarization.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Usage

1. Paste a news article URL and click **Summarize**
2. The article is extracted, chunked, and summarized — then added to your session collection
3. View each article's original text and summary side by side
4. Review metrics: word counts, compression ratio, duration, token counts
5. Manage your collection: reorder articles (Up/Down) or remove them
6. Adjust generation parameters (max_length, num_beams, etc.) in the sidebar
7. Export the full collection as JSON or CSV from the sidebar

## Features

- **Session collection** — summaries accumulate across the session, building a research collection
- **Side-by-side display** — original article text and generated summary shown in two columns
- **Generation controls** — sidebar expander with sliders and checkboxes for all generation parameters (max_length, min_length, num_beams, do_sample, length_penalty, early_stopping, no_repeat_ngram_size), with a reset-to-defaults button
- **Article metadata** — title, authors, publish date, keywords, and source URL
- **Word count metrics** — original word count, summary word count, and compression ratio
- **Collection management** — reorder (Up/Down) and remove articles from the collection
- **JSON and CSV export** — download the entire collection with full metadata, metrics, and generation parameters

## Development

```bash
uv run ruff check .        # lint
uv run ruff format .       # format
uv run ty check            # typecheck
uv run pytest              # test
```

Configuration is in `pyproject.toml`.
