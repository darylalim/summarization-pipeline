# CLAUDE.md

Streamlit web app for summarizing news articles using [granite-4.0-h-1b](https://huggingface.co/ibm-granite/granite-4.0-h-1b) by IBM via `mlx-lm`. Articles are extracted from URLs using [newspaper4k](https://github.com/AndyTheFactory/newspaper4k). Long articles are split into token-aware chunks before summarization. Summaries accumulate into a session collection that can be reordered, removed, and exported as JSON or CSV.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Commands

- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run ty check`
- **Test**: `uv run pytest`

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- isort with combine-as-imports (configured in `pyproject.toml`)

## Dependencies

- `newspaper4k` — article extraction from URLs
- `lxml_html_clean` — HTML cleaning (required by newspaper4k)
- `nltk` — NLP features (required by newspaper4k)
- `mlx-lm` — model loading and generation on Apple Silicon (mlx and transformers are transitive deps)
- `streamlit` — web UI

## Configuration

`pyproject.toml` — ruff lint isort (`combine-as-imports`), pytest (`pythonpath`), ty (`python-version = "3.12"`)

## Architecture

`streamlit_app.py` — single-file app

### Model

```python
from mlx_lm import generate, load
model, tokenizer = load("mlx-community/granite-4.0-h-1b-bf16")
```

### Constants

`MAX_CHUNK_TOKENS = 8192` — maximum tokens per chunk when splitting long articles.

`SUMMARIZE_PROMPT` — chat template user message prefix prepended to each chunk before summarization.

`DEFAULT_GENERATION_PARAMS` — default generation settings used as sidebar defaults and fallback when no custom params are provided:

```python
DEFAULT_GENERATION_PARAMS: dict[str, int | float] = {
    "max_tokens": 256,
    "temp": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.2,
}
```

### Session State

`st.session_state.collection` — `list[dict]` accumulating summary results across the session. Each dict contains article metadata, summary text, metrics, and generation parameters. Initialized to `[]` on first load.

### Layout

- **Sidebar** — "Generation Settings" expander (4 sliders: max_tokens, temp, top_p, repetition_penalty; "Reset to Defaults" button) and "Export" section (JSON and CSV download buttons)
- **Main area** — URL input, Summarize button, and collection cards

### Functions

- `load_model() -> tuple[nn.Module, PreTrainedTokenizerBase]` — loads model and tokenizer via `mlx_lm.load`, cached with `@st.cache_resource`
- `extract(url) -> Article` — downloads, parses, and runs NLP on a news article
- `chunk(text, tokenizer) -> list[str]` — splits text into token-aware chunks of up to `MAX_CHUNK_TOKENS` tokens
- `collection_to_csv(collection) -> str` — converts collection to CSV, joining `authors`/`keywords` with semicolons and flattening `generation_params` to individual columns
- `summarize(chunks, model, tokenizer, generation_params) -> tuple[str, int, int]` — summarizes text chunks using chat template prompting and `mlx_lm.generate`, returns (response, prompt_eval_count, eval_count)

### Collection Cards

Each summarized article displays:

- Title, metadata (authors, publish date, keywords), and URL
- Side-by-side original text and summary (read-only text areas)
- Metrics: Duration (s), Original Words, Summary Words, Compression Ratio, Model, Chunks, Prompt Tokens, Output Tokens
- Expandable "Generation Parameters" showing params used
- Reorder (Up/Down) and Remove buttons

### Performance

- `@st.cache_resource` caches the model
- MLX handles Apple Silicon (M-series) acceleration natively
- Token-based chunking splits articles exceeding `MAX_CHUNK_TOKENS` (8192) tokens
- Generation parameters configurable via sidebar; defaults in `DEFAULT_GENERATION_PARAMS`
- Timing: `time.perf_counter()` (fractional seconds)

### Error Handling

Unexpected exceptions shown with `st.exception()`.

### JSON/CSV Export

Sidebar export of the full session collection via `st.download_button` (disabled when empty).

Fields in each collection item:

- `model` — model name
- `url` — article URL
- `title` — article title
- `authors` — article authors
- `publish_date` — article publish date (ISO format)
- `keywords` — article keywords (from newspaper4k NLP)
- `original_text` — full original article text
- `response` — generated summary text
- `total_duration` — generation time in seconds
- `chunk_count` — number of text chunks
- `prompt_eval_count` — total input tokens across all chunks
- `eval_count` — total output tokens across all chunks
- `original_word_count` — word count of original article
- `summary_word_count` — word count of generated summary
- `compression_ratio` — summary_word_count / original_word_count
- `generation_params` — generation parameters used (max_tokens, temp, top_p, repetition_penalty; nested in JSON; flattened to columns in CSV)

## Tests

`tests/test_streamlit_app.py` — unit tests (mocked, no model download):

- `TestDefaultGenerationParams` — verifies constant keys and values
- `TestExtract` — article download/parse, error propagation
- `TestChunk` — short text, long text, boundaries, empty input (uses `MAX_CHUNK_TOKENS`)
- `TestSummarize` — single/multi-chunk, custom generation params, empty input (mocks `mlx_lm.generate`)
- `TestCollectionToCsv` — single/multi-item, flattened params, empty authors/keywords, empty collection
