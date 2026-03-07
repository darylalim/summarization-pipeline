# CLAUDE.md

Streamlit web app for summarizing news articles using [bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) by Facebook. Articles are extracted from URLs using [newspaper4k](https://github.com/AndyTheFactory/newspaper4k). Long articles are split into token-aware chunks before summarization. Summaries accumulate into a session collection that can be reordered, removed, and exported as JSON or CSV.

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
- `transformers` — Hugging Face model loading and generation
- `torch` — tensor operations
- `streamlit` — web UI

## Configuration

`pyproject.toml` — ruff lint isort (`combine-as-imports`), pytest (`pythonpath`), ty (`python-version = "3.12"`)

## Architecture

`streamlit_app.py` — single-file app

### Model

```python
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
```

### Constants

`DEFAULT_GENERATION_PARAMS` — default generation settings used as sidebar defaults and fallback when no custom params are provided:

```python
DEFAULT_GENERATION_PARAMS: dict[str, int | float | bool] = {
    "max_length": 130,
    "min_length": 30,
    "num_beams": 4,
    "do_sample": False,
    "length_penalty": 1.0,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
}
```

### Session State

`st.session_state.collection` — `list[dict]` accumulating summary results across the session. Each dict contains article metadata, summary text, metrics, and generation parameters. Initialized to `[]` on first load.

### Layout

- **Sidebar** — "Generation Settings" expander (sliders/checkboxes for all generation parameters, "Reset to Defaults" button) and "Export" section (JSON and CSV download buttons)
- **Main area** — URL input, Summarize button, and collection cards

### Functions

- `get_device() -> str` — detects best device (MPS > CUDA > CPU)
- `load_model(device) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]` — loads model and tokenizer, cached with `@st.cache_resource`
- `extract(url) -> Article` — downloads, parses, and runs NLP on a news article
- `chunk(text, tokenizer) -> list[str]` — splits text into token-aware chunks of up to 1024 tokens
- `collection_to_csv(collection) -> str` — converts collection to CSV, joining `authors`/`keywords` with semicolons and flattening `generation_params` to individual columns
- `summarize(chunks, model, tokenizer, device, generation_params) -> tuple[str, int, int]` — summarizes text chunks with configurable generation parameters, returns (response, prompt_eval_count, eval_count)

### Collection Cards

Each summarized article displays:

- Title, metadata (authors, publish date, keywords), and URL
- Side-by-side original text and summary (read-only text areas)
- Metrics: Duration (s), Original Words, Summary Words, Compression Ratio, Model, Chunks, Prompt Tokens, Output Tokens
- Expandable "Generation Parameters" showing params used
- Reorder (Up/Down) and Remove buttons

### Performance

- `@st.cache_resource` caches the model
- Device priority: MPS > CUDA > CPU
- `model.eval()` disables dropout; `torch.inference_mode()` disables autograd
- Token-based chunking splits articles exceeding 1024 tokens
- Tokenizer truncation: `max_length=1024` (bart-large-cnn max position embeddings)
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
- `generation_params` — generation parameters used (nested in JSON; flattened to columns in CSV)

## Tests

`tests/test_streamlit_app.py` — unit tests (mocked, no model download):

- `TestDefaultGenerationParams` — verifies constant keys and values
- `TestGetDevice` — MPS, CUDA, CPU detection
- `TestExtract` — article download/parse, error propagation
- `TestChunk` — short text, long text, boundaries, empty input
- `TestSummarize` — single/multi-chunk, custom generation params, empty input
- `TestCollectionToCsv` — single/multi-item, flattened params, empty authors/keywords, empty collection
