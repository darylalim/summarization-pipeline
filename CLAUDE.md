# CLAUDE.md

Streamlit web app for summarizing news articles using [bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) by Facebook. Articles are extracted from URLs using [newspaper4k](https://github.com/AndyTheFactory/newspaper4k). Long articles are split into token-aware chunks before summarization.

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
- Use dataclasses and abstract base classes

## Dependencies

- `newspaper4k` — article extraction from URLs
- `lxml_html_clean` — required by newspaper4k for HTML cleaning
- `nltk` — NLP features for newspaper4k
- `transformers` — Hugging Face model loading and generation
- `torch` — tensor operations
- `streamlit` — web UI

## Configuration

`pyproject.toml` — ruff lint isort (`combine-as-imports`), pytest (`pythonpath`), ty (`python-version = "3.12"`)

## Architecture

`streamlit_app.py` — single-file app

### Model

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
```

### Performance

- `@st.cache_resource` caches the model
- Device priority: MPS > CUDA > CPU
- `model.eval()` disables dropout; `torch.inference_mode()` disables autograd
- Token-based chunking splits articles exceeding 1024 tokens
- Tokenizer truncation: `max_length=1024` (bart-large-cnn max position embeddings)
- Generation: `max_length=130, min_length=30, num_beams=4, do_sample=False, length_penalty=1.0, early_stopping=True, no_repeat_ngram_size=3`
- Timing: `time.perf_counter()` (fractional seconds)

### Error Handling

Unexpected exceptions shown with `st.exception()`.

### JSON Download

Fields in the downloadable JSON via `st.download_button`:

- `model` — model name
- `url` — article URL
- `title` — article title
- `authors` — article authors
- `publish_date` — article publish date (ISO format)
- `response` — generated summary text
- `total_duration` — generation time in seconds
- `chunk_count` — number of text chunks
- `prompt_eval_count` — total input tokens across all chunks
- `eval_count` — total output tokens across all chunks

`st.metric` displays `model`, `total_duration`, `prompt_eval_count`, and `eval_count`.

## Tests

- `tests/test_streamlit_app.py` — unit tests for `get_device`, `extract`, `chunk`, `summarize` (mocked, no model download)
