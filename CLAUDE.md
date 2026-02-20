# CLAUDE.md

Streamlit web app for converting PDF documents to Markdown using [Docling](https://docling-project.github.io/docling/) and summarizing text using [bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) by Facebook.

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

- `docling` — PDF-to-Markdown conversion
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

- `@st.cache_resource` caches the model and `DocumentConverter`
- Docling models pre-downloaded to `~/.cache/docling/models/`
- Device priority: MPS > CUDA > CPU
- `model.eval()` disables dropout; `torch.inference_mode()` disables autograd
- Tokenizer truncation: `max_length=1024` (BART positional embedding limit)
- Generation: `max_length=1000, min_length=30, do_sample=False`
- Timing: `time.perf_counter()` (fractional seconds)

### Error Handling

Unexpected exceptions shown with `st.exception()`.

### JSON Download

Fields in the downloadable JSON via `st.download_button`:

- `model` — model name
- `response` — generated summary text
- `total_duration` — generation time in seconds
- `prompt_eval_count` — number of input tokens
- `eval_count` — number of output tokens

`st.metric` displays all fields except `response`.

## Tests

- `tests/test_streamlit_app.py` — unit tests for `get_device`, `convert`, `summarize` (mocked, no model download)
- `tests/data/pdf/test_solar_system.pdf` — single-page PDF
- `tests/data/pdf/test_ai_history.pdf` — two-page PDF
