# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app for converting PDF documents to Markdown using [Docling](https://docling-project.github.io/docling/) and summarizing text using [text_summarization](https://huggingface.co/Falconsai/text_summarization) transformer model by [Falcons.ai](https://falcons.ai/).

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Commands

- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run pyright`
- **Test**: `uv run pytest`

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- isort with combine-as-imports (configured in `pyproject.toml`)
- Use dataclasses and abstract base classes

## Dependencies

- `docling` - PDF document to Markdown conversion
- `transformers` - Hugging Face model loading and generation
- `torch` - Tensor operations
- `streamlit` - Web user interface

## Configuration

`pyproject.toml` — ruff lint isort (`combine-as-imports`), pytest (`pythonpath`), and pyright (`pythonVersion = "3.12"`).

## Architecture

### Entry Point

`streamlit_app.py` - single-file app.

### Usage

```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization")
```

### Performance

- `@st.cache_resource` to cache both the model and the `DocumentConverter`
- Docling models are pre-downloaded to `~/.cache/docling/models/` inside the cached `DocumentConverter` initializer
- Auto-detect device priority: MPS > CUDA > CPU
- `model.eval()` to disable dropout during inference
- `torch.inference_mode()` for inference (disables autograd and version counting)
- Generated summary: `max_length=1000, min_length=30, do_sample=False`
- `time.perf_counter()` for timing (fractional seconds)

### Error Handling

- Unexpected exceptions shown with `st.exception()` for debugging

### Summary Display

`st.write(response)` displays the generated summary under a "Summary" subheader.

### JSON Download

Fields in the downloadable JSON via `st.download_button`:

- `model` (string): Model name
- `response` (string): The model's generated text response
- `total_duration` (float): Time spent generating the response in seconds
- `prompt_eval_count` (integer): Number of input tokens in the prompt
- `eval_count` (integer): Number of output tokens generated in the response

### Metrics

`st.metric` displays all JSON fields except response.

## Test Data

- `tests/data/pdf/test_solar_system.pdf` - single-page PDF for testing the conversion and summarization pipeline
