# MLX Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate from facebook/bart-large-cnn (transformers + torch) to mlx-community/granite-4.0-h-1b-bf16 (mlx-lm) for Apple Silicon native inference.

**Architecture:** Replace the seq2seq summarization pipeline with a decoder-only LLM using chat-template prompting via mlx-lm. Remove PyTorch dependency entirely. Update generation parameters from beam search (`num_beams`, `length_penalty`, etc.) to sampling (`temp`, `top_p`, `repetition_penalty`). Increase chunk size from 1024 to 8192 tokens to leverage the model's 128K context window.

**Tech Stack:** mlx-lm, mlx (transitive), transformers (transitive, tokenizer only), streamlit, newspaper4k

---

## File Map

- **Modify:** `pyproject.toml` — swap dependencies (remove `torch` + `transformers`, add `mlx-lm`)
- **Modify:** `streamlit_app.py` — all app code (imports, constants, functions, UI, main flow)
- **Modify:** `tests/test_streamlit_app.py` — all tests (new mocks, new assertions, remove obsolete tests)
- **Modify:** `CLAUDE.md` — update documentation to reflect MLX changes

---

### Task 1: Update Dependencies

**Files:**
- Modify: `pyproject.toml:1-13`

- [ ] **Step 1: Update pyproject.toml**

Replace `torch` and `transformers` with `mlx-lm`:

```python
[project]
name = "news-article-summarizer"
version = "0.1.0"
description = "Streamlit web app for summarizing news articles using granite-4.0-h-1b"
requires-python = ">=3.12"
dependencies = [
    "lxml_html_clean",
    "mlx-lm",
    "newspaper4k",
    "nltk",
    "streamlit",
]
```

- [ ] **Step 2: Run uv sync**

Run: `uv sync`
Expected: Dependencies resolve successfully. `mlx-lm` installs along with `mlx` and `transformers` as transitive deps.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: replace torch and transformers with mlx-lm"
```

---

### Task 2: Migrate Constants, Chunk, Collection CSV, and Sidebar UI

These components are coupled: `DEFAULT_GENERATION_PARAMS` keys must match the sidebar widgets, the `generation_params` dict, and the `collection_to_csv` fieldnames. They must change together.

**Files:**
- Modify: `streamlit_app.py:50-60` (chunk), `streamlit_app.py:63-100` (collection_to_csv), `streamlit_app.py:103-111` (DEFAULT_GENERATION_PARAMS), `streamlit_app.py:153-234` (sidebar + generation_params dict)
- Modify: `tests/test_streamlit_app.py` (TestDefaultGenerationParams, TestChunk, TestCollectionToCsv)

- [ ] **Step 1: Write updated tests for DEFAULT_GENERATION_PARAMS**

In `tests/test_streamlit_app.py`, update the import to include `MAX_CHUNK_TOKENS`:

```python
from streamlit_app import (
    DEFAULT_GENERATION_PARAMS,
    MAX_CHUNK_TOKENS,
    chunk,
    collection_to_csv,
    extract,
    get_device,
    summarize,
)
```

Replace `TestDefaultGenerationParams`:

```python
class TestDefaultGenerationParams:
    def test_has_expected_keys(self) -> None:
        expected_keys = {
            "max_tokens",
            "temp",
            "top_p",
            "repetition_penalty",
        }
        assert set(DEFAULT_GENERATION_PARAMS.keys()) == expected_keys

    def test_has_expected_values(self) -> None:
        assert DEFAULT_GENERATION_PARAMS["max_tokens"] == 256
        assert DEFAULT_GENERATION_PARAMS["temp"] == 0.0
        assert DEFAULT_GENERATION_PARAMS["top_p"] == 1.0
        assert DEFAULT_GENERATION_PARAMS["repetition_penalty"] == 1.2
```

- [ ] **Step 2: Write updated tests for chunk**

Replace `TestChunk` to use `MAX_CHUNK_TOKENS` instead of hardcoded 1024:

```python
class TestChunk:
    def test_short_text_single_chunk(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(100))

        result = chunk("Short article text.", tokenizer)

        assert result == ["Short article text."]
        tokenizer.encode.assert_called_once_with(
            "Short article text.", add_special_tokens=False
        )

    def test_long_text_splits_into_chunks(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(MAX_CHUNK_TOKENS * 2))
        tokenizer.decode.side_effect = ["chunk one text", "chunk two text"]

        result = chunk("Long article text.", tokenizer)

        assert result == ["chunk one text", "chunk two text"]
        assert tokenizer.decode.call_count == 2
        tokenizer.decode.assert_any_call(
            list(range(MAX_CHUNK_TOKENS)), skip_special_tokens=True
        )
        tokenizer.decode.assert_any_call(
            list(range(MAX_CHUNK_TOKENS, MAX_CHUNK_TOKENS * 2)),
            skip_special_tokens=True,
        )

    def test_exact_max_tokens_single_chunk(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(MAX_CHUNK_TOKENS))

        result = chunk("Exactly max tokens.", tokenizer)

        assert result == ["Exactly max tokens."]
        tokenizer.decode.assert_not_called()

    def test_max_plus_one_tokens_splits(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(MAX_CHUNK_TOKENS + 1))
        tokenizer.decode.side_effect = ["chunk one text", "chunk two text"]

        result = chunk("Just over max tokens.", tokenizer)

        assert result == ["chunk one text", "chunk two text"]
        assert tokenizer.decode.call_count == 2
        tokenizer.decode.assert_any_call(
            list(range(MAX_CHUNK_TOKENS)), skip_special_tokens=True
        )
        tokenizer.decode.assert_any_call(
            [MAX_CHUNK_TOKENS], skip_special_tokens=True
        )

    def test_empty_text_returns_empty(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = []

        result = chunk("", tokenizer)

        assert result == []
```

- [ ] **Step 3: Write updated tests for collection_to_csv**

Replace `test_flattened_generation_params` to assert new param column names:

```python
def test_flattened_generation_params(self) -> None:
    collection = [_make_collection_item()]

    result = collection_to_csv(collection)
    reader = csv.DictReader(io.StringIO(result))
    rows = list(reader)

    assert rows[0]["max_tokens"] == "256"
    assert rows[0]["temp"] == "0.0"
    assert rows[0]["top_p"] == "1.0"
    assert rows[0]["repetition_penalty"] == "1.2"
    assert "generation_params" not in rows[0]
```

No other tests in `TestCollectionToCsv` need changes (they don't assert specific param column names, and `_make_collection_item` uses `dict(DEFAULT_GENERATION_PARAMS)` which auto-updates).

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: `TestDefaultGenerationParams`, `TestChunk`, and `TestCollectionToCsv::test_flattened_generation_params` FAIL (constants and fieldnames haven't changed yet).

- [ ] **Step 5: Update DEFAULT_GENERATION_PARAMS and add MAX_CHUNK_TOKENS**

In `streamlit_app.py`, replace `DEFAULT_GENERATION_PARAMS` (lines 103-111):

```python
DEFAULT_GENERATION_PARAMS: dict[str, int | float] = {
    "max_tokens": 256,
    "temp": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.2,
}
```

Add `MAX_CHUNK_TOKENS` above the `chunk` function (before line 50):

```python
MAX_CHUNK_TOKENS = 8192
```

- [ ] **Step 6: Update chunk function**

In `streamlit_app.py`, replace the `chunk` function (lines 50-60) to use `MAX_CHUNK_TOKENS`:

```python
def chunk(text: str, tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Split text into token-aware chunks of up to MAX_CHUNK_TOKENS tokens."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    if len(token_ids) <= MAX_CHUNK_TOKENS:
        return [text]
    return [
        tokenizer.decode(token_ids[i : i + MAX_CHUNK_TOKENS], skip_special_tokens=True)
        for i in range(0, len(token_ids), MAX_CHUNK_TOKENS)
    ]
```

- [ ] **Step 7: Update collection_to_csv fieldnames**

In `streamlit_app.py`, replace the `fieldnames` list in `collection_to_csv` (lines 67-90):

```python
fieldnames = [
    "model",
    "url",
    "title",
    "authors",
    "publish_date",
    "keywords",
    "original_text",
    "response",
    "total_duration",
    "chunk_count",
    "prompt_eval_count",
    "eval_count",
    "original_word_count",
    "summary_word_count",
    "compression_ratio",
    "max_tokens",
    "temp",
    "top_p",
    "repetition_penalty",
]
```

- [ ] **Step 8: Update sidebar UI and generation_params dict**

In `streamlit_app.py`, replace the sidebar generation settings (lines 154-200) with 4 controls:

```python
with st.sidebar:
    with st.expander("Generation Settings", expanded=False):
        max_tokens = st.slider(
            "max_tokens",
            64,
            1024,
            DEFAULT_GENERATION_PARAMS["max_tokens"],
            key="max_tokens",
        )
        temp = st.slider(
            "temp",
            0.0,
            2.0,
            float(DEFAULT_GENERATION_PARAMS["temp"]),
            step=0.1,
            key="temp",
        )
        top_p = st.slider(
            "top_p",
            0.0,
            1.0,
            float(DEFAULT_GENERATION_PARAMS["top_p"]),
            step=0.05,
            key="top_p",
        )
        repetition_penalty = st.slider(
            "repetition_penalty",
            1.0,
            2.0,
            float(DEFAULT_GENERATION_PARAMS["repetition_penalty"]),
            step=0.1,
            key="repetition_penalty",
        )

        if st.button("Reset to Defaults"):
            for key, value in DEFAULT_GENERATION_PARAMS.items():
                st.session_state[key] = value
            st.rerun()
```

The export section stays unchanged.

Replace the `generation_params` dict (lines 226-234):

```python
generation_params: dict[str, int | float] = {
    "max_tokens": max_tokens,
    "temp": temp,
    "top_p": top_p,
    "repetition_penalty": repetition_penalty,
}
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::TestDefaultGenerationParams tests/test_streamlit_app.py::TestChunk tests/test_streamlit_app.py::TestCollectionToCsv -v`
Expected: All PASS.

- [ ] **Step 10: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "refactor: migrate generation params, chunk size, and sidebar to MLX config"
```

---

### Task 3: Migrate Model Loading, Summarization, and Main Flow

**Files:**
- Modify: `streamlit_app.py:1-16` (imports), `streamlit_app.py:18` (MODEL_NAME), `streamlit_app.py:24-38` (get_device + load_model), `streamlit_app.py:114-147` (summarize), `streamlit_app.py:150-151` (title), `streamlit_app.py:237-264` (main flow)
- Modify: `tests/test_streamlit_app.py` (imports, remove TestGetDevice + _make_encoded, update TestSummarize)

- [ ] **Step 1: Update test imports and remove obsolete test helpers**

In `tests/test_streamlit_app.py`, update the imports:

```python
import csv
import io
from unittest.mock import MagicMock, patch

import pytest

from streamlit_app import (
    DEFAULT_GENERATION_PARAMS,
    MAX_CHUNK_TOKENS,
    SUMMARIZE_PROMPT,
    chunk,
    collection_to_csv,
    extract,
    summarize,
)
```

Remove the `import torch` line (line 6).

Remove the `_make_encoded` helper function entirely (lines 18-28).

Remove `TestGetDevice` entirely (lines 54-67).

- [ ] **Step 2: Write updated TestSummarize**

Replace `TestSummarize` with tests that mock `mlx_lm.generate`:

```python
class TestSummarize:
    @patch("streamlit_app.generate")
    def test_returns_response_and_counts(self, mock_generate: MagicMock) -> None:
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted prompt"
        tokenizer.encode.side_effect = [
            [1, 2, 3, 4, 5],  # prompt tokens
            [1, 2, 3],  # output tokens
        ]
        mock_generate.return_value = "A short summary."
        model = MagicMock()

        response, prompt_eval_count, eval_count = summarize(
            ["Some long document text."], model, tokenizer
        )

        assert response == "A short summary."
        assert prompt_eval_count == 5
        assert eval_count == 3
        tokenizer.apply_chat_template.assert_called_once_with(
            [
                {
                    "role": "user",
                    "content": f"{SUMMARIZE_PROMPT}Some long document text.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        mock_generate.assert_called_once_with(
            model,
            tokenizer,
            prompt="formatted prompt",
            **DEFAULT_GENERATION_PARAMS,
        )

    @patch("streamlit_app.generate")
    def test_multi_chunk_concatenates(self, mock_generate: MagicMock) -> None:
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = ["prompt one", "prompt two"]
        tokenizer.encode.side_effect = [
            [1, 2, 3],  # prompt 1 tokens
            [1, 2],  # output 1 tokens
            [4, 5],  # prompt 2 tokens
            [1, 2, 3],  # output 2 tokens
        ]
        mock_generate.side_effect = ["Summary one.", "Summary two."]
        model = MagicMock()

        response, prompt_eval_count, eval_count = summarize(
            ["Chunk one text.", "Chunk two text."], model, tokenizer
        )

        assert response == "Summary one. Summary two."
        assert prompt_eval_count == 5  # 3 + 2
        assert eval_count == 5  # 2 + 3
        assert mock_generate.call_count == 2
        assert tokenizer.apply_chat_template.call_count == 2

    @patch("streamlit_app.generate")
    def test_custom_generation_params(self, mock_generate: MagicMock) -> None:
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted prompt"
        tokenizer.encode.side_effect = [
            [1, 2, 3],  # prompt tokens
            [1, 2],  # output tokens
        ]
        mock_generate.return_value = "Custom summary."
        model = MagicMock()

        custom_params: dict[str, int | float] = {
            "max_tokens": 512,
            "temp": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.5,
        }

        response, prompt_eval_count, eval_count = summarize(
            ["Some text."], model, tokenizer, custom_params
        )

        assert response == "Custom summary."
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["max_tokens"] == 512
        assert call_kwargs["temp"] == 0.7
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["repetition_penalty"] == 1.5

    @patch("streamlit_app.generate")
    def test_empty_chunks(self, mock_generate: MagicMock) -> None:
        model = MagicMock()
        tokenizer = MagicMock()

        response, prompt_eval_count, eval_count = summarize([], model, tokenizer)

        assert response == ""
        assert prompt_eval_count == 0
        assert eval_count == 0
        tokenizer.apply_chat_template.assert_not_called()
        mock_generate.assert_not_called()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestSummarize -v`
Expected: FAIL (summarize function hasn't been updated yet).

- [ ] **Step 4: Update imports in streamlit_app.py**

Replace the imports (lines 1-16):

```python
import csv
import io
import json
import time
import uuid
from typing import Any

import mlx.nn as nn
import streamlit as st
from mlx_lm import generate, load
from newspaper import Article
from transformers import PreTrainedTokenizerBase
```

- [ ] **Step 5: Update MODEL_NAME and add SUMMARIZE_PROMPT**

Replace `MODEL_NAME` (line 18):

```python
MODEL_NAME = "mlx-community/granite-4.0-h-1b-bf16"
```

Add `SUMMARIZE_PROMPT` above the `summarize` function:

```python
SUMMARIZE_PROMPT = "Summarize the following news article concisely:\n\n"
```

- [ ] **Step 6: Remove get_device and update load_model**

Delete the `get_device` function entirely (lines 24-30).

Replace `load_model` (lines 33-38):

```python
@st.cache_resource
def load_model() -> tuple[nn.Module, PreTrainedTokenizerBase]:
    """Load model and tokenizer at application startup."""
    model, tokenizer = load(MODEL_NAME)
    return model, tokenizer
```

- [ ] **Step 7: Update summarize function**

Replace the `summarize` function (lines 114-147):

```python
def summarize(
    chunks: list[str],
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    generation_params: dict[str, int | float] | None = None,
) -> tuple[str, int, int]:
    """Summarize text chunks and return (response, prompt_eval_count, eval_count)."""
    params = (
        DEFAULT_GENERATION_PARAMS if generation_params is None else generation_params
    )
    summaries: list[str] = []
    total_prompt_tokens = 0
    total_output_tokens = 0

    for text in chunks:
        messages = [{"role": "user", "content": f"{SUMMARIZE_PROMPT}{text}"}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        total_prompt_tokens += len(tokenizer.encode(prompt, add_special_tokens=False))

        response = generate(model, tokenizer, prompt=prompt, **params)

        total_output_tokens += len(
            tokenizer.encode(response, add_special_tokens=False)
        )
        summaries.append(response)

    return " ".join(summaries), total_prompt_tokens, total_output_tokens
```

- [ ] **Step 8: Update main flow**

Replace the subtitle (line 151):

```python
st.write(f"Summarize news articles with {MODEL_NAME}.")
```

Replace the model loading lines (lines 237-238):

```python
model, tokenizer = load_model()
```

Remove the `device` argument from the `summarize` call (lines 262-264):

```python
response, prompt_eval_count, eval_count = summarize(
    chunks, model, tokenizer, generation_params
)
```

- [ ] **Step 9: Run full test suite**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS.

- [ ] **Step 10: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: migrate to mlx-community/granite-4.0-h-1b-bf16 with mlx-lm"
```

---

### Task 4: Update Documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md**

Apply these changes throughout `CLAUDE.md`:

**Line 3 (description):** Replace with:
```
Streamlit web app for summarizing news articles using [granite-4.0-h-1b](https://huggingface.co/ibm-granite/granite-4.0-h-1b) by IBM via [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms). Articles are extracted from URLs using [newspaper4k](https://github.com/AndyTheFactory/newspaper4k). Long articles are split into token-aware chunks before summarization. Summaries accumulate into a session collection that can be reordered, removed, and exported as JSON or CSV.
```

**Dependencies section:** Replace with:
```markdown
## Dependencies

- `newspaper4k` — article extraction from URLs
- `lxml_html_clean` — HTML cleaning (required by newspaper4k)
- `nltk` — NLP features (required by newspaper4k)
- `mlx-lm` — MLX model loading and generation (Apple Silicon)
- `streamlit` — web UI
```

**Model section:** Replace with:
```markdown
### Model

```python
from mlx_lm import generate, load
model, tokenizer = load("mlx-community/granite-4.0-h-1b-bf16")
```
```

**Constants section:** Replace with:
```markdown
### Constants

`DEFAULT_GENERATION_PARAMS` — default generation settings used as sidebar defaults and fallback when no custom params are provided:

```python
DEFAULT_GENERATION_PARAMS: dict[str, int | float] = {
    "max_tokens": 256,
    "temp": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.2,
}
```

`MAX_CHUNK_TOKENS = 8192` — maximum tokens per chunk (model supports 128K context).

`SUMMARIZE_PROMPT` — user message prefix for the chat template.
```

**Functions section:** Replace `get_device` and `load_model` entries:
```markdown
- `load_model() -> tuple[nn.Module, PreTrainedTokenizerBase]` — loads model and tokenizer via mlx-lm, cached with `@st.cache_resource`
```

Remove the `get_device` entry.

Update `chunk` entry:
```markdown
- `chunk(text, tokenizer) -> list[str]` — splits text into token-aware chunks of up to `MAX_CHUNK_TOKENS` (8192) tokens
```

Update `summarize` entry:
```markdown
- `summarize(chunks, model, tokenizer, generation_params) -> tuple[str, int, int]` — summarizes text chunks using chat-template prompting with configurable generation parameters, returns (response, prompt_eval_count, eval_count)
```

**Performance section:** Replace with:
```markdown
### Performance

- `@st.cache_resource` caches the model
- MLX runs natively on Apple Silicon (no device selection needed)
- Token-based chunking splits articles exceeding `MAX_CHUNK_TOKENS` (8192) tokens
- Generation parameters configurable via sidebar; defaults in `DEFAULT_GENERATION_PARAMS`
- Timing: `time.perf_counter()` (fractional seconds)
```

**Sidebar layout entry:** Update to:
```markdown
- **Sidebar** — "Generation Settings" expander (sliders for max_tokens, temp, top_p, repetition_penalty; "Reset to Defaults" button) and "Export" section (JSON and CSV download buttons)
```

**Metrics in Collection Cards:** Update to:
```markdown
- Metrics: Duration (s), Original Words, Summary Words, Compression Ratio, Model, Chunks, Prompt Tokens, Output Tokens
```

**generation_params field description:** Update to:
```markdown
- `generation_params` — generation parameters used: max_tokens, temp, top_p, repetition_penalty (nested in JSON; flattened to columns in CSV)
```

**Tests section:** Replace with:
```markdown
## Tests

`tests/test_streamlit_app.py` — unit tests (mocked, no model download):

- `TestDefaultGenerationParams` — verifies constant keys and values
- `TestExtract` — article download/parse, error propagation
- `TestChunk` — short text, long text, boundaries, empty input (uses `MAX_CHUNK_TOKENS`)
- `TestSummarize` — single/multi-chunk, custom generation params, empty input (mocks `mlx_lm.generate`)
- `TestCollectionToCsv` — single/multi-item, flattened params, empty authors/keywords, empty collection
```

- [ ] **Step 2: Run final verification**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS.

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: No lint or format issues.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for MLX migration"
```
