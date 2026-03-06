# News Article Summarizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the PDF summarization app into a news article summarizer that takes URLs, extracts articles with newspaper4k, and generates summaries with tuned generation parameters.

**Architecture:** Replace the Docling PDF pipeline with newspaper4k for URL-based article extraction. Add token-based chunking for long articles. Tune bart-large-cnn generation params for news-length content. Single-file Streamlit app structure stays the same.

**Tech Stack:** Python 3.12, Streamlit, newspaper4k, transformers, torch, uv

---

### Task 1: Update dependencies

**Files:**
- Modify: `pyproject.toml:1-11`

**Step 1: Update pyproject.toml**

Replace `docling` with `newspaper4k` and `lxml_html_clean` in dependencies. Update the project description.

```toml
[project]
name = "summarization-pipeline"
version = "0.1.0"
description = "Streamlit web app for summarizing news articles using bart-large-cnn"
requires-python = ">=3.12"
dependencies = [
    "lxml_html_clean",
    "newspaper4k",
    "streamlit",
    "torch",
    "transformers",
]
```

**Step 2: Sync dependencies**

Run: `uv sync`
Expected: Resolves successfully, installs newspaper4k and lxml_html_clean, removes docling

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: replace docling with newspaper4k and lxml_html_clean"
```

---

### Task 2: Implement extract function (TDD)

**Files:**
- Modify: `streamlit_app.py`
- Modify: `tests/test_streamlit_app.py`

**Step 1: Write the failing test**

Add `TestExtract` to `tests/test_streamlit_app.py`. This replaces `TestConvert`. Remove the old `TestConvert` and `TestChunk` classes, and update the import line.

Replace the import line at the top:
```python
from streamlit_app import chunk, convert, get_device, summarize
```
with:
```python
from streamlit_app import chunk, extract, get_device, summarize
```

Remove `TestConvert` (lines 38-51) and `TestChunk` (lines 54-85). Add:

```python
class TestExtract:
    @patch("streamlit_app.Article")
    def test_returns_article(self, mock_article_cls: MagicMock) -> None:
        mock_article = MagicMock()
        mock_article.title = "Breaking News"
        mock_article.authors = ["Jane Doe"]
        mock_article.publish_date = "2026-01-15"
        mock_article.text = "Article body text."
        mock_article_cls.return_value = mock_article

        result = extract("https://example.com/article")

        mock_article_cls.assert_called_once_with("https://example.com/article")
        mock_article.download.assert_called_once()
        mock_article.parse.assert_called_once()
        mock_article.nlp.assert_called_once()
        assert result is mock_article
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streamlit_app.py::TestExtract -v`
Expected: FAIL — `ImportError: cannot import name 'extract'`

**Step 3: Write minimal implementation**

In `streamlit_app.py`, replace the docling imports (lines 7-14) and add the newspaper import. Remove `ARTIFACTS_PATH`. Remove `load_doc_converter`, `convert`, and the old `chunk` function. Add `extract`.

Remove these imports:
```python
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.model_downloader import download_models
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
```

Remove `ARTIFACTS_PATH = str(Path.home() / ".cache" / "docling" / "models")` and the `from pathlib import Path` import (no longer needed). Also remove `import tempfile` (no longer needed).

Add this import:
```python
from newspaper import Article
```

Remove the `load_doc_converter` function (lines 43-55), the `convert` function (lines 58-63), and the old `chunk` function (lines 66-71).

Add the `extract` function after `load_model`:
```python
def extract(url: str) -> Article:
    """Download and parse a news article from a URL."""
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()
    return article
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_streamlit_app.py::TestExtract -v`
Expected: PASS

**Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add extract function using newspaper4k, remove docling code"
```

---

### Task 3: Implement token-based chunk function (TDD)

**Files:**
- Modify: `streamlit_app.py`
- Modify: `tests/test_streamlit_app.py`

**Step 1: Write the failing tests**

Add a new `TestChunk` class to `tests/test_streamlit_app.py`:

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
        tokenizer.encode.return_value = list(range(2048))
        tokenizer.decode.side_effect = ["chunk one text", "chunk two text"]

        result = chunk("Long article text.", tokenizer)

        assert result == ["chunk one text", "chunk two text"]
        assert tokenizer.decode.call_count == 2
        tokenizer.decode.assert_any_call(
            list(range(1024)), skip_special_tokens=True
        )
        tokenizer.decode.assert_any_call(
            list(range(1024, 2048)), skip_special_tokens=True
        )

    def test_empty_text_returns_empty(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = []

        result = chunk("", tokenizer)

        assert result == []
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestChunk -v`
Expected: FAIL — `chunk` signature mismatch (old function removed, new one not yet added)

**Step 3: Write minimal implementation**

Add the new `chunk` function to `streamlit_app.py` after `extract`:

```python
def chunk(text: str, tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Split text into token-aware chunks of up to 1024 tokens."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    if len(token_ids) <= 1024:
        return [text]
    return [
        tokenizer.decode(token_ids[i : i + 1024], skip_special_tokens=True)
        for i in range(0, len(token_ids), 1024)
    ]
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::TestChunk -v`
Expected: PASS

**Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add token-based chunk function for news articles"
```

---

### Task 4: Update summarize generation parameters (TDD)

**Files:**
- Modify: `streamlit_app.py:74-103` (the `summarize` function)
- Modify: `tests/test_streamlit_app.py`

**Step 1: Update the tests**

In `TestSummarize.test_returns_response_and_counts`, the test currently doesn't assert on generation params directly (it uses a MagicMock model). Add an assertion on the `model.generate` call kwargs. Update in both `test_returns_response_and_counts` and `test_multi_chunk_concatenates`.

In `test_returns_response_and_counts`, add after the existing assertions:

```python
        generate_kwargs = model.generate.call_args[1]
        assert generate_kwargs["max_length"] == 130
        assert generate_kwargs["min_length"] == 30
        assert generate_kwargs["length_penalty"] == 1.0
```

In `test_multi_chunk_concatenates`, add after the existing assertions:

```python
        generate_kwargs = model.generate.call_args[1]
        assert generate_kwargs["max_length"] == 130
        assert generate_kwargs["min_length"] == 30
        assert generate_kwargs["length_penalty"] == 1.0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streamlit_app.py::TestSummarize -v`
Expected: FAIL — `assert 142 == 130` (old max_length value)

**Step 3: Update the summarize function**

In `streamlit_app.py`, update the `model.generate` call inside `summarize`:

```python
            output_ids = model.generate(  # type: ignore[operator]
                **encoded,
                max_length=130,
                min_length=30,
                num_beams=4,
                do_sample=False,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py::TestSummarize -v`
Expected: PASS

**Step 5: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: tune generation params for news article summaries"
```

---

### Task 5: Update Streamlit UI

**Files:**
- Modify: `streamlit_app.py:112-173` (the UI section at bottom of file)

**Step 1: Rewrite the UI section**

Replace everything from `st.title(...)` (line 112) to the end of the file with:

```python
st.title("News Article Summarizer")
st.write("Summarize news articles with facebook/bart-large-cnn.")

url = st.text_input("Article URL", placeholder="https://example.com/article")
device = get_device()
model, tokenizer = load_model(device)

if st.button("Summarize", type="primary", disabled=not url):
    try:
        with st.spinner("Extracting article..."):
            article = extract(url)

        if not article.text:
            st.warning("No text content could be extracted from the article.")
            st.stop()

        with st.spinner("Summarizing..."):
            chunks = chunk(article.text, tokenizer)
            start = time.perf_counter()
            response, prompt_eval_count, eval_count = summarize(
                chunks, model, tokenizer, device
            )
            total_duration = time.perf_counter() - start

        st.success("Done.")

        st.subheader("Article")
        st.markdown(f"**{article.title}**")
        if article.authors:
            st.write(f"By {', '.join(article.authors)}")
        if article.publish_date:
            st.write(f"Published: {article.publish_date.strftime('%Y-%m-%d')}")

        st.subheader("Summary")
        st.write(response)

        st.subheader("Metrics")
        st.metric("Model", MODEL_NAME)
        st.metric("Total Duration (seconds)", f"{total_duration:.4f}")
        st.metric("Prompt Eval Count", prompt_eval_count)
        st.metric("Eval Count", eval_count)

        summary_data = {
            "model": MODEL_NAME,
            "url": url,
            "title": article.title,
            "authors": article.authors,
            "publish_date": (
                article.publish_date.isoformat() if article.publish_date else None
            ),
            "response": response,
            "total_duration": total_duration,
            "chunk_count": len(chunks),
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
        }

        st.download_button(
            label="Download JSON",
            data=json.dumps(summary_data, indent=2),
            file_name="summary.json",
            mime="application/json",
        )

    except Exception as e:
        st.exception(e)
```

**Step 2: Run all tests**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 3: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`
Expected: No errors

**Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: update UI for news article URL input and metadata display"
```

---

### Task 6: Remove test data and update docs

**Files:**
- Delete: `tests/data/` directory
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Remove test PDF data**

```bash
rm -rf tests/data/
```

**Step 2: Update README.md**

Replace the full contents of `README.md`:

```markdown
# News Article Summarizer

Streamlit web app for summarizing news articles using [bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) by Facebook. Articles are extracted from URLs using [newspaper4k](https://github.com/AndyTheFactory/newspaper4k). Long articles are split into token-aware chunks before summarization.

## Setup

\`\`\`bash
uv sync
uv run streamlit run streamlit_app.py
\`\`\`

## Usage

1. Paste a news article URL
2. Click **Summarize**
3. View article metadata, generated summary, and metrics
4. Download results as JSON

## Development

\`\`\`bash
uv run ruff check .        # lint
uv run ruff format .       # format
uv run ty check            # typecheck
uv run pytest              # test
\`\`\`

Configuration is in `pyproject.toml`.
```

**Step 3: Update CLAUDE.md**

Replace the full contents of `CLAUDE.md`:

```markdown
# CLAUDE.md

Streamlit web app for summarizing news articles using [bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) by Facebook. Articles are extracted from URLs using [newspaper4k](https://github.com/AndyTheFactory/newspaper4k). Long articles are split into token-aware chunks before summarization.

## Setup

\`\`\`bash
uv sync
uv run streamlit run streamlit_app.py
\`\`\`

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
- `transformers` — Hugging Face model loading and generation
- `torch` — tensor operations
- `streamlit` — web UI

## Configuration

`pyproject.toml` — ruff lint isort (`combine-as-imports`), pytest (`pythonpath`), ty (`python-version = "3.12"`)

## Architecture

`streamlit_app.py` — single-file app

### Model

\`\`\`python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
\`\`\`

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
```

**Step 4: Run all tests and lint**

Run: `uv run pytest -v && uv run ruff check . && uv run ruff format .`
Expected: All pass, no lint errors

**Step 5: Commit**

```bash
git add -A
git commit -m "docs: update CLAUDE.md and README.md, remove test PDF data"
```

---

### Task 7: Final verification

**Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 2: Run lint**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: Clean

**Step 3: Run typecheck**

Run: `uv run ty check`
Expected: No errors (or only pre-existing ones from dependencies)
