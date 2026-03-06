# UI Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the Streamlit news article summarizer into a session-based research collection tool with advanced generation controls, side-by-side text display, reorder/remove, and JSON/CSV export.

**Architecture:** Use `st.session_state` to store a list of summary dicts. Sidebar holds generation controls (in an expander) and export buttons. Main area has the URL input at top and the collection of cards below. Each card shows article metadata, side-by-side original/summary text, metrics, generation params, and action buttons.

**Tech Stack:** Streamlit, Python csv module, existing dependencies (torch, transformers, newspaper4k)

---

### Task 1: Add generation parameters to summarize function signature

**Files:**
- Modify: `streamlit_app.py:56-91`
- Test: `tests/test_streamlit_app.py`

**Step 1: Write the failing test**

Add a new test to `TestSummarize` in `tests/test_streamlit_app.py` that passes custom generation params:

```python
def test_custom_generation_params(self) -> None:
    input_ids = torch.tensor([[1, 2, 3]])
    output_ids = torch.tensor([[10, 11]])

    encoded = _make_encoded(input_ids)

    tokenizer = MagicMock()
    tokenizer.return_value = encoded
    tokenizer.decode.return_value = "Custom summary."

    model = MagicMock()
    model.generate.return_value = output_ids

    generation_params = {
        "max_length": 200,
        "min_length": 50,
        "num_beams": 2,
        "do_sample": True,
        "length_penalty": 0.5,
        "early_stopping": False,
        "no_repeat_ngram_size": 4,
    }

    response, prompt_eval_count, eval_count = summarize(
        ["Some text."], model, tokenizer, "cpu", generation_params
    )

    assert response == "Custom summary."
    generate_kwargs = model.generate.call_args[1]
    assert generate_kwargs["max_length"] == 200
    assert generate_kwargs["min_length"] == 50
    assert generate_kwargs["num_beams"] == 2
    assert generate_kwargs["do_sample"] is True
    assert generate_kwargs["length_penalty"] == 0.5
    assert generate_kwargs["early_stopping"] is False
    assert generate_kwargs["no_repeat_ngram_size"] == 4
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streamlit_app.py::TestSummarize::test_custom_generation_params -v`
Expected: FAIL — `summarize()` does not accept a 5th argument

**Step 3: Write minimal implementation**

Modify `summarize` in `streamlit_app.py` to accept an optional `generation_params` dict. When provided, use it instead of hardcoded values:

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


def summarize(
    chunks: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    generation_params: dict[str, int | float | bool] | None = None,
) -> tuple[str, int, int]:
    """Summarize text chunks and return (response, prompt_eval_count, eval_count)."""
    params = generation_params or DEFAULT_GENERATION_PARAMS
    summaries: list[str] = []
    total_prompt_tokens = 0
    total_output_tokens = 0

    with torch.inference_mode():
        for text in chunks:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(device)

            output_ids = model.generate(  # type: ignore[operator]
                **encoded,
                **params,
            )

            summaries.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))
            total_prompt_tokens += int(encoded["input_ids"].shape[1])
            total_output_tokens += int(output_ids.shape[1])

    return " ".join(summaries), total_prompt_tokens, total_output_tokens
```

**Step 4: Run all tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All tests PASS (existing tests still work because `generation_params` defaults to `None`)

**Step 5: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`

**Step 6: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add optional generation_params to summarize function"
```

---

### Task 2: Add sidebar with generation controls

**Files:**
- Modify: `streamlit_app.py:94-104`

**Step 1: Add sidebar generation controls**

Replace the existing UI section (lines 94-104) with sidebar controls and updated main area top. Add this before the button logic:

```python
st.title("News Article Summarizer")
st.write("Summarize news articles with facebook/bart-large-cnn.")

with st.sidebar:
    st.header("Generation Settings")
    with st.expander("Generation Settings", expanded=False):
        max_length = st.slider("max_length", 10, 512, DEFAULT_GENERATION_PARAMS["max_length"])
        min_length = st.slider("min_length", 1, 128, DEFAULT_GENERATION_PARAMS["min_length"])
        num_beams = st.slider("num_beams", 1, 10, DEFAULT_GENERATION_PARAMS["num_beams"])
        do_sample = st.checkbox("do_sample", value=DEFAULT_GENERATION_PARAMS["do_sample"])
        length_penalty = st.slider(
            "length_penalty", 0.0, 2.0, float(DEFAULT_GENERATION_PARAMS["length_penalty"]), step=0.1
        )
        early_stopping = st.checkbox(
            "early_stopping", value=DEFAULT_GENERATION_PARAMS["early_stopping"]
        )
        no_repeat_ngram_size = st.slider(
            "no_repeat_ngram_size", 0, 5, DEFAULT_GENERATION_PARAMS["no_repeat_ngram_size"]
        )

        if st.button("Reset to Defaults"):
            for key, value in DEFAULT_GENERATION_PARAMS.items():
                st.session_state[key] = value
            st.rerun()

url = st.text_input("Article URL", placeholder="https://example.com/article")
device = get_device()
model, tokenizer = load_model(device)
```

**Step 2: Build the generation_params dict from sidebar values**

Add this right after the sidebar block, before the button:

```python
generation_params: dict[str, int | float | bool] = {
    "max_length": max_length,
    "min_length": min_length,
    "num_beams": num_beams,
    "do_sample": do_sample,
    "length_penalty": length_penalty,
    "early_stopping": early_stopping,
    "no_repeat_ngram_size": no_repeat_ngram_size,
}
```

**Step 3: Run existing tests**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All PASS (UI changes don't affect unit-tested functions)

**Step 4: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`

**Step 5: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add sidebar with generation controls"
```

---

### Task 3: Initialize session state collection and refactor summarization to append

**Files:**
- Modify: `streamlit_app.py`

**Step 1: Initialize session state**

Add after imports, below `MODEL_NAME`:

```python
if "collection" not in st.session_state:
    st.session_state.collection: list[dict] = []
```

**Step 2: Refactor the button handler to append to collection**

Replace the existing button handler block (the `if st.button(...)` through `except`) with logic that:
- Extracts the article
- Chunks and summarizes with `generation_params`
- Computes `original_word_count`, `summary_word_count`, `compression_ratio`
- Builds the card dict (per the data model in the design doc) including `generation_params`, `keywords`, and `original_text`
- Appends to `st.session_state.collection`

```python
if st.button("Summarize", type="primary", disabled=not url):
    if not url.startswith(("http://", "https://")):
        st.warning("Please enter a valid URL starting with http:// or https://.")
        st.stop()

    try:
        with st.spinner("Extracting article..."):
            article = extract(url)

        if not article.text:
            st.warning("No text content could be extracted from the article.")
            st.stop()

        with st.spinner("Chunking article..."):
            chunks = chunk(article.text, tokenizer)

        if not chunks:
            st.warning("No text content could be extracted from the article.")
            st.stop()

        with st.spinner("Summarizing..."):
            start = time.perf_counter()
            response, prompt_eval_count, eval_count = summarize(
                chunks, model, tokenizer, device, generation_params
            )
            total_duration = time.perf_counter() - start

        original_word_count = len(article.text.split())
        summary_word_count = len(response.split())
        compression_ratio = (
            summary_word_count / original_word_count if original_word_count > 0 else 0.0
        )

        summary_data = {
            "model": MODEL_NAME,
            "url": url,
            "title": article.title,
            "authors": article.authors,
            "publish_date": (
                article.publish_date.isoformat() if article.publish_date else None
            ),
            "keywords": article.keywords,
            "original_text": article.text,
            "response": response,
            "total_duration": total_duration,
            "chunk_count": len(chunks),
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "original_word_count": original_word_count,
            "summary_word_count": summary_word_count,
            "compression_ratio": compression_ratio,
            "generation_params": generation_params,
        }

        st.session_state.collection.append(summary_data)
        st.success("Article summarized and added to collection.")

    except Exception as e:
        st.exception(e)
```

**Step 3: Run existing tests**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All PASS

**Step 4: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`

**Step 5: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add session state collection and append on summarize"
```

---

### Task 4: Render collection cards with side-by-side text and metrics

**Files:**
- Modify: `streamlit_app.py`

**Step 1: Add collection rendering below the button handler**

After the button handler block, add the collection display. This replaces all the old inline display code (the `st.success`, `st.subheader("Article")`, `st.subheader("Summary")`, `st.subheader("Metrics")`, and `st.download_button` blocks that were inside the old button handler):

```python
if not st.session_state.collection:
    st.info("No articles summarized yet.")
else:
    for i, item in enumerate(st.session_state.collection):
        st.divider()
        st.subheader(item["title"])

        meta_parts = []
        if item["authors"]:
            meta_parts.append(f"By {', '.join(item['authors'])}")
        if item["publish_date"]:
            meta_parts.append(f"Published: {item['publish_date']}")
        if item["keywords"]:
            meta_parts.append(f"Keywords: {', '.join(item['keywords'])}")
        if meta_parts:
            st.write(" | ".join(meta_parts))

        st.write(f"[{item['url']}]({item['url']})")

        col_original, col_summary = st.columns(2)
        with col_original:
            st.markdown("**Original Text**")
            st.text_area(
                "Original",
                value=item["original_text"],
                height=300,
                disabled=True,
                key=f"original_{i}",
                label_visibility="collapsed",
            )
        with col_summary:
            st.markdown("**Summary**")
            st.text_area(
                "Summary",
                value=item["response"],
                height=300,
                disabled=True,
                key=f"summary_{i}",
                label_visibility="collapsed",
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Duration (s)", f"{item['total_duration']:.4f}")
        m2.metric("Original Words", item["original_word_count"])
        m3.metric("Summary Words", item["summary_word_count"])
        m4.metric("Compression Ratio", f"{item['compression_ratio']:.2%}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Model", item["model"])
        m6.metric("Chunks", item["chunk_count"])
        m7.metric("Prompt Tokens", item["prompt_eval_count"])
        m8.metric("Output Tokens", item["eval_count"])

        with st.expander("Generation Parameters"):
            st.json(item["generation_params"])

        # Action buttons (reorder/remove) added in Task 5
```

**Step 2: Run existing tests**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All PASS

**Step 3: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`

**Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: render collection cards with side-by-side text and metrics"
```

---

### Task 5: Add reorder and remove buttons to cards

**Files:**
- Modify: `streamlit_app.py`

**Step 1: Add action buttons inside the collection rendering loop**

Replace the `# Action buttons` comment from Task 4 with:

```python
        btn_cols = st.columns([1, 1, 1, 7])
        with btn_cols[0]:
            if i > 0 and st.button("Up", key=f"up_{i}"):
                collection = st.session_state.collection
                collection[i - 1], collection[i] = collection[i], collection[i - 1]
                st.rerun()
        with btn_cols[1]:
            if i < len(st.session_state.collection) - 1 and st.button(
                "Down", key=f"down_{i}"
            ):
                collection = st.session_state.collection
                collection[i], collection[i + 1] = collection[i + 1], collection[i]
                st.rerun()
        with btn_cols[2]:
            if st.button("Remove", key=f"remove_{i}"):
                st.session_state.collection.pop(i)
                st.rerun()
```

**Step 2: Run existing tests**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All PASS

**Step 3: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`

**Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add reorder and remove buttons to collection cards"
```

---

### Task 6: Add JSON and CSV export to sidebar

**Files:**
- Modify: `streamlit_app.py` (add `import csv, io` at top, add export section in sidebar)

**Step 1: Write the failing test for CSV export helper**

Add a helper function `collection_to_csv` and test it. In `tests/test_streamlit_app.py`:

```python
import csv
import io

from streamlit_app import collection_to_csv


class TestCollectionToCsv:
    def test_single_item(self) -> None:
        collection = [
            {
                "model": "facebook/bart-large-cnn",
                "url": "https://example.com",
                "title": "Test Article",
                "authors": ["Alice", "Bob"],
                "publish_date": "2026-01-15",
                "keywords": ["news", "test"],
                "original_text": "Original text here.",
                "response": "Summary text here.",
                "total_duration": 1.5,
                "chunk_count": 1,
                "prompt_eval_count": 100,
                "eval_count": 30,
                "original_word_count": 3,
                "summary_word_count": 3,
                "compression_ratio": 1.0,
                "generation_params": {
                    "max_length": 130,
                    "min_length": 30,
                    "num_beams": 4,
                    "do_sample": False,
                    "length_penalty": 1.0,
                    "early_stopping": True,
                    "no_repeat_ngram_size": 3,
                },
            }
        ]

        result = collection_to_csv(collection)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["title"] == "Test Article"
        assert rows[0]["authors"] == "Alice;Bob"
        assert rows[0]["keywords"] == "news;test"
        assert rows[0]["url"] == "https://example.com"

    def test_empty_collection(self) -> None:
        result = collection_to_csv([])
        assert result == ""
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streamlit_app.py::TestCollectionToCsv -v`
Expected: FAIL — `collection_to_csv` not defined

**Step 3: Implement `collection_to_csv` in `streamlit_app.py`**

```python
def collection_to_csv(collection: list[dict[str, object]]) -> str:
    """Convert the summary collection to a CSV string."""
    if not collection:
        return ""
    fieldnames = [
        "model", "url", "title", "authors", "publish_date", "keywords",
        "original_text", "response", "total_duration", "chunk_count",
        "prompt_eval_count", "eval_count", "original_word_count",
        "summary_word_count", "compression_ratio",
        "max_length", "min_length", "num_beams", "do_sample",
        "length_penalty", "early_stopping", "no_repeat_ngram_size",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for item in collection:
        flat = {k: v for k, v in item.items() if k != "generation_params"}
        flat["authors"] = ";".join(flat["authors"])
        flat["keywords"] = ";".join(flat["keywords"])
        flat.update(item["generation_params"])
        writer.writerow(flat)
    return output.getvalue()
```

Add `import csv, io` to the imports at the top of `streamlit_app.py`.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All PASS

**Step 5: Add export buttons to sidebar**

In the sidebar section of `streamlit_app.py`, after the generation controls expander:

```python
    st.header("Export")
    has_items = len(st.session_state.collection) > 0

    st.download_button(
        label="Export JSON",
        data=json.dumps(st.session_state.collection, indent=2),
        file_name="summaries.json",
        mime="application/json",
        disabled=not has_items,
    )

    st.download_button(
        label="Export CSV",
        data=collection_to_csv(st.session_state.collection),
        file_name="summaries.csv",
        mime="text/csv",
        disabled=not has_items,
    )
```

**Step 6: Run all tests**

Run: `uv run pytest tests/test_streamlit_app.py -v`
Expected: All PASS

**Step 7: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`

**Step 8: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add JSON and CSV export to sidebar"
```

---

### Task 7: Update CLAUDE.md and README.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Update CLAUDE.md**

Update the following sections to reflect the new UI:
- Add `csv`, `io` to imports mentioned
- Document the `DEFAULT_GENERATION_PARAMS` constant
- Document the session state collection model
- Document `collection_to_csv` function
- Update the JSON Download section to describe JSON/CSV export of the full collection
- Update the `st.metric` section to describe the new metrics (word counts, compression ratio)
- Update Architecture to mention sidebar layout, generation controls, collection cards

**Step 2: Update README.md**

Update to reflect the new features: session collection, generation controls, side-by-side display, export options.

**Step 3: Lint and format**

Run: `uv run ruff check . && uv run ruff format .`

**Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: update CLAUDE.md and README.md for UI improvements"
```

---

### Task 8: Manual smoke test

**Step 1: Run the app**

Run: `uv run streamlit run streamlit_app.py`

**Step 2: Verify**

- Sidebar shows generation controls in a collapsed expander
- Sidebar shows disabled export buttons when collection is empty
- Submitting a URL appends a card to the collection
- Card shows title, metadata, keywords, side-by-side text, metrics, compression ratio, generation params
- Up/Down/Remove buttons work correctly
- Export JSON and CSV produce valid files with all fields
- Changing generation settings and re-summarizing stores the new params per card

**Step 3: Run full test suite**

Run: `uv run pytest tests/test_streamlit_app.py -v && uv run ruff check . && uv run ruff format --check .`
Expected: All PASS, no lint errors, no format changes
