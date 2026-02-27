# Flan-T5 + Docling Chunking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Switch the summarization model from facebook/bart-large-cnn to google/flan-t5-large and add Docling HybridChunker so long documents are chunked before summarization.

**Architecture:** PDF is converted to a DoclingDocument via Docling, then split into token-aware chunks using HybridChunker configured with the flan-t5-large tokenizer (max 512 tokens). Each chunk is summarized independently with a `"summarize: "` prefix, and the results are concatenated.

**Tech Stack:** transformers (AutoModelForSeq2SeqLM, AutoTokenizer), docling (DocumentConverter, HybridChunker), torch, streamlit

---

### Task 1: Update `convert()` to return DoclingDocument

**Files:**
- Modify: `streamlit_app.py:8-10,55-60`
- Modify: `tests/test_streamlit_app.py:5,24-38`

**Step 1: Update the test**

In `tests/test_streamlit_app.py`, update the import and `TestConvert`:

```python
from streamlit_app import chunk, convert, get_device, summarize
```

```python
class TestConvert:
    def test_returns_docling_document(self) -> None:
        mock_document = MagicMock()
        doc_converter = MagicMock()
        doc_converter.convert.return_value.document = mock_document

        result = convert("/tmp/test.pdf", doc_converter)

        assert result is mock_document
        doc_converter.convert.assert_called_once_with(
            source="/tmp/test.pdf",
            max_num_pages=100,
            max_file_size=20 * 1024 * 1024,
        )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streamlit_app.py::TestConvert -v`
Expected: FAIL — `convert` still returns a string

**Step 3: Update `convert()` in `streamlit_app.py`**

Add import:
```python
from docling.datamodel.document import DoclingDocument
```

Change the function:
```python
def convert(source: str, doc_converter: DocumentConverter) -> DoclingDocument:
    """Convert a PDF file to a DoclingDocument."""
    result = doc_converter.convert(
        source=source, max_num_pages=100, max_file_size=20 * 1024 * 1024
    )
    return result.document
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_streamlit_app.py::TestConvert -v`
Expected: PASS

**Step 5: Commit**

```
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "refactor: convert() returns DoclingDocument instead of markdown string"
```

---

### Task 2: Add `chunk()` function

**Files:**
- Modify: `streamlit_app.py` (add new imports and `chunk` function after `convert`)
- Modify: `tests/test_streamlit_app.py` (add `TestChunk` class)

**Step 1: Write the test**

Add to `tests/test_streamlit_app.py`:

```python
class TestChunk:
    def test_returns_chunk_texts(self) -> None:
        chunk_1 = MagicMock()
        chunk_1.text = "First section content."
        chunk_2 = MagicMock()
        chunk_2.text = "Second section content."

        doc = MagicMock()
        tokenizer = MagicMock()

        with patch("streamlit_app.HybridChunker") as mock_chunker_cls:
            mock_chunker = MagicMock()
            mock_chunker.chunk.return_value = iter([chunk_1, chunk_2])
            mock_chunker_cls.return_value = mock_chunker

            result = chunk(doc, tokenizer)

        assert result == ["First section content.", "Second section content."]
        mock_chunker.chunk.assert_called_once_with(dl_doc=doc)

    def test_single_chunk(self) -> None:
        chunk_1 = MagicMock()
        chunk_1.text = "Only section."

        doc = MagicMock()
        tokenizer = MagicMock()

        with patch("streamlit_app.HybridChunker") as mock_chunker_cls:
            mock_chunker = MagicMock()
            mock_chunker.chunk.return_value = iter([chunk_1])
            mock_chunker_cls.return_value = mock_chunker

            result = chunk(doc, tokenizer)

        assert result == ["Only section."]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streamlit_app.py::TestChunk -v`
Expected: FAIL — `chunk` is not defined or ImportError

**Step 3: Implement `chunk()` in `streamlit_app.py`**

Add import:
```python
from docling.chunking import HybridChunker
```

Add function after `convert`:
```python
def chunk(doc: DoclingDocument, tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Split a DoclingDocument into token-aware text chunks."""
    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=512),
    )
    return [c.text for c in chunker.chunk(dl_doc=doc)]
```

Also add the `HuggingFaceTokenizer` import:
```python
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_streamlit_app.py::TestChunk -v`
Expected: PASS

**Step 5: Commit**

```
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: add chunk() using Docling HybridChunker"
```

---

### Task 3: Switch model and update `summarize()` for multi-chunk input

**Files:**
- Modify: `streamlit_app.py:19,63-83,87`
- Modify: `tests/test_streamlit_app.py:41-81`

**Step 1: Update the test**

Replace `TestSummarize` in `tests/test_streamlit_app.py`:

```python
class TestSummarize:
    def test_returns_response_and_counts(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones_like(input_ids)
        output_ids = torch.tensor([[1, 2, 3]])

        encoded = MagicMock()
        encoded.__getitem__ = lambda self, key: {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }[key]
        encoded.keys.return_value = ["input_ids", "attention_mask"]
        encoded.__iter__ = lambda self: iter(["input_ids", "attention_mask"])
        encoded.to.return_value = encoded

        tokenizer = MagicMock()
        tokenizer.return_value = encoded
        tokenizer.decode.return_value = "A short summary."

        model = MagicMock()
        model.generate.return_value = output_ids

        response, prompt_eval_count, eval_count = summarize(
            ["Some long document text."], model, tokenizer, "cpu"
        )

        assert response == "A short summary."
        assert prompt_eval_count == 5
        assert eval_count == 3
        tokenizer.assert_called_once_with(
            "summarize: Some long document text.",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded.to.assert_called_once_with("cpu")
        tokenizer.decode.assert_called_once()
        decode_args, decode_kwargs = tokenizer.decode.call_args
        assert torch.equal(decode_args[0], output_ids[0])
        assert decode_kwargs == {"skip_special_tokens": True}

    def test_multi_chunk_concatenates(self) -> None:
        input_ids_1 = torch.tensor([[1, 2, 3]])
        input_ids_2 = torch.tensor([[4, 5]])
        attention_mask_1 = torch.ones_like(input_ids_1)
        attention_mask_2 = torch.ones_like(input_ids_2)
        output_ids_1 = torch.tensor([[10, 11]])
        output_ids_2 = torch.tensor([[12, 13, 14]])

        encoded_1 = MagicMock()
        encoded_1.__getitem__ = lambda self, key: {
            "input_ids": input_ids_1,
            "attention_mask": attention_mask_1,
        }[key]
        encoded_1.keys.return_value = ["input_ids", "attention_mask"]
        encoded_1.__iter__ = lambda self: iter(["input_ids", "attention_mask"])
        encoded_1.to.return_value = encoded_1

        encoded_2 = MagicMock()
        encoded_2.__getitem__ = lambda self, key: {
            "input_ids": input_ids_2,
            "attention_mask": attention_mask_2,
        }[key]
        encoded_2.keys.return_value = ["input_ids", "attention_mask"]
        encoded_2.__iter__ = lambda self: iter(["input_ids", "attention_mask"])
        encoded_2.to.return_value = encoded_2

        tokenizer = MagicMock()
        tokenizer.side_effect = [encoded_1, encoded_2]
        tokenizer.decode.side_effect = ["Summary one.", "Summary two."]

        model = MagicMock()
        model.generate.side_effect = [output_ids_1, output_ids_2]

        response, prompt_eval_count, eval_count = summarize(
            ["Chunk one text.", "Chunk two text."], model, tokenizer, "cpu"
        )

        assert response == "Summary one. Summary two."
        assert prompt_eval_count == 5  # 3 + 2
        assert eval_count == 5  # 2 + 3
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streamlit_app.py::TestSummarize -v`
Expected: FAIL — `summarize` still takes a `str`, not `list[str]`

**Step 3: Update `MODEL_NAME`, `summarize()`, and subtitle in `streamlit_app.py`**

Change the constant:
```python
MODEL_NAME = "google/flan-t5-large"
```

Replace `summarize`:
```python
def summarize(
    chunks: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
) -> tuple[str, int, int]:
    """Summarize text chunks and return (response, prompt_eval_count, eval_count)."""
    summaries: list[str] = []
    total_prompt_tokens = 0
    total_output_tokens = 0

    for text in chunks:
        encoded = tokenizer(
            f"summarize: {text}",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.inference_mode():
            output_ids = model.generate(  # type: ignore[operator]
                **encoded,
                max_length=150,
                min_length=30,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        summaries.append(
            tokenizer.decode(output_ids[0], skip_special_tokens=True)
        )
        total_prompt_tokens += int(encoded["input_ids"].shape[1])
        total_output_tokens += int(output_ids.shape[1])

    return " ".join(summaries), total_prompt_tokens, total_output_tokens
```

Update the subtitle:
```python
st.write("Summarize documents with google/flan-t5-large.")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_streamlit_app.py::TestSummarize -v`
Expected: PASS

**Step 5: Commit**

```
git add streamlit_app.py tests/test_streamlit_app.py
git commit -m "feat: switch to flan-t5-large with multi-chunk summarization"
```

---

### Task 4: Wire up the UI with chunking and new metrics

**Files:**
- Modify: `streamlit_app.py:94-137` (the UI section)

**Step 1: Update the UI section in `streamlit_app.py`**

Replace the `if st.button(...)` block (lines 94-137):

```python
if st.button("Summarize", type="primary", disabled=uploaded_file is None):
    if uploaded_file is not None:
        try:
            with st.spinner("Converting document..."):
                with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file.flush()
                    doc = convert(tmp_file.name, doc_converter)

            with st.spinner("Chunking document..."):
                chunks = chunk(doc, tokenizer)

            with st.spinner("Summarizing..."):
                start = time.perf_counter()
                response, prompt_eval_count, eval_count = summarize(
                    chunks, model, tokenizer, device
                )
                total_duration = time.perf_counter() - start

            st.success("Done.")

            st.subheader("Summary")
            st.write(response)

            st.subheader("Metrics")
            st.metric("Model", MODEL_NAME)
            st.metric("Total Duration (seconds)", f"{total_duration:.4f}")
            st.metric("Chunk Count", len(chunks))
            st.metric("Prompt Eval Count", prompt_eval_count)
            st.metric("Eval Count", eval_count)

            summary_data = {
                "model": MODEL_NAME,
                "response": response,
                "total_duration": total_duration,
                "chunk_count": len(chunks),
                "prompt_eval_count": prompt_eval_count,
                "eval_count": eval_count,
            }

            st.download_button(
                label="Download JSON",
                data=json.dumps(summary_data, indent=2),
                file_name=f"{uploaded_file.name}_summary.json",
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

```
git add streamlit_app.py
git commit -m "feat: wire up chunking pipeline and chunk_count metric in UI"
```

---

### Task 5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

Update these sections:
- Description: change "bart-large-cnn" to "flan-t5-large"
- Model section: update `from_pretrained` model name
- Dependencies: add `docling` chunking note
- Performance: update tokenizer `max_length` to 512, update generation params, add chunking info
- JSON Download: add `chunk_count` field
- Tests: add `chunk` to the tested functions list

**Step 2: Commit**

```
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for flan-t5-large and chunking"
```
