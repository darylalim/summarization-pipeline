# Design: Switch to flan-t5-large with Docling Chunking

## Context

The app currently uses `facebook/bart-large-cnn` with a 1024-token input limit and no chunking. Multi-page PDFs are truncated. We are switching to `google/flan-t5-large` and adding Docling's `HybridChunker` for token-aware document chunking.

## Model Switch

- `MODEL_NAME`: `"facebook/bart-large-cnn"` -> `"google/flan-t5-large"`
- Same `AutoModelForSeq2SeqLM` / `AutoTokenizer` API
- Tokenizer `max_length`: 1024 -> 512 (flan-t5-large training limit)
- Prepend `"summarize: "` prefix to all input text
- Generation params: add `num_beams=4`, `early_stopping=True`, `no_repeat_ngram_size=3`

## Chunking Pipeline

Replace the current flow (convert PDF -> markdown string -> summarize) with:

1. `DocumentConverter.convert()` -> `DoclingDocument`
2. `HybridChunker(tokenizer=flan_t5_tokenizer, max_tokens=512)` -> chunk the document
3. Summarize each chunk independently
4. Concatenate chunk summaries with `" "` separator

### Function Changes

- `convert(source, doc_converter)` -> returns `DoclingDocument` instead of `str`
- New `chunk(doc, tokenizer)` -> returns `list[str]` of chunk texts
- `summarize(chunks, model, tokenizer, device)` -> accepts `list[str]`, summarizes each, concatenates

## Metrics

- `prompt_eval_count`: total input tokens across all chunks
- `eval_count`: total output tokens across all chunks
- New field: `chunk_count`

## Dependencies

- Add `docling-core[chunking]` to `pyproject.toml`

## Tests

- Update `TestConvert` for new return type (mock `DoclingDocument`)
- Add `TestChunk` for the `chunk()` function
- Update `TestSummarize` for multi-chunk input, `"summarize: "` prefix, new generation params

## What stays the same

- `get_device()`, `load_model()`, `load_doc_converter()` structure
- `@st.cache_resource` caching
- JSON download format (with added `chunk_count`)
- Error handling with `st.exception()`
