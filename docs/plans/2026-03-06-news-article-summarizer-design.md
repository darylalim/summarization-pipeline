# News Article Summarizer Design

## Overview

Refactor the summarization pipeline from a general-purpose PDF summarizer into a focused news article summarizer. Replace PDF upload + Docling with URL input + newspaper4k. Tune generation parameters for news-length content.

## Input & Extraction

- Remove PDF upload, Docling dependency, and all PDF-related code (`load_doc_converter`, `convert`, `chunk`)
- Add `st.text_input` for article URL
- Add `newspaper4k` and `lxml_html_clean` dependencies
- New `extract(url: str) -> Article` function using newspaper's `Article` class
  - Calls `download()`, `parse()`, `nlp()`
  - Returns article object with `.title`, `.authors`, `.publish_date`, `.text`

## Chunking

- Most news articles fit within 1024 tokens (bart-large-cnn max input)
- For longer articles: tokenize text, split into chunks at 1024-token boundaries
- Simple token-count splitting replaces Docling's `HybridChunker`

## Summarization

- Keep bart-large-cnn (trained on CNN/DailyMail news data)
- Keep: `@st.cache_resource`, `torch.inference_mode()`, device detection, `model.eval()`
- Keep: `num_beams=4`, `do_sample=False`, `no_repeat_ngram_size=3`
- Change `max_length`: 142 -> 130
- Change `min_length`: 56 -> 30
- Change `length_penalty`: 2.0 -> 1.0

## UI

- Title: "News Article Summarizer"
- URL text input replaces file uploader
- Display article metadata after extraction: title, authors, publish date
- Spinner steps: "Extracting article..." -> "Summarizing..."
- Remove chunk count from metrics display (keep in JSON)

## JSON Download

Fields: `model`, `response`, `total_duration`, `chunk_count`, `prompt_eval_count`, `eval_count`, `url`, `title`, `authors`, `publish_date`

## Dependencies

- Remove: `docling`
- Add: `newspaper4k`, `lxml_html_clean`

## Tests

- Remove: `TestConvert`, `TestChunk` (old), `tests/data/pdf/` directory
- Add: `TestExtract` — mock `newspaper.Article`
- Add: `TestChunk` (new) — token-based chunking logic
- Update: `TestSummarize` — verify updated generation params
- Keep: `TestGetDevice`

## Docs

- Update `CLAUDE.md` and `README.md` to reflect new scope
