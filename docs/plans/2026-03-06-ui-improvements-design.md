# UI Improvements Design

## Context

The current Streamlit app has a minimal UI: a text input, a "Summarize" button, and output showing article metadata, summary text, metrics, and a JSON download. The target users are researchers and practitioners in journalism who need to build up collections of article summaries across a session and export them for analysis.

## Approach

Session state collection using `st.session_state`. Summaries accumulate in a list during the browser session. No external persistence — researchers export their data via JSON or CSV before closing.

## Layout

- **Sidebar** — Generation controls (in a collapsed expander) and export buttons (JSON/CSV).
- **Main area, top** — URL input and "Summarize" button. On submit, the article is processed and appended to the session collection.
- **Main area, below** — The collection. Each article is a card-like block with metadata, side-by-side text, metrics, and action buttons.

## Summary Card Data Model

Each item in the collection list:

```python
{
    "model": str,
    "url": str,
    "title": str,
    "authors": list[str],
    "publish_date": str | None,
    "keywords": list[str],
    "original_text": str,
    "response": str,
    "total_duration": float,
    "chunk_count": int,
    "prompt_eval_count": int,
    "eval_count": int,
    "original_word_count": int,
    "summary_word_count": int,
    "compression_ratio": float,
    "generation_params": {
        "max_length": int,
        "min_length": int,
        "num_beams": int,
        "do_sample": bool,
        "length_penalty": float,
        "early_stopping": bool,
        "no_repeat_ngram_size": int,
    },
}
```

## Card Display

Each card shows:

- `st.subheader` for the title
- `st.columns([1, 1])` for side-by-side original text (left) and summary (right)
- Metrics row: model, duration, token counts, word counts (original vs summary), compression ratio, generation parameters
- Action buttons: up, down (reorder), remove

First card hides up button, last card hides down button. Empty state shows "No articles summarized yet."

## Collection Interaction

- **Adding:** Each successful summarization appends to the end. Duplicate URLs allowed (different generation settings).
- **Removing:** Remove button deletes the item and triggers rerun.
- **Reordering:** Up/down buttons swap with neighbor and rerun.

## Export

Sidebar, below generation controls. Two `st.download_button` widgets disabled when collection is empty.

- **JSON:** Array of card dicts, `indent=2`. Filename: `summaries.json`.
- **CSV:** Flat table via `csv` module. `authors` and `keywords` joined with semicolons. Filename: `summaries.csv`.

Per-article download button removed (superseded by collection export).

## Advanced Generation Controls

Sidebar, in `st.expander("Generation Settings")`, collapsed by default.

| Parameter | Widget | Default | Range |
|---|---|---|---|
| `max_length` | `st.slider` | 130 | 10-512 |
| `min_length` | `st.slider` | 30 | 1-128 |
| `num_beams` | `st.slider` | 4 | 1-10 |
| `do_sample` | `st.checkbox` | False | - |
| `length_penalty` | `st.slider` | 1.0 | 0.0-2.0, step 0.1 |
| `early_stopping` | `st.checkbox` | True | - |
| `no_repeat_ngram_size` | `st.slider` | 3 | 0-5 |

"Reset to Defaults" button restores all values. Parameters are captured per-card at summarization time. Changing settings does not affect already-collected summaries.
