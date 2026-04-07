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

MODEL_NAME = "mlx-community/granite-4.0-h-1b-bf16"

if "collection" not in st.session_state:
    st.session_state.collection: list[dict] = []


@st.cache_resource
def load_model() -> tuple[nn.Module, PreTrainedTokenizerBase]:
    """Load model and tokenizer at application startup."""
    model, tokenizer = load(MODEL_NAME)
    return model, tokenizer


def extract(url: str) -> Article:
    """Download and parse a news article from a URL."""
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()
    return article


MAX_CHUNK_TOKENS = 8192


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


def collection_to_csv(collection: list[dict[str, Any]]) -> str:
    """Convert the summary collection to a CSV string."""
    if not collection:
        return ""
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
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for item in collection:
        flat = {k: v for k, v in item.items() if k not in ("_id", "generation_params")}
        flat["authors"] = ";".join(flat["authors"])
        flat["keywords"] = ";".join(flat["keywords"])
        flat.update(item["generation_params"])
        writer.writerow(flat)
    return output.getvalue()


DEFAULT_GENERATION_PARAMS: dict[str, int | float] = {
    "max_tokens": 256,
    "temp": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.2,
}

SUMMARIZE_PROMPT = "Summarize the following news article concisely:\n\n"


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

        total_output_tokens += len(tokenizer.encode(response, add_special_tokens=False))
        summaries.append(response)

    return " ".join(summaries), total_prompt_tokens, total_output_tokens


st.title("News Article Summarizer")
st.write(f"Summarize news articles with {MODEL_NAME}.")

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

    with st.expander("Export", expanded=True):
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

generation_params: dict[str, int | float] = {
    "max_tokens": max_tokens,
    "temp": temp,
    "top_p": top_p,
    "repetition_penalty": repetition_penalty,
}

url = st.text_input("Article URL", placeholder="https://example.com/article")
model, tokenizer = load_model()

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
                chunks, model, tokenizer, generation_params
            )
            total_duration = time.perf_counter() - start

        original_word_count = len(article.text.split())
        summary_word_count = len(response.split())
        compression_ratio = (
            summary_word_count / original_word_count if original_word_count > 0 else 0.0
        )

        summary_data = {
            "_id": str(uuid.uuid4()),
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

if not st.session_state.collection:
    st.info("No articles summarized yet.")
else:
    for i, item in enumerate(st.session_state.collection):
        uid = item["_id"]
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
                key=f"original_{uid}",
                label_visibility="collapsed",
            )
        with col_summary:
            st.markdown("**Summary**")
            st.text_area(
                "Summary",
                value=item["response"],
                height=300,
                disabled=True,
                key=f"summary_{uid}",
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

        btn_cols = st.columns([1, 1, 1, 7])
        with btn_cols[0]:
            if i > 0 and st.button("Up", key=f"up_{uid}"):
                collection = st.session_state.collection
                collection[i - 1], collection[i] = collection[i], collection[i - 1]
                st.rerun()
        with btn_cols[1]:
            if i < len(st.session_state.collection) - 1 and st.button(
                "Down", key=f"down_{uid}"
            ):
                collection = st.session_state.collection
                collection[i], collection[i + 1] = collection[i + 1], collection[i]
                st.rerun()
        with btn_cols[2]:
            if st.button("Remove", key=f"remove_{uid}"):
                st.session_state.collection.pop(i)
                st.rerun()
