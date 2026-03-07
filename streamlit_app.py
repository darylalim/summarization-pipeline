import csv
import io
import json
import time
from typing import Any

import streamlit as st
import torch
from newspaper import Article
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

MODEL_NAME = "facebook/bart-large-cnn"

if "collection" not in st.session_state:
    st.session_state.collection: list[dict] = []


def get_device() -> str:
    """Automatically detect the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@st.cache_resource
def load_model(device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer at application startup."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device).eval()
    return model, tokenizer


def extract(url: str) -> Article:
    """Download and parse a news article from a URL."""
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()
    return article


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
        "max_length",
        "min_length",
        "num_beams",
        "do_sample",
        "length_penalty",
        "early_stopping",
        "no_repeat_ngram_size",
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
    params = (
        DEFAULT_GENERATION_PARAMS if generation_params is None else generation_params
    )
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


st.title("News Article Summarizer")
st.write("Summarize news articles with facebook/bart-large-cnn.")

with st.sidebar:
    with st.expander("Generation Settings", expanded=False):
        max_length = st.slider(
            "max_length",
            10,
            512,
            DEFAULT_GENERATION_PARAMS["max_length"],
            key="max_length",
        )
        min_length = st.slider(
            "min_length",
            1,
            128,
            DEFAULT_GENERATION_PARAMS["min_length"],
            key="min_length",
        )
        num_beams = st.slider(
            "num_beams",
            1,
            10,
            DEFAULT_GENERATION_PARAMS["num_beams"],
            key="num_beams",
        )
        do_sample = st.checkbox(
            "do_sample",
            value=DEFAULT_GENERATION_PARAMS["do_sample"],
            key="do_sample",
        )
        length_penalty = st.slider(
            "length_penalty",
            0.0,
            2.0,
            float(DEFAULT_GENERATION_PARAMS["length_penalty"]),
            step=0.1,
            key="length_penalty",
        )
        early_stopping = st.checkbox(
            "early_stopping",
            value=DEFAULT_GENERATION_PARAMS["early_stopping"],
            key="early_stopping",
        )
        no_repeat_ngram_size = st.slider(
            "no_repeat_ngram_size",
            0,
            5,
            DEFAULT_GENERATION_PARAMS["no_repeat_ngram_size"],
            key="no_repeat_ngram_size",
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

generation_params: dict[str, int | float | bool] = {
    "max_length": max_length,
    "min_length": min_length,
    "num_beams": num_beams,
    "do_sample": do_sample,
    "length_penalty": length_penalty,
    "early_stopping": early_stopping,
    "no_repeat_ngram_size": no_repeat_ngram_size,
}

url = st.text_input("Article URL", placeholder="https://example.com/article")
device = get_device()
model, tokenizer = load_model(device)

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
