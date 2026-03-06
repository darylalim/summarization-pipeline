import json
import time

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
                max_length=130,
                min_length=30,
                num_beams=4,
                do_sample=False,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

            summaries.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))
            total_prompt_tokens += int(encoded["input_ids"].shape[1])
            total_output_tokens += int(output_ids.shape[1])

    return " ".join(summaries), total_prompt_tokens, total_output_tokens


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
