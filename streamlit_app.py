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
    """Split text into token-aware chunks."""
    raise NotImplementedError


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
                max_length=142,
                min_length=56,
                num_beams=4,
                do_sample=False,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

            summaries.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))
            total_prompt_tokens += int(encoded["input_ids"].shape[1])
            total_output_tokens += int(output_ids.shape[1])

    return " ".join(summaries), total_prompt_tokens, total_output_tokens


st.title("Summarization Pipeline")
st.write("Summarize documents with facebook/bart-large-cnn.")

device = get_device()
model, tokenizer = load_model(device)
