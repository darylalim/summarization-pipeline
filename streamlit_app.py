import json
import tempfile
import time
from pathlib import Path

import streamlit as st
import torch
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.model_downloader import download_models
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

MODEL_NAME = "google/flan-t5-large"
ARTIFACTS_PATH = str(Path.home() / ".cache" / "docling" / "models")


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


@st.cache_resource
def load_doc_converter() -> DocumentConverter:
    """Pre-download Docling models and build the PDF converter."""
    download_models()
    pipeline_options = PdfPipelineOptions(
        artifacts_path=ARTIFACTS_PATH,
        do_table_structure=True,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def convert(source: str, doc_converter: DocumentConverter) -> DoclingDocument:
    """Convert a PDF file to a DoclingDocument."""
    result = doc_converter.convert(
        source=source, max_num_pages=100, max_file_size=20 * 1024 * 1024
    )
    return result.document


def chunk(doc: DoclingDocument, tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Split a DoclingDocument into token-aware text chunks."""
    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=512),
    )
    return [c.text for c in chunker.chunk(dl_doc=doc)]


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

        summaries.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))
        total_prompt_tokens += int(encoded["input_ids"].shape[1])
        total_output_tokens += int(output_ids.shape[1])

    return " ".join(summaries), total_prompt_tokens, total_output_tokens


st.title("Summarization Pipeline")
st.write("Summarize documents with google/flan-t5-large.")

uploaded_file = st.file_uploader("Upload file", type=["pdf"])
device = get_device()
model, tokenizer = load_model(device)
doc_converter = load_doc_converter()

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
