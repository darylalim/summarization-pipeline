# Summarization Pipeline

Streamlit web app that converts PDF documents to Markdown using [Docling](https://docling-project.github.io/docling/) and summarizes text using [Falconsai/text_summarization](https://huggingface.co/Falconsai/text_summarization).

## Setup

```bash
python3.12 -m venv streamlit_env
source streamlit_env/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Usage

1. Upload a PDF file
2. Click **Summarize**
3. View the generated summary and metrics
4. Download results as JSON

## Development

```bash
ruff check .        # lint
ruff format .       # format
pyright             # typecheck
pytest              # test
```

Configuration is in `pyproject.toml`.
