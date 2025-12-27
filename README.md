# Summarization Pipeline

A Streamlit application that converts documents to text using Docling and generates summaries with an IBM Granite 4.0 language model.

## Installation

1. Create and activate a Python virtual environment:

```bash
python -m venv streamlit_env
source streamlit_env/bin/activate  # On Windows: streamlit_env\Scripts\activate
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Requirements

- Python 3.12
- See `requirements.txt` for package dependencies

## Notes

- First run will download the model (~few hundred MB)
- Model is cached after first load for faster subsequent runs
- PDF size is limited to 2MB to ensure reasonable processing times
- Summaries are generated with max 200 tokens
