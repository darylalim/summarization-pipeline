import streamlit as st
import tempfile
import torch
from docling.document_converter import DocumentConverter
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(
    page_title="Summarization Pipeline",
    layout="centered"
)

st.title("Summarization Pipeline")
st.markdown("Summarize documents with Docling and an IBM Granite 4.0 language model.")

@st.cache_resource
def get_device():
    """Detect and return the best available device"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

@st.cache_resource
def load_model():
    """Load the tokenizer and model"""
    device = get_device()
    model_path = "ibm-granite/granite-4.0-h-tiny"

    with st.spinner(f"Loading model on {device.upper()}..."):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16)
        model = model.to(device)
    
    return tokenizer, model, device

uploaded_file = st.file_uploader(
    "Upload file",
    type=['pdf'],
    help="Maximum file size: 2MB"
)

if uploaded_file is not None:
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    
    if file_size_mb > 2:
        st.error(f"⚠️ File size ({file_size_mb:.2f}MB) exceeds the 2MB limit. Please upload a smaller file.")
    else:
        st.success(f"✅ File uploaded: {uploaded_file.name} ({file_size_mb:.2f}MB)")
        
        # Button to convert document and generate summary
        if st.button("Summarize", type="primary"):
            try:
                # Load model
                tokenizer, model, device = load_model()
                
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Convert PDF to Markdown
                with st.spinner("Converting..."):
                    converter = DocumentConverter()
                    result = converter.convert(tmp_file_path)
                    markdown_text = result.document.export_to_markdown()
                
                # Clean up temp file
                Path(tmp_file_path).unlink()
                
                # Generate summary
                with st.spinner("Summarizing..."):
                    # Create chat template
                    chat = [
                        {
                            "role": "user", 
                            "content": f"Summarize the following document. Your response should only include the answer. Do not provide any further explanation.\n{markdown_text}\nSummary:"
                        }
                    ]
                    
                    # Apply chat template
                    chat_formatted = tokenizer.apply_chat_template(
                        chat, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    
                    # Tokenize
                    input_tokens = tokenizer(chat_formatted, return_tensors="pt").to(device)
                    
                    # Generate summary
                    output = model.generate(
                        **input_tokens,
                        max_new_tokens=200
                    )
                    
                    # Decode output
                    output_text = tokenizer.batch_decode(output)[0]
                    
                    # Extract summary from output
                    summary = output_text.split("Summary:")[-1].strip()
                    
                    # Clean up special tokens
                    summary = summary.replace("<|end_of_text|>", "")
                    summary = summary.replace("<|endoftext|>", "")
                    summary = summary.replace("<|start_of_role|>assistant<|end_of_role|>", "")
                    summary = summary.replace("<|start_of_role|>", "")
                    summary = summary.replace("<|end_of_role|>", "")
                    summary = summary.strip()
                
                # Store summary in session state
                st.session_state.summary = summary
                st.session_state.original_filename = uploaded_file.name
                
                st.success("✅ Done")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)

# Display summary if available
if 'summary' in st.session_state:
    st.subheader("Summary")
    st.markdown(st.session_state.summary)
      
    # Generate filename for download
    original_name = Path(st.session_state.original_filename).stem
    download_filename = f"{original_name}_summary.md"
    
    st.download_button(
        label="Download",
        data=st.session_state.summary,
        file_name=download_filename,
        mime="text/markdown",
        type="secondary"
    )
