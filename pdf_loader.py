import streamlit as st
import fitz  # PyMuPDF

@st.cache_data
def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract text from a Streamlit uploaded PDF file using PyMuPDF.
    """
    try:
        # PyMuPDF requires a stream or byte array
        file_bytes = uploaded_file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        if doc.page_count == 0:
            raise ValueError("The PDF has no pages.")
            
        full_text = []
        for page in doc:
            full_text.append(page.get_text())
            
        text = "\n".join(full_text).strip()
        
        if not text:
            raise ValueError("No readable text found in PDF. It might be an image scan.")
            
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}")
