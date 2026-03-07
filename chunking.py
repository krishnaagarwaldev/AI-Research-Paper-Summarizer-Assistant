import streamlit as st
from typing import List

@st.cache_data
def chunk_text(text: str, chunk_size: int = 600, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks based on word count.
    
    Args:
        text: The full extracted text.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of words to overlap between chunks to preserve context.
        
    Returns:
        A list of string chunks.
    """
    words = text.split()
    chunks = []
    
    if not words:
        return chunks
        
    if len(words) <= chunk_size:
        return [text]
        
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        
    return chunks
