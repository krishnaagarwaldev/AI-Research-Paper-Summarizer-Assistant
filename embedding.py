# embedding.py
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

# Cache the model so we don't reload it from disk on every Streamlit rerun
@st.cache_resource
def get_embedding_model():
    # all-MiniLM-L6-v2 is fast, lightweight, and perfect for CPU retrieval
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def get_embeddings(chunks: List[str]) -> np.ndarray:
    """
    Generate dense numerical vectors for a list of text chunks.
    """
    if not chunks:
        return np.array([])
        
    model = get_embedding_model()
    # encode() returns a numpy array of shape (len(chunks), 384)
    embeddings = model.encode(chunks, show_progress_bar=False)
    return embeddings
