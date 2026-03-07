import streamlit as st
import faiss
import numpy as np
from typing import List, Tuple
from embedding import get_embedding_model

@st.cache_resource
def create_vector_store(embeddings: np.ndarray):
    """
    Store numpy embeddings in an efficient FAISS vector database.
    
    Args:
        embeddings: A NumPy array containing the vector embeddings.
    
    Returns:
        A fitted FAISS index object.
    """
    if embeddings.size == 0:
        return None
        
    dimension = embeddings.shape[1]
    
    # Use L2 (Euclidean) distance for similarity search.
    # For small/medium documents, IndexFlatL2 provides exact, fast search.
    index = faiss.IndexFlatL2(dimension)
    
    # FAISS requires explicit float32 type
    index.add(embeddings.astype('float32'))
    
    return index

def search_chunks(query: str, index: faiss.IndexFlatL2, chunks: List[str], k: int = 5) -> List[str]:
    """
    Search the FAISS index for the chunks most similar to the query.
    
    Args:
        query:  The search string (e.g., "what are the key findings?").
        index:  The FAISS index built from `create_vector_store`.
        chunks: The original list of text chunks.
        k:      Number of top chunks to retrieve.
        
    Returns:
        A list containing the top-k most relevant text chunk strings.
    """
    if not index or not chunks:
        return []
        
    # 1. Embed the query to match the vector space
    model = get_embedding_model()
    query_embedding = model.encode([query], show_progress_bar=False).astype('float32')
    
    # 2. Search FAISS index
    # distances shapes: (1, k), indices shapes: (1, k)
    distances, indices = index.search(query_embedding, k)
    
    # 3. Retrieve chunks corresponding to the best indices
    retrieved_chunks = []
    for idx in indices[0]:
        if idx < len(chunks): # Fail-safe bounds checking
            retrieved_chunks.append(chunks[idx])
            
    return retrieved_chunks
