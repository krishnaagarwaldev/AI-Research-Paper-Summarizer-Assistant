from chunking import chunk_text
from embedding import get_embeddings
from vector_store import create_vector_store, search_chunks
from summarizer import generate_summary

text = "This is a test document. " * 300
chunks = chunk_text(text, chunk_size=50, overlap=10)
print(f"Chunks: {len(chunks)}")
embeddings = get_embeddings(chunks)
print(f"Embeddings shape: {embeddings.shape}")
index = create_vector_store(embeddings)
query = "What is this document about?"
retrieved = search_chunks(query, index, chunks, k=2)
print(f"Retrieved: {len(retrieved)}")
try:
    summary = generate_summary(retrieved, "Paragraph", "Short")
    print("Summary:", summary)
except Exception as e:
    print("Error generating summary:", e)
