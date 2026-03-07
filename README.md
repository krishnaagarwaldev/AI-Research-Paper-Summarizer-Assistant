# 📚 Research Paper Companion

> **Transform complex research papers into clear, actionable insights instantly.**
> A high-performance RAG (Retrieval-Augmented Generation) assistant that summarizes papers, extracts figures, and chats with documents.

---

## ✨ Key Features

- 📝 **Smart Summarization**: Get executive summaries tailored by length, tone, and academic style.
- 🤖 **Llama 3 8B Brilliance**: Powered by Meta's high-capacity **Meta-Llama-3-8B-Instruct** model for superior technical accuracy.
- ⚡ **Real-time Streaming**: Watch your summaries and chat answers appear instantly, word-by-word.
- 📊 **Advanced RAG System**: Uses **FAISS** and **Sentence Transformers** to retrieve exact context from even the largest PDFs.
- 🖼️ **Automated Figure Extraction**: Mines charts and diagrams from PDFs with automatic **AI Captioning**.
- 💬 **Technical Chat**: Ask deep questions, explain formulas, and break down methodologies in a persistent chat interface.
- 📐 **Professional LaTeX**: Native rendering of mathematical equations and recurrence relations.

---


## 🗂️ Project Structure

| Component              | Responsibility                              |
| :--------------------- | :------------------------------------------ |
| `app.py`             | Main entry point & Streamlit UI logic.      |
| `summarizer.py`      | LLM integration (Llama 3 8B) & Streaming.   |
| `image_extractor.py` | Figure extraction & Caption detection.      |
| `embedding.py`       | Dense vector generation (all-MiniLM-L6-v2). |
| `vector_store.py`    | Similarity search via FAISS.                |
| `pdf_loader.py`      | Reliable document text extraction.          |
