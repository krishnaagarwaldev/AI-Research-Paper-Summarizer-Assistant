import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from typing import List
import streamlit as st  # Only used for secrets management in this file

# Load environment variables
load_dotenv()

# We continue using the free serverless Inference API.
# This requires HF_TOKEN in the .env file.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

def _get_client() -> InferenceClient:
    token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN is missing. Please create a .env file and add:\n"
            "HF_TOKEN=your_huggingface_token"
        )
    return InferenceClient(token=token)

def generate_summary(retrieved_chunks: List[str], style: str, length: str, tone: str):
    """
    Build a RAG prompt using the retrieved context and stream it from the LLaMA model.
    """
    context_text = "\n\n---\n\n".join(retrieved_chunks)
    
    prompt = (
        f"You are a professional research assistant.\n\n"
        f"Goal: Provide a highly detailed and comprehensive summary of the following research paper content.\n\n"
        f"Summary length: {length}\n"
        f"Style: {style}\n"
        f"Tone: {tone}\n\n"
        f"Requirements:\n"
        f"1. Summarize key ideas, methodology, contributions, and conclusions in depth.\n"
        f"2. Seamlessly blend in extra insights: Connect the paper's findings to broader trends in the field or general industry knowledge without using labels like 'From my side'.\n"
        f"3. Concrete Examples: Integrate real-world examples or analogies naturally into the text to explain core concepts.\n"
        f"4. Math: Wrap equations in double dollar signs ($$ formula $$). DO NOT use code blocks for math.\n\n"
        f"Context:\n{context_text}\n"
    )
    
    client = _get_client()
    
    max_tokens_map = {
        "Short": 800,
        "Medium": 1500,
        "Long": 3000
    }
    target_tokens = max_tokens_map.get(length, 1500)
    
    try:
        for message in client.chat_completion(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=target_tokens,
            temperature=0.4,
            top_p=0.9,
            stream=True
        ):
            if hasattr(message, "choices") and len(message.choices) > 0:
                token = message.choices[0].delta.content
                if token:
                    yield token
        
    except Exception as e:
        error_str = str(e)
        if "401" in error_str:
            yield "❌ **401 Unauthorized:** Your Hugging Face token is invalid."
        elif "403" in error_str:
            yield "❌ **403 Forbidden:** You must accept the model license on HuggingFace first."
        else:
            yield f"❌ **API Error:** {error_str}"

def answer_question(query: str, retrieved_chunks: List[str]):
    """
    Answer a user question based on the document while adding detail, examples, and external knowledge.
    Ensures a natural flow without explicit labels.
    """
    context_text = "\n\n---\n\n".join(retrieved_chunks)
    
    prompt = (
        f"You are a professional research assistant. Your goal is to answer the user's question using the provided context from a research paper.\n\n"
        f"INSTRUCTIONS:\n"
        f"1. **Flowing Response**: Provide a detailed answer that flows naturally. DO NOT use labels like 'Document First' or 'From my side insights' in your response.\n"
        f"2. **Context Grounding**: Use the provided context below to form your core answer. If the information is not discussed in the document, you may use your internal knowledge to provide a helpful answer, but mention it naturally (e.g., 'While not explicitly mentioned in this document, general industry practices suggest...').\n"
        f"3. **Expansion & Detail**: Explain the 'why' and 'how' in depth. Use analogies and real-world examples to make it easy to understand.\n"
        f"4. **Math**: Wrap every mathematical formula in double dollar signs ($$ formula $$). DO NOT use code blocks for math.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        f"Natural, Detailed Answer:"
    )
    
    client = _get_client()
    try:
        for message in client.chat_completion(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.3,
            stream=True
        ):
            if hasattr(message, "choices") and len(message.choices) > 0:
                token = message.choices[0].delta.content
                if token:
                    yield token
    except Exception as e:
        yield f"Error analyzing document: {e}"
