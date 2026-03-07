import os
import base64
from huggingface_hub import InferenceClient
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import streamlit as st  # Only used for secrets management in this file

load_dotenv()

# Use Llama Vision which is actively supported on the free HF inference API
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

def _get_client() -> InferenceClient:
    """
    Returns a Hugging Face InferenceClient.
    Works locally (.env) or on Streamlit (Secrets).
    """
    token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN is missing. Please create a .env file and add:\n"
            "HF_TOKEN=your_huggingface_token"
        )
    return InferenceClient(token=token)

def _image_to_base64_data_uri(image_path: str) -> str:
    """Convert an image file to a base64 data URI acceptable by HF."""
    img = Image.open(image_path)
    
    # Resize if too large to save bandwidth & prevent API memory errors
    if img.width > 1024 or img.height > 1024:
        img.thumbnail((1024, 1024))
        
    buffered = BytesIO()
    # Save as JPEG to compress the payload
    # Qwen-VL handles JPEG perfectly fine
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img.save(buffered, format="JPEG", quality=85)
    
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def analyze_image(image_path: str) -> str:
    """
    Send an image to the Qwen2-VL model via the HF Inference API for analysis.
    
    Args:
        image_path: Local path to the extracted image.
        
    Returns:
        The analysis string returned by the VLM.
    """
    client = _get_client()
    data_uri = _image_to_base64_data_uri(image_path)
    
    prompt = (
        "You are analyzing a figure from a research paper. "
        "Explain what the figure shows, describe the key trend or insight, "
        "and explain why it is important. Be clear and concise."
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]
        }
    ]
    
    try:
        response = client.chat_completion(
            model=MODEL_ID,
            messages=messages,
            max_tokens=250,
            temperature=0.3
        )
        content = response.choices[0].message.content
        return content.strip() if content else "No analysis available"
    except Exception as e:
        return f"Failed to analyze figure: {e}"
