import os
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Use SDXL which is currently supported on the HF free Serverless tier
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

def generate_image(prompt: str) -> Image.Image:
    """
    Generate an AI image based on a text prompt using Hugging Face Serverless Inference.
    Works locally with .env or on Streamlit with Secrets.
    """
    # First check Streamlit secrets, then local .env
    token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN is missing. Please create a .env file.")
        
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=token)
        image = client.text_to_image(prompt, model=MODEL_ID)
        return image
    except Exception as e:
        raise RuntimeError(f"Text-to-Image API Error: {e}")
