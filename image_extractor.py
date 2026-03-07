import streamlit as st
import fitz  # PyMuPDF
import os
import numpy as np
from io import BytesIO
from PIL import Image
from typing import List, Dict

@st.cache_data
def extract_images_from_pdf(uploaded_file, output_dir: str = "extracted_images") -> List[Dict[str, str]]:
    """
    Extract images from PDF and attempt to find nearby captions for context.
    
    Returns:
        List of dicts: [{"path": "...", "caption": "..."}]
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    
    results = []
    image_count = 0
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        image_list = page.get_images(full=True)
        
        # Get all text blocks on the page to search for captions later
        text_blocks = page.get_text("blocks")
        
        for img_info in image_list:
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            try:
                img = Image.open(BytesIO(image_bytes))
                
                # Filter out small icons/noise
                if img.width < 150 or img.height < 150:
                    continue
                    
                # Filter out extreme aspect ratios
                aspect_ratio = img.width / img.height
                if aspect_ratio < 0.1 or aspect_ratio > 10.0:
                    continue
                
                # Check variance to filter out solid color blocks/logos
                img_array = np.array(img.convert("L"))
                if np.var(img_array) < 400:
                    continue
                    
                # Attempt to find caption
                img_rects = page.get_image_rects(xref)
                caption = ""
                if img_rects:
                    r = img_rects[0]
                    potential_captions = []
                    for b in text_blocks:
                        if b[1] > r.y1 - 5 and b[1] < r.y1 + 100:
                            text = b[4].strip()
                            if text.lower().startswith(("fig", "figure", "table", "chart")):
                                potential_captions.append(text)
                    
                    if potential_captions:
                        caption = potential_captions[0].replace("\n", " ")
                
                image_filename = f"img_p{page_idx + 1}_{image_count}.png"
                image_path = os.path.join(output_dir, image_filename)
                
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                    
                img.save(image_path, format="PNG")
                
                results.append({
                    "path": image_path,
                    "caption": caption if caption else f"Extracted from page {page_idx + 1}"
                })
                image_count += 1
                
            except Exception as e:
                print(f"Skipped image on page {page_idx + 1}: {e}")
                
    return results
