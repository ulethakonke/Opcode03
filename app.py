import streamlit as st
import numpy as np
from PIL import Image
import io
import json
import base64

# --- Helper functions ---

def image_to_patches(img, patch_size=32, quality=70):
    """Convert image into patches and return symbolic JSON with JPEG base64."""
    w, h = img.size
    img = img.convert("RGB")
    patches = []
    symbolic = {"width": w, "height": h, "patch_size": patch_size, "patches": []}
    
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            box = (x, y, min(x+patch_size, w), min(y+patch_size, h))
            patch = img.crop(box)
            buffer = io.BytesIO()
            patch.save(buffer, format="JPEG", quality=quality)
            b64 = base64.b64encode(buffer.getvalue()).decode()
            symbolic["patches"].append({"x": x, "y": y, "data": b64})
    return symbolic

def patches_to_image(symbolic):
    """Decode symbolic JSON back to image."""
    w = symbolic["width"]
    h = symbolic["height"]
    patch_size = symbolic["patch_size"]
    canvas = Image.new("RGB", (w, h))
    for p in symbolic["patches"]:
        patch_data = base64.b64decode(p["data"])
        patch_img = Image.open(io.BytesIO(patch_data))
        canvas.paste(patch_img, (p["x"], p["y"]))
    return canvas

# --- Streamlit UI ---

st.title("Symbolic Image Codec with JPEG Patches")

mode = st.radio("Select mode:", ["Encode Image", "Decode JSON"])

if mode == "Encode Image":
    uploaded = st.file_uploader("Upload image (PNG, JPEG)", type=["png", "jpg", "jpeg"])
    patch_size = st.number_input("Patch size (px)", min_value=8, max_value=128, value=32)
    quality = st.slider("JPEG quality", min_value=10, max_value=95, value=70)
    
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Original Image", use_column_width=True)
        
        symbolic = image_to_patches(img, patch_size=patch_size, quality=quality)
        json_bytes = json.dumps(symbolic, indent=2).encode()
        
        st.download_button(
            label="Download Symbolic JSON",
            data=json_bytes,
            file_name="symbolic.json",
            mime="application/json"
        )
        st.success(f"Encoded! Symbolic JSON size: {len(json_bytes)//1024} KB")

elif mode == "Decode JSON":
    uploaded_json = st.file_uploader("Upload symbolic JSON", type=["json"])
    
    if uploaded_json:
        symbolic = json.load(uploaded_json)
        decoded_img = patches_to_image(symbolic)
        st.image(decoded_img, caption="Decoded Image", use_column_width=True)
        
        buffer = io.BytesIO()
        decoded_img.save(buffer, format="PNG")
        st.download_button(
            label="Download Decoded PNG",
            data=buffer.getvalue(),
            file_name="decoded.png",
            mime="image/png"
        )
        st.success("Decoded image ready!")
