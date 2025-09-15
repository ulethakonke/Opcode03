import streamlit as st
from PIL import Image
import numpy as np
import json
import io
import random

PATCH_SIZE = 16  # each patch is 16x16

# --- Encoder ---
def encode_image(img: Image.Image):
    img = img.convert("RGB")
    w, h = img.size
    symbolic = {
        "width": w,
        "height": h,
        "patch_size": PATCH_SIZE,
        "patches": []
    }
    
    pixels = np.array(img)
    
    for y in range(0, h, PATCH_SIZE):
        for x in range(0, w, PATCH_SIZE):
            patch = pixels[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            avg_color = patch.mean(axis=(0,1)).astype(int).tolist()
            seed = random.randint(0, 65535)
            
            symbolic["patches"].append({
                "x": int(x),
                "y": int(y),
                "seed": seed,
                "avg_color": avg_color
            })
    
    return symbolic

# --- Decoder ---
def decode_symbolic(symbolic):
    w, h = symbolic["width"], symbolic["height"]
    canvas = Image.new("RGB", (w,h))
    for patch in symbolic["patches"]:
        x, y = patch["x"], patch["y"]
        avg_color = patch["avg_color"]
        seed = patch["seed"]
        
        # Generate pattern from seed
        random.seed(seed)
        patch_array = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
        for i in range(PATCH_SIZE):
            for j in range(PATCH_SIZE):
                patch_array[i,j] = [min(255,max(0, avg_color[c] + random.randint(-10,10))) for c in range(3)]
        
        canvas.paste(Image.fromarray(patch_array), (x,y))
    
    return canvas

# --- Streamlit UI ---
st.title("Observer Effect Inspired Image Encoder")

uploaded_file = st.file_uploader("Upload an Image", type=["png","jpg","jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Original Image", use_column_width=True)
    
    if st.button("Encode to Symbolic JSON"):
        symbolic = encode_image(img)
        json_bytes = json.dumps(symbolic).encode()
        st.download_button("Download JSON", data=json_bytes, file_name="symbolic.json")
    
    st.markdown("---")
    
    uploaded_json = st.file_uploader("Upload Symbolic JSON to Decode", type=["json"])
    if uploaded_json:
        symbolic_data = json.load(uploaded_json)
        canvas = decode_symbolic(symbolic_data)
        st.image(canvas, caption="Decoded Image", use_column_width=True)
