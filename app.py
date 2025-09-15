# app.py
import streamlit as st
import json
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# ------------------------------
# Helper Functions
# ------------------------------

# 1. Simplified symbolic alphabet (16 colors, 4-bit per pixel)
SYMBOLS = "0123456789ABCDEF"
COLOR_PALETTE = [
    (0,0,0), (255,255,255), (255,0,0), (0,255,0),
    (0,0,255), (255,255,0), (255,0,255), (0,255,255),
    (128,0,0), (0,128,0), (0,0,128), (128,128,0),
    (128,0,128), (0,128,128), (192,192,192), (128,128,128)
]

# Encode image to symbolic JSON
def image_to_symbolic(img: Image.Image):
    img = img.convert("RGB").resize((16,16))  # 16x16 for 1KB goal
    arr = np.array(img)
    grid = []
    for y in range(16):
        row = []
        for x in range(16):
            # Find closest color in palette
            pixel = tuple(arr[y,x])
            distances = [sum((c-p)**2 for c,p in zip(color,pixel)) for color in COLOR_PALETTE]
            index = int(np.argmin(distances))
            row.append(SYMBOLS[index])
        grid.append(row)
    return {"width":16, "height":16, "grid":grid}

# Decode symbolic JSON back to image
def symbolic_to_image(symbolic):
    w, h = symbolic["width"], symbolic["height"]
    grid = symbolic["grid"]
    img = Image.new("RGB", (w,h))
    for y in range(h):
        for x in range(w):
            symbol = grid[y][x]
            index = SYMBOLS.index(symbol)
            img.putpixel((x,y), COLOR_PALETTE[index])
    return img

# ------------------------------
# Streamlit App
# ------------------------------

st.title("1KB Symbolic Image Encoder/Decoder")

# Upload image to encode
uploaded_file = st.file_uploader("Upload Image to Encode", type=["png","jpg","jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Original Image", use_column_width=True)
    
    symbolic = image_to_symbolic(img)
    json_str = json.dumps(symbolic, indent=2)
    
    st.download_button(
        label="Download Symbolic JSON",
        data=json_str,
        file_name="symbolic.json",
        mime="application/json"
    )
    
    st.write("Encoded symbolic JSON preview:")
    st.code(json_str[:500] + "\n...")  # preview first 500 chars

# Upload symbolic JSON to decode
uploaded_json = st.file_uploader("Upload Symbolic JSON to Decode", type=["json"])
if uploaded_json:
    symbolic = json.load(uploaded_json)
    img_decoded = symbolic_to_image(symbolic)
    st.image(img_decoded, caption="Decoded Image", use_column_width=True)
    
    # Download decoded image
    buffered = BytesIO()
    img_decoded.save(buffered, format="PNG")
    st.download_button(
        label="Download Decoded PNG",
        data=buffered.getvalue(),
        file_name="decoded.png",
        mime="image/png"
    )
