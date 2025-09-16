import streamlit as st
from PIL import Image
import numpy as np
import io
import base64

# ---------- Helpers ----------

SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"  # 64-symbol alphabet

def float_to_symbol(f, min_val, max_val, num_symbols=len(SYMBOLS)):
    """Convert float to symbol"""
    f_clamped = max(min(f, max_val), min_val)
    index = int((f_clamped - min_val) / (max_val - min_val) * (num_symbols - 1))
    return SYMBOLS[index]

def symbol_to_float(s, min_val, max_val):
    """Convert symbol back to float"""
    index = SYMBOLS.index(s)
    return min_val + (max_val - min_val) * index / (len(SYMBOLS) - 1)

def encode_image_to_symbolic(img, block_size=4):
    """Encode image to symbolic text using simple interference pattern approximation"""
    img = img.convert("RGB")
    arr = np.array(img)
    h, w, _ = arr.shape
    symbolic = ""
    
    # Process blocks
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = arr[y:y+block_size, x:x+block_size]
            # Compute simple pattern: mean RGB per block
            mean_color = block.mean(axis=(0,1))  # [R_mean, G_mean, B_mean]
            for c in mean_color:
                symbolic += float_to_symbol(c, 0, 255)
    return symbolic, w, h, block_size

def decode_symbolic_to_image(symbolic, w, h, block_size=4):
    """Decode symbolic text back to image"""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    idx = 0
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            if idx + 3 > len(symbolic):
                break
            r = int(symbol_to_float(symbolic[idx], 0, 255))
            g = int(symbol_to_float(symbolic[idx+1], 0, 255))
            b = int(symbol_to_float(symbolic[idx+2], 0, 255))
            idx += 3
            arr[y:y+block_size, x:x+block_size] = [r, g, b]
    return Image.fromarray(arr)

def get_download_link(img, filename="decoded.png"):
    """Generate a download link for PIL Image"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_data = buf.getvalue()
    b64 = base64.b64encode(byte_data).decode()
    return f'<a href="data:file/png;base64,{b64}" download="{filename}">Download Decoded Image</a>'

# ---------- Streamlit App ----------

st.title("Interference-Pattern Symbolic Encoder/Decoder")

uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
block_size = st.slider("Block size (higher = more compression)", 2, 16, 4)

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Original Image", use_column_width=True)
    
    # Encode
    symbolic, w, h, bs = encode_image_to_symbolic(img, block_size=block_size)
    st.text_area("Symbolic Representation", symbolic, height=200)
    st.write(f"Symbolic length: {len(symbolic)} chars")
    
    # Decode
    decoded_img = decode_symbolic_to_image(symbolic, w, h, block_size=block_size)
    st.image(decoded_img, caption="Decoded Image", use_column_width=True)
    
    # Download
    st.markdown(get_download_link(decoded_img), unsafe_allow_html=True)
