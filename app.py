# app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
import zlib

# ---------- Custom ASCII alphabet for dense encoding ----------
ASCII_ALPHABET = [chr(i) for i in range(33, 127)]  # 94 printable chars

def bytes_to_custom_ascii(data: bytes) -> str:
    """Encode bytes to custom ASCII using 94 chars"""
    result = ""
    bit_str = ''.join(f'{byte:08b}' for byte in data)
    for i in range(0, len(bit_str), 7):
        chunk = bit_str[i:i+7]
        if len(chunk) < 7:
            chunk = chunk.ljust(7, '0')
        idx = int(chunk, 2) % len(ASCII_ALPHABET)
        result += ASCII_ALPHABET[idx]
    return result

def custom_ascii_to_bytes(s: str) -> bytes:
    """Decode custom ASCII back to bytes"""
    bit_str = ""
    for c in s:
        idx = ASCII_ALPHABET.index(c)
        bit_str += f'{idx:07b}'
    # Group into bytes
    bytes_list = []
    for i in range(0, len(bit_str), 8):
        chunk = bit_str[i:i+8]
        if len(chunk) < 8:
            break
        bytes_list.append(int(chunk, 2))
    return bytes(bytes_list)

# ---------- Palette + RLE Encoding ----------
def image_to_symbolic(img: Image.Image, palette_size=16) -> str:
    """Convert image to symbolic text"""
    img = img.convert("RGB")
    # Quantize to palette_size colors
    img_small = img.quantize(colors=palette_size)
    palette = img_small.getpalette()[:palette_size*3]  # R,G,B triplets
    indices = np.array(img_small)
    # Flatten and run-length encode
    flat = indices.flatten()
    rle = []
    count = 1
    for i in range(1, len(flat)):
        if flat[i] == flat[i-1]:
            count += 1
        else:
            rle.append((flat[i-1], count))
            count = 1
    rle.append((flat[-1], count))
    # Convert RLE to bytes: each pair (value,count) as two bytes
    rle_bytes = bytearray()
    for val, cnt in rle:
        rle_bytes.append(val)
        rle_bytes.append(min(cnt, 255))  # cap count at 255
    # Compress bytes for extra shrink
    compressed = zlib.compress(bytes(rle_bytes))
    # Encode with custom ASCII
    symbolic_text = bytes_to_custom_ascii(compressed)
    # Add palette metadata (palette RGB triplets)
    palette_str = ','.join(map(str, palette))
    metadata = f"{img.width},{img.height},{palette_size},{palette_str}|"
    return metadata + symbolic_text

def symbolic_to_image(symbolic_text: str) -> Image.Image:
    """Decode symbolic text back to image"""
    # Split metadata
    metadata, data_str = symbolic_text.split('|', 1)
    width, height, palette_size, *palette_list = metadata.split(',')
    width = int(width)
    height = int(height)
    palette_size = int(palette_size)
    palette = list(map(int, palette_list))
    # Decode ASCII to bytes and decompress
    compressed = custom_ascii_to_bytes(data_str)
    rle_bytes = zlib.decompress(compressed)
    # Expand RLE
    flat = []
    for i in range(0, len(rle_bytes), 2):
        val = rle_bytes[i]
        cnt = rle_bytes[i+1]
        flat.extend([val]*cnt)
    flat = np.array(flat, dtype=np.uint8)
    img_array = flat.reshape((height, width))
    img = Image.fromarray(img_array, mode='P')
    img.putpalette(palette + [0]*(768-len(palette)))  # pad palette to 256*3
    return img.convert("RGB")

# ---------- Streamlit UI ----------
st.title("Image → Symbolic Text Encoder/Decoder")

tab1, tab2 = st.tabs(["Encode", "Decode"])

with tab1:
    uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
    palette_size = st.slider("Palette Size", min_value=2, max_value=32, value=16)
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Original Image", use_column_width=True)
        symbolic_text = image_to_symbolic(img, palette_size=palette_size)
        st.text_area("Symbolic Representation", symbolic_text, height=200)
        # Download button
        st.download_button(
            label="Download Symbolic File",
            data=symbolic_text,
            file_name="symbolic.txt"
        )

with tab2:
    uploaded_symbolic = st.file_uploader("Upload Symbolic Text", type=["txt"])
    if uploaded_symbolic is not None:
        symbolic_text = uploaded_symbolic.read().decode()
        reconstructed_img = symbolic_to_image(symbolic_text)
        st.image(reconstructed_img, caption="Decoded Image", use_column_width=True)
        # Download reconstructed image
        buffer = io.BytesIO()
        reconstructed_img.save(buffer, format="PNG")
        st.download_button(
            label="Download Decoded PNG",
            data=buffer.getvalue(),
            file_name="decoded.png"
        )
