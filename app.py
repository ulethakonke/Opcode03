import base64
import zlib
from PIL import Image
import numpy as np
import streamlit as st
from io import BytesIO

# --- Encoding ---
def encode_image_to_symbolic_printable(img, size=(64, 64), palette_size=16):
    img = img.resize(size, Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=palette_size)
    palette = img.getpalette()[:palette_size * 3]
    arr = np.array(img, dtype=np.uint8)

    # Flatten + combine palette + pixels
    data = bytes([palette_size]) + bytes(palette) + arr.tobytes()

    # Compress
    compressed = zlib.compress(data, level=9)

    # Encode using Ascii85 (portable)
    printable = base64.a85encode(compressed).decode("ascii")

    return printable

# --- Decoding ---
def decode_symbolic_printable(text):
    compressed = base64.a85decode(text.encode("ascii"))
    data = zlib.decompress(compressed)

    palette_size = data[0]
    palette = list(data[1:1 + palette_size * 3])
    pixel_data = data[1 + palette_size * 3:]

    # Rebuild image
    side = int(len(pixel_data) ** 0.5)
    arr = np.frombuffer(pixel_data, dtype=np.uint8).reshape((side, side))
    img = Image.fromarray(arr, mode="P")
    img.putpalette(palette + [0] * (768 - len(palette)))
    return img

# --- Streamlit UI ---
st.title("Symbolic Image Encoder/Decoder")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Original", use_column_width=True)

    symbolic = encode_image_to_symbolic_printable(img, size=(64, 64), palette_size=16)
    st.text_area("Symbolic Representation", symbolic, height=200)

    decoded = decode_symbolic_printable(symbolic)
    st.image(decoded, caption="Decoded Image", use_column_width=True)

    buf = BytesIO()
    decoded.save(buf, format="PNG")
    st.download_button("Download Decoded Image", buf.getvalue(), "decoded.png", "image/png")
