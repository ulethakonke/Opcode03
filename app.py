import streamlit as st
from PIL import Image
import numpy as np
import io, base64, zlib

# ------------------------------
# Encode image -> compressed text
# ------------------------------
def image_to_symbolic(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")  # keep lossless format
    raw_bytes = buf.getvalue()
    compressed = zlib.compress(raw_bytes, level=9)
    symbolic = base64.b64encode(compressed).decode("utf-8")
    return symbolic

# ------------------------------
# Decode text -> image
# ------------------------------
def symbolic_to_image(symbolic: str) -> Image.Image:
    compressed = base64.b64decode(symbolic.encode("utf-8"))
    raw_bytes = zlib.decompress(compressed)
    buf = io.BytesIO(raw_bytes)
    img = Image.open(buf)
    return img

# ------------------------------
# Streamlit app
# ------------------------------
st.title("Image Encoder/Decoder")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original Image", use_container_width=True)

    # Encode
    symbolic = image_to_symbolic(img)
    st.success(f"Encoded length: {len(symbolic)} characters")
    st.text_area("Symbolic Representation", symbolic[:1000] + "...", height=200)

    # Decode
    decoded_img = symbolic_to_image(symbolic)
    st.image(decoded_img, caption="Decoded Image", use_container_width=True)

    # Download decoded image
    buf = io.BytesIO()
    decoded_img.save(buf, format="PNG")
    byte_data = buf.getvalue()
    st.download_button("Download Decoded Image", data=byte_data,
                       file_name="decoded.png", mime="image/png")
