import streamlit as st
from PIL import Image
import io, base64, bz2

# ------------------------------
# Encode image -> compressed text
# ------------------------------
def image_to_symbolic(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")  # keep lossless
    raw_bytes = buf.getvalue()
    compressed = bz2.compress(raw_bytes)  # better than zlib
    symbolic = base64.b64encode(compressed).decode("utf-8")
    return symbolic

# ------------------------------
# Decode text -> image
# ------------------------------
def symbolic_to_image(symbolic: str) -> Image.Image:
    compressed = base64.b64decode(symbolic.encode("utf-8"))
    raw_bytes = bz2.decompress(compressed)
    buf = io.BytesIO(raw_bytes)
    img = Image.open(buf)
    return img

# ------------------------------
# Streamlit app
# ------------------------------
st.title("Image Encoder/Decoder (Compact Symbolic)")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original Image", use_container_width=True)

    # Encode
    symbolic = image_to_symbolic(img)
    st.success(f"Encoded length: {len(symbolic)} characters")
    st.text_area("Symbolic Representation (preview)", symbolic[:1000] + "...", height=200)

    # Download symbolic representation
    st.download_button("Download Symbolic File",
                       data=symbolic.encode("utf-8"),
                       file_name="image.sym",
                       mime="text/plain")

    # Decode
    decoded_img = symbolic_to_image(symbolic)
    st.image(decoded_img, caption="Decoded Image", use_container_width=True)

    # Download decoded image
    buf = io.BytesIO()
    decoded_img.save(buf, format="PNG")
    byte_data = buf.getvalue()
    st.download_button("Download Decoded Image",
                       data=byte_data,
                       file_name="decoded.png",
                       mime="image/png")

# Optional: Upload symbolic text directly
symbolic_upload = st.file_uploader("Or upload a symbolic file (.sym)", type=["sym"])
if symbolic_upload:
    sym_data = symbolic_upload.read().decode("utf-8")
    decoded_img = symbolic_to_image(sym_data)
    st.image(decoded_img, caption="Decoded from Symbolic File", use_container_width=True)
