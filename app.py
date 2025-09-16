import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import json
import io

# ===================== ENCODER =====================
def image_to_symbolic(img):
    img = img.convert("RGB")
    w, h = img.size
    pixels = np.array(img)

    symbolic = {"width": w, "height": h, "pixels": []}

    for y in range(h):
        for x in range(w):
            color = pixels[y, x].tolist()
            symbolic["pixels"].append({"x": x, "y": y, "color": color})

    return symbolic


# ===================== DECODER =====================
def symbolic_to_image(symbolic):
    w, h = symbolic["width"], symbolic["height"]
    canvas = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(canvas)

    for p in symbolic["pixels"]:
        x, y = p["x"], p["y"]
        color = tuple(p["color"])
        # draw single pixel
        draw.point((x, y), fill=color)

    return canvas


# ===================== STREAMLIT APP =====================
st.title("🖼️ Sharp Symbolic Image Codec")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Original", use_column_width=True)

    if st.button("Encode → Symbolic"):
        symbolic = image_to_symbolic(img)
        st.success("Image encoded with pixel precision!")

        # ✅ Save symbolic JSON properly
        json_str = json.dumps(symbolic, indent=2)
        json_bytes = io.BytesIO(json_str.encode("utf-8"))
        st.download_button("⬇️ Download Symbolic JSON", json_bytes, "symbolic.json", "application/json")

        # ✅ Preview decoded image
        decoded = symbolic_to_image(symbolic)
        st.image(decoded, caption="Decoded (Sharp)", use_column_width=True)

        # ✅ Save decoded PNG properly
        png_buf = io.BytesIO()
        decoded.save(png_buf, format="PNG")
        png_buf.seek(0)
        st.download_button("⬇️ Download Decoded PNG", png_buf, "decoded.png", "image/png")
