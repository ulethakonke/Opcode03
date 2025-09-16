import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import json
import io

# ===================== ENCODER =====================
def image_to_symbolic(img, block_size=8):
    img = img.convert("RGB")
    w, h = img.size
    pixels = np.array(img)

    symbolic = {"width": w, "height": h, "block_size": block_size, "shapes": []}

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = pixels[y:y+block_size, x:x+block_size]
            avg_color = block.mean(axis=(0, 1)).astype(int).tolist()

            # Pick shape based on variance
            variance = block.var()
            if variance < 100:   # flat → square
                shape_type = "square"
            elif variance < 500: # medium detail → circle
                shape_type = "circle"
            else:                # high detail → triangle
                shape_type = "triangle"

            symbolic["shapes"].append({
                "x": x, "y": y,
                "color": avg_color,
                "shape": shape_type
            })

    return symbolic


# ===================== DECODER =====================
def symbolic_to_image(symbolic):
    w, h, block_size = symbolic["width"], symbolic["height"], symbolic["block_size"]
    canvas = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(canvas)

    for s in symbolic["shapes"]:
        x, y = s["x"], s["y"]
        color = tuple(s["color"])
        shape = s["shape"]

        # Always fill background rectangle to avoid gaps
        draw.rectangle([x, y, x+block_size, y+block_size], fill=color)

        # Overlay shape for detail
        if shape == "circle":
            draw.ellipse([x, y, x+block_size, y+block_size], fill=color)
        elif shape == "triangle":
            draw.polygon(
                [(x, y+block_size), (x+block_size/2, y), (x+block_size, y+block_size)],
                fill=color
            )
        # squares are already drawn by the background rectangle

    return canvas


# ===================== STREAMLIT APP =====================
st.title("🖼️ Symbolic Image Codec (No Gaps Fix)")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Original", use_column_width=True)

    if st.button("Encode → Symbolic"):
        symbolic = image_to_symbolic(img, block_size=8)
        st.success("Image encoded!")

        # Save symbolic JSON
        json_bytes = json.dumps(symbolic).encode("utf-8")
        st.download_button("⬇️ Download Symbolic JSON", json_bytes, "symbolic.json")

        # Preview symbolic
        decoded = symbolic_to_image(symbolic)
        st.image(decoded, caption="Decoded (Fixed)", use_column_width=True)

        buf = io.BytesIO()
        decoded.save(buf, format="PNG")
        st.download_button("⬇️ Download Decoded PNG", buf.getvalue(), "decoded.png")
