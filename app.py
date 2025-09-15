import streamlit as st
from PIL import Image
import numpy as np
import json
import io

# Grid size (smaller = higher compression, bigger = better quality)
GRID = 16  

def image_to_symbolic(img, grid=GRID, palette_size=16):
    img = img.convert("RGB")
    w, h = img.size
    img = img.resize(((w // grid) * grid, (h // grid) * grid))  # fit to grid
    w, h = img.size

    # Quantize to limited colors (palette)
    img = img.convert("P", palette=Image.ADAPTIVE, colors=palette_size).convert("RGB")
    pixels = np.array(img)

    symbolic = {
        "width": w,
        "height": h,
        "grid": grid,
        "palette": list({tuple(p) for row in pixels for p in row}),  # unique colors
        "blocks": []
    }
    palette = symbolic["palette"]

    # Encode blocks
    for y in range(0, h, grid):
        for x in range(0, w, grid):
            block = pixels[y:y+grid, x:x+grid]
            avg = np.mean(block.reshape(-1,3), axis=0).astype(int).tolist()
            # Find nearest palette index
            color_idx = min(range(len(palette)), key=lambda i: np.linalg.norm(np.array(palette[i])-avg))
            symbolic["blocks"].append(f"B{x//grid},{y//grid},{color_idx}")

    return symbolic

def symbolic_to_image(symbolic):
    w, h, grid = symbolic["width"], symbolic["height"], symbolic["grid"]
    palette = symbolic["palette"]

    img = Image.new("RGB", (w, h), "white")
    for block in symbolic["blocks"]:
        _, coords, color_idx = block[0], block[1:-2], block.split(",")[-1]
        x, y = map(int, block[1:-2].split(","))
        color = tuple(palette[int(color_idx)])
        for yy in range(y*grid, (y+1)*grid):
            for xx in range(x*grid, (x+1)*grid):
                img.putpixel((xx, yy), color)
    return img

# ---------------- Streamlit UI ----------------
st.title("🟢 Image to Tiny Symbolic JSON")

option = st.radio("Choose Action:", ["Encode Image → JSON", "Decode JSON → Image"])

if option == "Encode Image → JSON":
    uploaded = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Original", use_column_width=True)

        symbolic = image_to_symbolic(img)
        json_bytes = json.dumps(symbolic, separators=(",",":")).encode()

        st.download_button("⬇️ Download Symbolic JSON", data=json_bytes, file_name="symbolic.json", mime="application/json")

elif option == "Decode JSON → Image":
    uploaded = st.file_uploader("Upload Symbolic JSON", type=["json"])
    if uploaded:
        symbolic = json.load(uploaded)
        img = symbolic_to_image(symbolic)
        st.image(img, caption="Decoded Image", use_column_width=True)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.download_button("⬇️ Download Decoded Image", data=buf.getvalue(), file_name="decoded.png", mime="image/png")
