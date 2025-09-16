import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import io

# Symbolic alphabet
SYMBOLS = {
    "square": "⬛",
    "circle": "⬤",
    "triangle_up": "▲",
    "triangle_down": "▼",
    "diamond": "◆",
    "gradient": "▒",
}

def block_is_uniform(block, tolerance=15):
    """Check if a block is mostly one color."""
    pixels = np.array(block).reshape(-1, 3)
    mean_color = pixels.mean(axis=0)
    diffs = np.abs(pixels - mean_color).mean()
    return diffs < tolerance, tuple(map(int, mean_color))

def choose_shape(block):
    """Decide best shape for this block based on variance & edges."""
    arr = np.array(block).astype(np.int32)
    gray = arr.mean(axis=2)

    var = gray.var()
    gy, gx = np.gradient(gray.astype(float))
    edge_strength = np.mean(np.abs(gx) + np.abs(gy))

    if var < 80:  # very flat
        return "square"
    elif edge_strength < 40:  # smooth → circle or gradient
        return "circle" if var < 200 else "gradient"
    elif edge_strength > 300:  # strong edges → triangles or diamonds
        return "diamond" if var > 500 else "triangle_up"
    else:
        return "triangle_down"

def encode_block(x, y, size, img, min_size=2):
    """Recursively encode a block of the image."""
    block = img.crop((x, y, x + size, y + size))
    uniform, color = block_is_uniform(block)

    if uniform or size <= min_size:
        shape = choose_shape(block)
        symbol = f"{SYMBOLS[shape]}{color}@({x},{y})S{size}"
        return [symbol]
    else:
        half = size // 2
        symbols = []
        for dx in [0, half]:
            for dy in [0, half]:
                symbols.extend(encode_block(x + dx, y + dy, half, img, min_size))
        return symbols

def encode_image_to_language(img, block_size=16):
    """Encode entire image into symbolic language."""
    width, height = img.size
    symbols = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            symbols.extend(encode_block(x, y, block_size, img))
    return "\n".join(symbols)

def decode_language_to_image(symbols, width, height):
    """Decode symbolic language back into an image."""
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for line in symbols.splitlines():
        try:
            symbol, rest = line[0], line[1:]
            color_str, pos_size = rest.split("@")
            color = tuple(map(int, color_str.strip("()").split(",")))
            pos, size_str = pos_size.split("S")
            x, y = map(int, pos.strip("()").split(","))
            size = int(size_str)

            if symbol == SYMBOLS["square"]:
                draw.rectangle([x, y, x+size, y+size], fill=color)
            elif symbol == SYMBOLS["circle"]:
                draw.ellipse([x, y, x+size, y+size], fill=color)
            elif symbol == SYMBOLS["triangle_up"]:
                draw.polygon([(x+size//2, y), (x, y+size), (x+size, y+size)], fill=color)
            elif symbol == SYMBOLS["triangle_down"]:
                draw.polygon([(x, y), (x+size, y), (x+size//2, y+size)], fill=color)
            elif symbol == SYMBOLS["diamond"]:
                draw.polygon([(x+size//2, y), (x+size, y+size//2),
                              (x+size//2, y+size), (x, y+size//2)], fill=color)
            elif symbol == SYMBOLS["gradient"]:
                for i in range(size):
                    c = tuple(int(c * (i/size)) for c in color)
                    draw.line([x+i, y, x+i, y+size], fill=c)
        except Exception as e:
            print("Decode error:", line, e)
    return canvas

# ---------------- Streamlit UI ----------------
st.title("Advanced Symbolic Image Language")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Original", use_column_width=True)

    # Encode
    lang = encode_image_to_language(img, block_size=16)
    st.subheader("Encoded Language")
    st.text_area("Symbols", lang[:2000] + ("..." if len(lang) > 2000 else ""), height=300)

    # Decode
    decoded_img = decode_language_to_image(lang, *img.size)
    st.subheader("Decoded Image")
    st.image(decoded_img, caption="Reconstructed", use_column_width=True)

    # Download language
    lang_bytes = io.BytesIO(lang.encode())
    st.download_button("Download Symbolic Language", lang_bytes, "symbols.txt", "text/plain")

    # Download decoded image
    img_bytes = io.BytesIO()
    decoded_img.save(img_bytes, format="PNG")
    st.download_button("Download Decoded Image", img_bytes.getvalue(), "decoded.png", "image/png")
