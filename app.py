import streamlit as st
from PIL import Image, ImageDraw
import numpy as np

# --- Define a tiny color alphabet ---
COLOR_ALPHABET = {
    "A": (255, 0, 0),     # Red
    "B": (0, 0, 255),     # Blue
    "C": (0, 255, 0),     # Green
    "D": (255, 255, 0),   # Yellow
    "E": (0, 0, 0),       # Black
    "F": (255, 255, 255)  # White
}

REVERSE_COLOR = {v: k for k, v in COLOR_ALPHABET.items()}


def closest_color(rgb):
    """Find the closest color in the alphabet to a given RGB pixel."""
    diffs = []
    for c in COLOR_ALPHABET.values():
        diff = sum((p - q) ** 2 for p, q in zip(rgb, c))
        diffs.append((diff, c))
    return min(diffs, key=lambda x: x[0])[1]


def encode_image(img, grid_size=8):
    """
    Encode image into symbolic language by dividing into blocks.
    """
    img = img.convert("RGB")
    w, h = img.size
    lang = []

    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            block = img.crop((x, y, x + grid_size, y + grid_size))
            arr = np.array(block)
            avg_color = tuple(np.mean(arr.reshape(-1, 3), axis=0).astype(int))
            nearest = closest_color(avg_color)
            sym = REVERSE_COLOR[nearest]
            size = grid_size
            lang.append(f"⬛{sym}({x},{y})S{size}")

    return " | ".join(lang)


def decode_language(lang, img_size=(128, 128)):
    """
    Decode symbolic language back into an image.
    """
    canvas = Image.new("RGB", img_size, (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    commands = lang.split("|")
    for cmd in commands:
        cmd = cmd.strip()
        if not cmd:
            continue

        try:
            shape = cmd[0]      # always ⬛ for now
            color_key = cmd[1]  # e.g., "A"
            inside = cmd[2:].split(")")[0]
            coords = inside[1:].split(",")
            x, y = int(coords[0]), int(coords[1])
            size = int(cmd.split("S")[-1])

            color = COLOR_ALPHABET[color_key]

            if shape == "⬛":
                draw.rectangle([x, y, x + size, y + size], fill=color)

        except Exception as e:
            print("Error decoding command:", cmd, e)

    return canvas


# ---------------- Streamlit UI ----------------
st.title("🖼️ Image Language Prototype")
st.write("Upload an image → Encode to text → Decode back to image.")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).resize((128, 128))  # small for demo
    st.image(img, caption="Original Image", use_column_width=True)

    # Encode
    lang = encode_image(img, grid_size=16)
    st.subheader("📝 Encoded Language")
    st.text_area("Symbolic text", value=lang, height=200)

    # Decode
    decoded = decode_language(lang, img_size=(128, 128))
    st.subheader("🔄 Decoded Image")
    st.image(decoded, caption="Reconstructed from Language", use_column_width=True)
