# app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
import zlib
import struct

MAGIC = b"SIG1"  # file signature

# -------------------------
# Binary helpers
# -------------------------
def pack_symbolic_binary(width, height, tile_size, palette_bytes, tile_count, tiles_bytes):
    palette_size = len(palette_bytes) // 3
    header = struct.pack(">4sHHBBI", MAGIC, width, height, tile_size, palette_size, tile_count)
    return header + palette_bytes + tiles_bytes

def unpack_symbolic_binary(blob):
    header_size = struct.calcsize(">4sHHBBI")
    header = blob[:header_size]
    magic, width, height, tile_size, palette_size, tile_count = struct.unpack(">4sHHBBI", header)
    if magic != MAGIC:
        raise ValueError("Invalid file magic")
    offset = header_size
    palette_bytes = blob[offset:offset + palette_size*3]
    offset += palette_size*3
    tiles_bytes = blob[offset:]
    return width, height, tile_size, palette_bytes, tile_count, tiles_bytes

# -------------------------
# Encoding / Decoding
# -------------------------
def encode_image_to_symbolic_bytes(img: Image.Image, tile_size: int = 8, palette_size: int = 256):
    img = img.convert("RGB")
    w0, h0 = img.size
    w = (w0 // tile_size) * tile_size
    h = (h0 // tile_size) * tile_size
    if w != w0 or h != h0:
        img = img.crop((0, 0, w, h))

    # quantize
    pal_img = img.quantize(colors=palette_size, method=Image.MEDIANCUT)

    # sanitize palette
    raw_palette = pal_img.getpalette()
    if raw_palette is None:
        raise ValueError("Palette not found")
    raw_palette = raw_palette[:palette_size*3]
    palette_bytes = bytes([min(255, max(0, int(v))) for v in raw_palette])

    # indexed pixels
    idx_arr = np.array(pal_img)

    tiles_bytes = bytearray()
    tiles_per_row = w // tile_size
    tiles_per_col = h // tile_size
    tile_area = tile_size * tile_size
    for ty in range(tiles_per_col):
        for tx in range(tiles_per_row):
            y0 = ty * tile_size
            x0 = tx * tile_size
            block = idx_arr[y0:y0+tile_size, x0:x0+tile_size]
            tiles_bytes.extend(block.astype(np.uint8).tobytes())

    tile_count = tiles_per_row * tiles_per_col
    binary = pack_symbolic_binary(w, h, tile_size, palette_bytes, tile_count, bytes(tiles_bytes))
    return zlib.compress(binary, level=9)

def decode_symbolic_bytes_to_image(compressed_blob: bytes):
    decompressed = zlib.decompress(compressed_blob)
    width, height, tile_size, palette_bytes, tile_count, tiles_bytes = unpack_symbolic_binary(decompressed)

    palette_size = len(palette_bytes) // 3
    palette = [tuple(palette_bytes[i*3:(i+1)*3]) for i in range(palette_size)]

    tile_area = tile_size * tile_size
    tiles_per_row = width // tile_size
    tiles_per_col = height // tile_size

    idx_arr = np.zeros((height, width), dtype=np.uint8)
    offset = 0
    for ty in range(tiles_per_col):
        for tx in range(tiles_per_row):
            block_flat = tiles_bytes[offset:offset+tile_area]
            offset += tile_area
            block = np.frombuffer(block_flat, dtype=np.uint8).reshape((tile_size, tile_size))
            idx_arr[ty*tile_size:(ty+1)*tile_size, tx*tile_size:(tx+1)*tile_size] = block

    rgb_arr = np.zeros((height, width, 3), dtype=np.uint8)
    for i, rgb in enumerate(palette):
        mask = (idx_arr == i)
        if mask.any():
            rgb_arr[mask] = rgb

    return Image.fromarray(rgb_arr, mode="RGB")

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Compact Symbolic Codec", layout="wide")
st.title("Compact Symbolic Image Codec")

col1, col2 = st.columns(2)

with col1:
    st.header("Encode")
    uploaded = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
    tile_size = st.selectbox("Tile size", options=[4, 8, 16, 32], index=1)
    palette_size = st.selectbox("Palette size", options=[64, 128, 256], index=2)
    if uploaded and st.button("Encode"):
        try:
            img = Image.open(uploaded)
            compressed = encode_image_to_symbolic_bytes(img, tile_size, palette_size)
            st.success(f"Encoded size: {len(compressed)} bytes")
            st.download_button("Download .sig", compressed, "image.sig", "application/octet-stream")
            st.image(decode_symbolic_bytes_to_image(compressed), caption="Decoded Preview", use_column_width=True)
        except Exception as e:
            st.error(str(e))

with col2:
    st.header("Decode")
    uploaded_bin = st.file_uploader("Upload .sig", type=["sig","bin"])
    if uploaded_bin and st.button("Decode"):
        try:
            blob = uploaded_bin.read()
            decoded = decode_symbolic_bytes_to_image(blob)
            st.image(decoded, caption="Decoded Image", use_column_width=True)
            buf = io.BytesIO()
            decoded.save(buf, format="PNG", optimize=True)
            st.download_button("Download PNG", buf.getvalue(), "decoded.png", "image/png")
        except Exception as e:
            st.error(str(e))
