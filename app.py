# app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
import zlib
import struct
import math

MAGIC = b"SIG1"  # Symbolic Image GENeration v1

# -------------------------
# Binary pack/unpack helpers
# -------------------------
def pack_symbolic_binary(width, height, tile_size, palette_bytes, tile_count, tiles_bytes):
    """
    Binary layout:
    [MAGIC:4][width:2][height:2][tile_size:1][palette_size:1][tile_count:4]
    [palette: palette_size*3 bytes]
    [tiles: tile_count * (tile_size*tile_size) bytes]  # each tile is palette indices
    Then the whole blob is compressed with zlib externally.
    """
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
    # tiles_bytes length should equal tile_count * tile_area
    return width, height, tile_size, palette_bytes, tile_count, tiles_bytes

# -------------------------
# Encoding / Decoding core
# -------------------------
def encode_image_to_symbolic_bytes(img: Image.Image, tile_size: int = 8, palette_size: int = 256):
    """
    Returns zlib-compressed binary blob representing the encoded symbolic image.
    """
    if palette_size < 2 or palette_size > 256:
        raise ValueError("palette_size must be in [2,256]")

    # Force RGB and pad/crop image to multiple of tile_size
    img = img.convert("RGB")
    w0, h0 = img.size
    w = (w0 // tile_size) * tile_size
    h = (h0 // tile_size) * tile_size
    if w != w0 or h != h0:
        img = img.crop((0, 0, w, h))

    # Quantize to palette_size using PIL adaptive palette
    pal_img = img.quantize(colors=palette_size, method=Image.MEDIANCUT)
    # Get palette (flat array of R,G,B)
    palette = pal_img.getpalette()[:palette_size*3]
    palette_bytes = bytes(palette)

    # Get indexed pixels (palette indices) as 2D array
    idx_arr = np.array(pal_img.convert("L"))  # single channel with palette indices 0..palette_size-1

    tiles = []
    tiles_bytes = bytearray()
    tile_area = tile_size * tile_size
    tiles_per_row = w // tile_size
    tiles_per_col = h // tile_size
    for ty in range(tiles_per_col):
        for tx in range(tiles_per_row):
            y0 = ty * tile_size
            x0 = tx * tile_size
            block = idx_arr[y0:y0+tile_size, x0:x0+tile_size]
            # Pack block row-major as bytes
            tiles_bytes.extend(block.astype(np.uint8).tobytes())

    tile_count = tiles_per_row * tiles_per_col
    binary = pack_symbolic_binary(w, h, tile_size, palette_bytes, tile_count, bytes(tiles_bytes))
    compressed = zlib.compress(binary, level=9)
    return compressed

def decode_symbolic_bytes_to_image(compressed_blob: bytes):
    decompressed = zlib.decompress(compressed_blob)
    width, height, tile_size, palette_bytes, tile_count, tiles_bytes = unpack_symbolic_binary(decompressed)

    palette_size = len(palette_bytes) // 3
    palette = [tuple(palette_bytes[i*3:(i+1)*3]) for i in range(palette_size)]

    tile_area = tile_size * tile_size
    expected_tiles_bytes_length = tile_count * tile_area
    if len(tiles_bytes) != expected_tiles_bytes_length:
        # tolerant: if mismatch, truncate or pad
        tiles_bytes = tiles_bytes[:expected_tiles_bytes_length].ljust(expected_tiles_bytes_length, b'\x00')

    # reconstruct indexed image
    tiles_per_row = width // tile_size
    tiles_per_col = height // tile_size
    idx_arr = np.zeros((height, width), dtype=np.uint8)

    offset = 0
    for ty in range(tiles_per_col):
        for tx in range(tiles_per_row):
            block_flat = tiles_bytes[offset:offset+tile_area]
            offset += tile_area
            block = np.frombuffer(block_flat, dtype=np.uint8).reshape((tile_size, tile_size))
            y0 = ty * tile_size
            x0 = tx * tile_size
            idx_arr[y0:y0+tile_size, x0:x0+tile_size] = block

    # convert indexed to RGB via palette
    rgb_arr = np.zeros((height, width, 3), dtype=np.uint8)
    for i, rgb in enumerate(palette):
        mask = (idx_arr == i)
        if mask.any():
            rgb_arr[mask] = rgb

    img = Image.fromarray(rgb_arr, mode="RGB")
    return img

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Compact Symbolic Image Codec", layout="wide")
st.title("Compact Symbolic Image Codec — Binary + zlib (tile+palette)")

st.markdown(
    """
    This app encodes images to a compact binary symbolic format (tile-based palette indices),
    compresses with zlib, and decodes back. Tune **tile size** and **palette size** to trade quality vs size.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.header("Encode")
    uploaded = st.file_uploader("Upload image to encode", type=["png","jpg","jpeg"])
    tile_size = st.selectbox("Tile size (px)", options=[4, 8, 16, 32], index=1)
    palette_size = st.selectbox("Palette size", options=[64, 128, 256], index=2)
    if uploaded and st.button("Encode & Compress"):
        img = Image.open(uploaded)
        try:
            compressed = encode_image_to_symbolic_bytes(img, tile_size=tile_size, palette_size=palette_size)
            st.success(f"Encoded & compressed: {len(compressed)} bytes")
            st.download_button(
                label="Download encoded .sig",
                data=compressed,
                file_name="image_symbolic.sig",
                mime="application/octet-stream"
            )
            # Show decoded preview
            decoded = decode_symbolic_bytes_to_image(compressed)
            st.image(decoded, caption="Decoded preview (from compressed blob)", use_column_width=True)
        except Exception as e:
            st.error(f"Encoding failed: {e}")

with col2:
    st.header("Decode")
    uploaded_bin = st.file_uploader("Upload .sig (encoded) file", type=["sig","bin"])
    if uploaded_bin and st.button("Decode & Download PNG"):
        try:
            blob = uploaded_bin.read()
            decoded = decode_symbolic_bytes_to_image(blob)
            st.image(decoded, caption="Decoded Image", use_column_width=True)
            buf = io.BytesIO()
            decoded.save(buf, format="PNG", optimize=True)
            png_bytes = buf.getvalue()
            st.download_button("Download decoded PNG", data=png_bytes, file_name="decoded.png", mime="image/png")
            st.success(f"Decoded PNG size: {len(png_bytes)} bytes")
        except Exception as e:
            st.error(f"Decoding failed: {e}")

st.markdown("---")
st.markdown("**Notes:**\n\n"
            "- Use smaller tile size (4 or 8) + larger palette (256) for best fidelity.\n"
            "- Use larger tile size + smaller palette for smallest encoded size.\n"
            "- This is lossy (color quantization) but much smaller and practical. "
            "If you want near-lossless, use tile_size=4 and palette_size=256.")
