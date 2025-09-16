# app.py
import streamlit as st
from PIL import Image
import numpy as np
import io, zlib, binascii, math
import struct
from typing import List, Tuple

# -------------------------
# Utilities
# -------------------------
MAGIC = b"SYM1"  # 4 bytes signature

def psnr(orig: np.ndarray, recon: np.ndarray):
    mse = np.mean((orig.astype(np.float32) - recon.astype(np.float32))**2)
    if mse == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# -------------------------
# Core Encoding / Decoding
# -------------------------
def image_to_indexed(img: Image.Image, size: int, palette_size: int) -> Tuple[np.ndarray, List[Tuple[int,int,int]]]:
    """Resize -> quantize -> return index grid and palette list"""
    img = img.convert("RGB").resize((size, size), Image.LANCZOS)
    pal_img = img.quantize(colors=palette_size, method=Image.MEDIANCUT)
    # palette is flat list of R,G,B repeating
    raw_palette = pal_img.getpalette()[:palette_size*3]
    palette = [(raw_palette[i], raw_palette[i+1], raw_palette[i+2]) for i in range(0, len(raw_palette), 3)]
    idx = np.array(pal_img, dtype=np.uint8)  # shape (size,size) indices 0..palette_size-1
    return idx, palette

def rle_encode_indices(idx: np.ndarray) -> bytes:
    """Simple RLE across rows: emit (value, length) pairs as bytes.
       value: 1 byte (0..255), length: 2 bytes unsigned big-endian (0..65535).
       If run >65535, split across multiple runs."""
    h, w = idx.shape
    out = bytearray()
    for y in range(h):
        row = idx[y]
        cur = int(row[0])
        run = 1
        for v in row[1:]:
            v = int(v)
            if v == cur and run < 65535:
                run += 1
            else:
                out.append(cur & 0xFF)
                out.extend(struct.pack(">H", run))
                cur = v
                run = 1
        # flush row end
        out.append(cur & 0xFF)
        out.extend(struct.pack(">H", run))
        # row separator marker: value 255, length 0 as sentinel? Not necessary because we sum widths, but keep structure consistent.
    return bytes(out)

def rle_decode_indices(rle_bytes: bytes, size: int) -> np.ndarray:
    """Decode the RLE bytes back into (size,size) index grid."""
    h = size
    w = size
    arr = np.zeros((h, w), dtype=np.uint8)
    ptr = 0
    y = 0
    x = 0
    total_len = len(rle_bytes)
    while ptr + 3 <= total_len and y < h:
        val = rle_bytes[ptr]; ptr += 1
        run = struct.unpack(">H", rle_bytes[ptr:ptr+2])[0]; ptr += 2
        while run > 0:
            take = min(run, w - x)
            arr[y, x:x+take] = val
            x += take
            run -= take
            if x >= w:
                x = 0
                y += 1
                if y >= h and run > 0:
                    # leftover; ignore - malformed
                    break
    return arr

def pack_symbolic(size: int, palette: List[Tuple[int,int,int]], rle_bytes: bytes) -> bytes:
    """Binary layout:
       MAGIC(4) | size:2 | palette_len:1 | palette_bytes (3*len) | rle_len:4 | rle_bytes
    """
    palette_len = len(palette)
    header = MAGIC + struct.pack(">H B", size, palette_len)
    palette_flat = bytearray()
    for (r,g,b) in palette:
        palette_flat.extend([int(r)&0xFF, int(g)&0xFF, int(b)&0xFF])
    rle_len = len(rle_bytes)
    header2 = struct.pack(">I", rle_len)
    return header + bytes(palette_flat) + header2 + rle_bytes

def unpack_symbolic(blob: bytes) -> Tuple[int, List[Tuple[int,int,int]], bytes]:
    # MAGIC
    if len(blob) < 4:
        raise ValueError("Blob too short")
    magic = blob[:4]
    if magic != MAGIC:
        raise ValueError("Invalid symbolic file")
    ptr = 4
    size = struct.unpack(">H", blob[ptr:ptr+2])[0]; ptr += 2
    palette_len = blob[ptr]; ptr += 1
    palette = []
    pal_bytes_needed = palette_len * 3
    pal_flat = blob[ptr:ptr+pal_bytes_needed]; ptr += pal_bytes_needed
    for i in range(palette_len):
        r = pal_flat[i*3]; g = pal_flat[i*3+1]; b = pal_flat[i*3+2]
        palette.append((r,g,b))
    rle_len = struct.unpack(">I", blob[ptr:ptr+4])[0]; ptr += 4
    rle_bytes = blob[ptr:ptr+rle_len]
    return size, palette, rle_bytes

# high-level encode/decode pipeline
def encode_image_to_symbolic_printable(img: Image.Image, size: int = 32, palette_size: int = 16) -> str:
    idx, palette = image_to_indexed(img, size=size, palette_size=palette_size)
    rle = rle_encode_indices(idx)
    packed = pack_symbolic(size, palette, rle)
    compressed = zlib.compress(packed, level=9)
    # ascii85 encode (binascii.a85encode) yields printable, more compact than base64
    printable = binascii.a85encode(compressed)  # bytes
    return printable.decode('ascii')

def decode_symbolic_printable_to_image(sym_text: str) -> Image.Image:
    compressed = binascii.a85decode(sym_text.encode('ascii'))
    packed = zlib.decompress(compressed)
    size, palette, rle_bytes = unpack_symbolic(packed)
    idx = rle_decode_indices(rle_bytes, size=size)
    # reconstruct RGB
    h, w = idx.shape
    rgb = np.zeros((h,w,3), dtype=np.uint8)
    for i, col in enumerate(palette):
        mask = (idx == i)
        if mask.any():
            rgb[mask] = col
    return Image.fromarray(rgb, 'RGB')

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Symbolic Image Codec (palette+RLE+a85)", layout="wide")
st.title("Prototype: palette → RLE → zlib → Ascii85 (compact printable symbolic)")

col1, col2 = st.columns([1,1])
with col1:
    st.header("Encode")
    upload = st.file_uploader("Upload image to encode", type=["png","jpg","jpeg"])
    size = st.selectbox("Target size (sx × sx)", options=[16, 24, 32, 48], index=2)
    palette_size = st.selectbox("Palette size (colors)", options=[8, 12, 16, 32], index=2)
    if upload and st.button("Encode"):
        img = Image.open(upload).convert("RGB")
        # keep a copy of original resized for psnr
        resized = img.resize((size,size), Image.LANCZOS)
        sym_text = encode_image_to_symbolic_printable(img, size=size, palette_size=palette_size)
        # compute sizes
        orig_buf = io.BytesIO(); resized.save(orig_buf, format="PNG"); orig_bytes = orig_buf.getvalue()
        sym_bytes = sym_text.encode('ascii')
        st.success(f"Symbolic length (chars): {len(sym_text)}  —  encoded bytes: {len(sym_bytes)}")
        st.download_button("Download symbolic (.sym)", data=sym_bytes, file_name="image.sym", mime="text/plain")
        # decode for preview and PSNR
        recon = decode_symbolic_printable_to_image(sym_text)
        st.image(resized, caption="Resized reference (target)", use_column_width=False)
        st.image(recon, caption="Reconstruction preview", use_column_width=False)
        # quality
        orig_arr = np.array(resized)
        recon_arr = np.array(recon)
        p = psnr(orig_arr, recon_arr)
        st.write(f"PSNR (resized vs recon): {p:.2f} dB")
        st.write(f"Compression ratio (resized PNG bytes / symbolic bytes): {len(orig_bytes)/len(sym_bytes):.2f}x")

with col2:
    st.header("Decode")
    sym_upload = st.file_uploader("Upload .sym (text) to decode", type=["sym","txt"])
    if sym_upload and st.button("Decode & Download PNG"):
        sym_text = sym_upload.read().decode('ascii')
        try:
            recon = decode_symbolic_printable_to_image(sym_text)
            st.image(recon, caption="Decoded image", use_column_width=True)
            buf = io.BytesIO()
            recon.save(buf, format="PNG", optimize=True)
            png_bytes = buf.getvalue()
            st.download_button("Download decoded PNG", data=png_bytes, file_name="decoded.png", mime="image/png")
            st.success(f"Decoded PNG size: {len(png_bytes)} bytes")
        except Exception as e:
            st.error(f"Decoding failed: {e}")

st.markdown("---")
st.markdown("Notes:\n- This is lossy (quantization to palette). Increase target size or palette to improve fidelity.\n- For best compactness, small target size (16–32) and palette 8–16 are effective.\n- Ascii85 is printable and smaller overhead than base64.")
