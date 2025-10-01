from __future__ import annotations

import base64
import struct
import zlib
from typing import Tuple


def make_solid_png_base64(width: int, height: int, rgb: Tuple[int, int, int]) -> str:
    """
    Return a Base64-encoded PNG for a solid-color image of the given size.

    Implementation is dependency-free (uses struct + zlib) and suitable for tests
    that need a deterministic, valid PNG without external assets.
    """
    r, g, b = rgb

    # PNG signature
    signature = b"\x89PNG\r\n\x1a\n"

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    # IHDR: width, height, bit depth 8, color type 2 (truecolor), no compression/filter/interlace
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    # Raw image data: each row starts with filter byte 0, followed by RGB triples
    scanline = bytes([0]) + bytes([r, g, b]) * width
    raw = b"".join(scanline for _ in range(height))
    idat = zlib.compress(raw, level=9)

    png = (
        signature + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def substring_from_span(content: str, span: str) -> str:
    """Return the best‑effort substring of ``content`` for a slice key "[x:y]".

    Rules implemented to match manager behaviour:
    - Supports negative and open‑ended indices.
    - Indices are clamped to the valid range [0, len(content)].
    - If start > end, indices are swapped to produce a forward slice.
    - On malformed input, returns an empty string.
    """
    try:
        import re as _re

        m = _re.fullmatch(r"\[\s*(-?\d+)?\s*:\s*(-?\d+)?\s*\]", str(span))
        if not m:
            return ""
        start_s, end_s = m.group(1), m.group(2)
        L = len(content)

        if start_s is None:
            start = 0
        else:
            start = int(start_s)
            if start < 0:
                start = max(0, L + start)
            else:
                start = min(L, start)

        if end_s is None:
            end = L
        else:
            end = int(end_s)
            if end < 0:
                end = max(0, L + end)
            else:
                end = min(L, end)

        if start > end:
            start, end = end, start

        return content[start:end]
    except Exception:
        return ""
