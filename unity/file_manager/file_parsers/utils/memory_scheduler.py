"""Memory-bounded file classification for adaptive batch scheduling.

When subprocess isolation is enabled, the batch scheduler needs to decide
which files can be parsed concurrently (light) and which must be serialised
one at a time (heavy) to avoid exceeding available memory.

Classification is **format-agnostic** — any file of any format can be heavy
if its on-disk size, multiplied by a configurable expansion factor, exceeds
a configurable percentage of total system RAM.

All thresholds are expressed as fractions of system memory (via
``psutil.virtual_memory().total``) so they adapt automatically across local
machines, CI runners, and cloud containers with varying memory.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from unity.file_manager.file_parsers.types.contracts import FileParseRequest
    from unity.file_manager.types.config import ParseConfig


def get_system_memory_bytes() -> int:
    """Total physical RAM on the host, in bytes."""
    return psutil.virtual_memory().total


def classify_file(
    request: "FileParseRequest",
    *,
    sys_memory: int,
    config: "ParseConfig",
) -> str:
    """Classify a parse request as ``"heavy"`` or ``"light"``.

    A file is *heavy* when::

        file_size_on_disk  *  config.expansion_factor
            >  sys_memory  *  config.heavy_file_memory_pct

    For example, on a 16 GB system with the defaults (expansion_factor=300,
    heavy_file_memory_pct=0.25):

    - 15 MB file → estimated 4.4 GB peak → exceeds 4 GB threshold → **heavy**
    - 1 MB file  → estimated 300 MB peak → under threshold          → **light**

    If the file cannot be stat'd (e.g. path doesn't exist yet) it is
    conservatively treated as light.
    """
    try:
        file_size = os.path.getsize(request.source_local_path)
    except OSError:
        return "light"
    estimated_peak = estimate_peak_memory_bytes(request, config=config)
    threshold = sys_memory * config.heavy_file_memory_pct
    return "heavy" if estimated_peak > threshold else "light"


def estimate_peak_memory_bytes(
    request: "FileParseRequest",
    *,
    config: "ParseConfig",
) -> int:
    """Estimate peak parse memory for a request using on-disk size expansion."""

    try:
        file_size = os.path.getsize(request.source_local_path)
    except OSError:
        return 0
    return int(file_size * config.expansion_factor)


def fmt_bytes(n: int) -> str:
    """Format a byte count into a human-readable string (e.g. ``'4.2 GB'``)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n} PB"
