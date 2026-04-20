"""File classification for adaptive batch scheduling.

When subprocess isolation is enabled, the batch scheduler splits files
into two lanes:

- **heavy** — serialised (one subprocess at a time).  A file is heavy
  when it uses a Docling backend (PDF, DOCX, DOC, HTML, XML, JSON —
  these load 500 MB+ of ONNX models regardless of file size) **or**
  exceeds ``LARGE_FILE_THRESHOLD_BYTES`` (any format).
- **light** — everything else.  Runs concurrently up to
  ``ParseConfig.max_concurrent_parses``.

If a light file still OOMs, the crash-isolation model ensures only
that file is affected, and the batch scheduler retries it once in
isolation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, FrozenSet

if TYPE_CHECKING:
    from unity.file_manager.file_parsers.types.contracts import FileParseRequest

DOCLING_FORMATS: FrozenSet[str] = frozenset(
    {
        ".pdf",
        ".docx",
        ".doc",
        ".html",
        ".xml",
        ".json",
    },
)

LARGE_FILE_THRESHOLD_BYTES: int = 100 * 1024 * 1024  # 100 MB


def _uses_docling(request: "FileParseRequest") -> bool:
    """Return True if the file's format routes to a Docling backend."""
    ext = Path(request.source_local_path).suffix.lower()
    return ext in DOCLING_FORMATS


def _is_large_file(request: "FileParseRequest") -> bool:
    """Return True if the file exceeds the large-file threshold."""
    try:
        return os.path.getsize(request.source_local_path) >= LARGE_FILE_THRESHOLD_BYTES
    except OSError:
        return False


def classify_file(request: "FileParseRequest") -> str:
    """Classify a parse request as ``"heavy"`` or ``"light"``.

    A file is **heavy** (serialised, one at a time) when:

    1. Its format routes to a Docling backend (PDF, DOCX, DOC, HTML,
       XML, JSON) — these load 500 MB+ of ONNX models regardless of
       file size.
    2. Its size exceeds ``LARGE_FILE_THRESHOLD_BYTES`` (100 MB) — even
       non-Docling formats can exhaust memory when large enough (e.g.,
       a 500 MB CSV exploded into a DataFrame).

    Everything else is **light** (concurrent up to
    ``max_concurrent_parses``).
    """
    if _uses_docling(request):
        return "heavy"
    if _is_large_file(request):
        return "heavy"
    return "light"


_COST_EXPANSION_FACTOR = 300.0


def estimate_peak_memory_bytes(request: "FileParseRequest") -> int:
    """Rough size-based memory estimate for **cost/metrics reporting only**.

    Not used for scheduling decisions.  Exists solely so cost-ledger
    line items can record an ``estimated_peak_bytes`` figure.
    """
    try:
        file_size = os.path.getsize(request.source_local_path)
    except OSError:
        return 0
    return int(file_size * _COST_EXPANSION_FACTOR)


def fmt_bytes(n: int) -> str:
    """Format a byte count into a human-readable string (e.g. ``'4.2 GB'``)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n} PB"
