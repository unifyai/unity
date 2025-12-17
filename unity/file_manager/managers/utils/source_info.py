from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from unity.file_manager.file_parsers.types.contracts import FileParseTrace
from unity.file_manager.types.filesystem import FileReference


@dataclass(frozen=True)
class SourceInfo:
    """Best-effort file identity metadata used for FileRecords rows."""

    size_bytes: Optional[int] = None
    created_at: Optional[str] = None
    modified_at: Optional[str] = None


def _try_stat(path: str | None) -> SourceInfo:
    """Best-effort local stat fallback (never raises)."""
    if not path:
        return SourceInfo()
    try:
        p = Path(path).expanduser().resolve()
        st = p.stat()
        # We intentionally keep created_at/modified_at as Optional[str] ISO placeholders here.
        # Many platforms do not provide a true creation time; we avoid inventing semantics.
        return SourceInfo(size_bytes=int(st.st_size))
    except Exception:
        return SourceInfo()


def source_info_for_file(
    *,
    adapter_ref: Optional[FileReference] = None,
    trace: Optional[FileParseTrace] = None,
) -> SourceInfo:
    """
    Resolve best-effort file metadata for FileRecords rows.

    Priority order
    --------------
    1) Adapter-provided metadata (`FileReference`)
    2) Local filesystem stat from parse trace paths (if available)
    """
    if adapter_ref is not None:
        try:
            return SourceInfo(
                size_bytes=adapter_ref.size_bytes,
                created_at=None,
                modified_at=adapter_ref.modified_at,
            )
        except Exception:
            # Fall back to stat
            pass

    local_path = None
    if trace is not None:
        local_path = trace.parsed_local_path or trace.source_local_path
    return _try_stat(local_path)
