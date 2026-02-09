from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _safe_stem_from_snapshot(snapshot: Dict[str, Any]) -> str:
    """Return a filesystem-friendly base name for a snapshot file.

    Preference order:
      1) loop_id (if provided),
      2) manager entrypoint "Class.method",
      3) the generic stem "loop".
    """

    def _sanitize(text: str) -> str:
        # Remove characters that are commonly problematic in filenames
        bad = '<>:"/\\|?*\n\r\t'
        out = "".join(ch for ch in str(text) if ch not in bad)
        # Collapse spaces
        out = " ".join(out.split())
        # Replace spaces with underscores for portability
        return out.replace(" ", "_") or "loop"

    try:
        if isinstance(snapshot, dict):
            # 1) loop_id
            loop_id = snapshot.get("loop_id")
            if isinstance(loop_id, str) and loop_id.strip():
                return _sanitize(loop_id)

            # 2) manager entrypoint
            ep = snapshot.get("entrypoint") or {}
            if isinstance(ep, dict) and ep.get("type") == "manager_method":
                cls = ep.get("class_name") or "Manager"
                meth = ep.get("method_name") or "method"
                return _sanitize(f"{cls}.{meth}")
    except Exception:
        pass

    return "loop"


def default_store(
    snapshot: Dict[str, Any],
    *,
    base_dir: Optional[str | Path] = None,
    filename: Optional[str] = None,
) -> str:
    """Persist a loop snapshot as JSON and return the file path.

    Parameters
    ----------
    snapshot : dict
        The snapshot payload to persist.
    base_dir : str | Path | None, optional
        Directory to write into. Defaults to a sibling "snapshots" directory
        under the current working directory.
    filename : str | None, optional
        Desired file name (e.g. "ContactManager.ask.json"). When omitted, a
        name is derived from the snapshot (loop_id or entrypoint) with a
        numeric suffix when a collision is detected.
    """

    # Resolve base directory
    root = Path(base_dir) if base_dir is not None else Path.home() / "Unity" / "Local" / "snapshots"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Best-effort; will raise on write below if not writable
        pass

    # Determine file name
    stem = filename or f"{_safe_stem_from_snapshot(snapshot)}.json"
    if not stem.lower().endswith(".json"):
        stem = f"{stem}.json"

    path = root / stem

    # De-duplicate if needed by adding an incrementing numeric suffix
    if path.exists():
        base = path.stem
        suffix = 1
        while path.exists():
            path = root / f"{base}.{suffix}.json"
            suffix += 1

    # Write atomically via a temp file then rename
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

    return str(path)


def default_loader(path: str | Path) -> Dict[str, Any]:
    """Load and return a loop snapshot JSON from ``path``.

    Accepts relative or absolute paths. Returns the parsed dict or raises on
    I/O / JSON errors.
    """

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


__all__ = (
    "default_store",
    "default_loader",
)
