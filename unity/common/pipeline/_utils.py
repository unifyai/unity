"""Shared pipeline utilities.

Small helpers used by multiple pipeline submodules. Kept private (underscore
prefix) to signal that callers outside the ``pipeline`` package should not
import from here directly.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from pydantic import BaseModel


def utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(UTC)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return utc_now().isoformat()


class JsonlWriter:
    """Thread-safe append-only JSONL writer.

    Used by ``JsonlRunLedger`` and ``JsonlCostLedger`` to avoid duplicating
    the same file-handle + lock boilerplate.
    """

    def __init__(self, *, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()

    def write_model(self, model: BaseModel) -> None:
        payload = model.model_dump(mode="json", exclude_none=True)
        with self._lock:
            self._fh.write(json.dumps(payload, ensure_ascii=False))
            self._fh.write("\n")

    def flush(self) -> None:
        with self._lock:
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.flush()
            finally:
                self._fh.close()
