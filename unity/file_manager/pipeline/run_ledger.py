from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class PipelineStageManifest(BaseModel):
    """Typed manifest for one pipeline stage outcome."""

    record_type: Literal["stage"] = "stage"
    run_id: str
    file_path: str
    stage_name: str
    status: Literal["success", "error"]
    duration_ms: float = 0.0
    retries_used: int = 0
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    recorded_at: str = Field(default_factory=lambda: _utc_now().isoformat())


class PipelineFileManifest(BaseModel):
    """Typed manifest for one file-level pipeline outcome."""

    record_type: Literal["file"] = "file"
    run_id: str
    file_path: str
    status: Literal["success", "error"]
    total_duration_ms: float = 0.0
    retries_used: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)
    recorded_at: str = Field(default_factory=lambda: _utc_now().isoformat())


class PipelineRunManifest(BaseModel):
    """Typed manifest for a run-level pipeline lifecycle record."""

    record_type: Literal["run"] = "run"
    run_id: str
    status: Literal["started", "completed"]
    file_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    parallel_files: bool = False
    total_duration_ms: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)
    recorded_at: str = Field(default_factory=lambda: _utc_now().isoformat())


class RunLedger(Protocol):
    """Port for emitting typed run/file/stage manifests."""

    def write(self, manifest: BaseModel) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class JsonlRunLedger:
    """Thread-safe JSONL ledger writer for local pipeline manifests."""

    def __init__(self, *, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()

    def write(self, manifest: BaseModel) -> None:
        payload = manifest.model_dump(mode="json", exclude_none=True)
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


def generate_run_ledger_path() -> str:
    """Return a timestamped local path for run-manifest output."""

    stamp = _utc_now().strftime("%Y%m%d_%H%M%S")
    path = Path("logs/file_manager_runs") / f"run_ledger_{stamp}.jsonl"
    return str(path.resolve())


def _utc_now() -> datetime:
    return datetime.now(UTC)
