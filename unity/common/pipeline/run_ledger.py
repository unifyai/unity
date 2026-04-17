from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from ._utils import JsonlWriter, utc_now_iso, utc_now


class PipelineStageManifest(BaseModel):
    """Typed manifest for one pipeline stage outcome."""

    record_type: Literal["stage"] = "stage"
    run_id: str
    stage_id: str | None = None
    file_path: str
    file_id: int | None = None
    storage_id: str | None = None
    table_id: str | None = None
    stage_name: str
    status: Literal["success", "error"]
    duration_ms: float = 0.0
    retries_used: int = 0
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    recorded_at: str = Field(default_factory=utc_now_iso)


class PipelineFileManifest(BaseModel):
    """Typed manifest for one file-level pipeline outcome."""

    record_type: Literal["file"] = "file"
    run_id: str
    file_path: str
    file_id: int | None = None
    storage_id: str | None = None
    status: Literal["success", "error"]
    total_duration_ms: float = 0.0
    retries_used: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)
    recorded_at: str = Field(default_factory=utc_now_iso)


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
    recorded_at: str = Field(default_factory=utc_now_iso)


class RunLedger(Protocol):
    """Port for emitting typed run/file/stage manifests."""

    def write(self, manifest: BaseModel) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class JsonlRunLedger:
    """Thread-safe JSONL ledger writer for local pipeline manifests."""

    def __init__(self, *, path: str | Path):
        self._writer = JsonlWriter(path=path)
        self.path = self._writer.path

    def write(self, manifest: BaseModel) -> None:
        self._writer.write_model(manifest)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def generate_run_ledger_path() -> str:
    """Return a timestamped local path for run-manifest output."""

    stamp = utc_now().strftime("%Y%m%d_%H%M%S")
    path = Path("logs/file_manager_runs") / f"run_ledger_{stamp}.jsonl"
    return str(path.resolve())
