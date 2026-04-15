from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from .row_streaming import iter_table_input_rows
from .types import ObjectStoreArtifactHandle, TableInputHandle


class ArtifactStore(Protocol):
    """Port for materializing table transports into durable artifacts."""

    def materialize_table_input(
        self,
        handle: TableInputHandle,
        *,
        logical_path: str,
        table_id: str,
        artifact_format: str,
    ) -> ObjectStoreArtifactHandle: ...


class LocalArtifactStore:
    """Filesystem-backed artifact store for local development and tests."""

    def __init__(self, *, root_dir: str | Path):
        self.root_dir = Path(root_dir).expanduser().resolve()

    def materialize_table_input(
        self,
        handle: TableInputHandle,
        *,
        logical_path: str,
        table_id: str,
        artifact_format: str,
    ) -> ObjectStoreArtifactHandle:
        if isinstance(handle, ObjectStoreArtifactHandle):
            return handle
        if artifact_format != "jsonl":
            raise ValueError(
                f"Unsupported artifact format for LocalArtifactStore: {artifact_format!r}",
            )

        target_path = self._artifact_path(
            logical_path=logical_path,
            table_id=table_id,
            artifact_format=artifact_format,
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)

        columns = list(getattr(handle, "columns", []) or [])
        actual_row_count = 0
        with target_path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in iter_table_input_rows(handle):
                payload = {str(key): value for key, value in dict(row).items()}
                if not columns:
                    columns = [str(key) for key in payload.keys()]
                fh.write(json.dumps(payload, ensure_ascii=False))
                fh.write("\n")
                actual_row_count += 1

        return ObjectStoreArtifactHandle(
            storage_uri=target_path.resolve().as_uri(),
            logical_path=str(logical_path or ""),
            artifact_format="jsonl",
            columns=columns,
            row_count=actual_row_count,
        )

    def _artifact_path(
        self,
        *,
        logical_path: str,
        table_id: str,
        artifact_format: str,
    ) -> Path:
        digest = hashlib.sha256(
            f"{logical_path}::{table_id}".encode("utf-8"),
        ).hexdigest()[:12]
        file_slug = _safe_fragment(Path(str(logical_path or "file")).stem or "file")
        table_slug = _safe_fragment(table_id or "table")
        return self.root_dir / file_slug / f"{table_slug}-{digest}.{artifact_format}"


def _safe_fragment(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "artifact"
    return "".join(
        char if char.isalnum() or char in ("-", "_") else "_" for char in text
    )
