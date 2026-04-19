from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Protocol

from .types import InlineRowsHandle, ObjectStoreArtifactHandle, TableInputHandle

#: Conventional ``table_id`` used when materialising lowered content rows
#: (``/Content/`` rows derived from the document graph) as an
#: ``ObjectStoreArtifactHandle``.  Keeping this constant here means callers
#: never have to coin a synthetic id for the content artifact.
CONTENT_ROWS_TABLE_ID: str = "__content__"


class ArtifactStore(Protocol):
    """Port for durable artifact storage (table data AND JSON manifests).

    Implementations must support two concerns:

    1. **Table materialisation** -- serialise a ``TableInputHandle`` into a
       durable artifact (JSONL today, Parquet/Arrow later).
    2. **Manifest CRUD** -- store and retrieve arbitrary JSON documents
       (run manifests, cost ledgers, bundle descriptors) keyed by a
       logical path.

    The local implementation uses the filesystem.  A future GCS
    implementation will target ``gs://`` URIs with the same interface.
    """

    def materialize_table_input(
        self,
        handle: TableInputHandle,
        *,
        logical_path: str,
        table_id: str,
        artifact_format: str,
    ) -> ObjectStoreArtifactHandle: ...

    def materialize_content_rows(
        self,
        rows: Iterable[Any],
        *,
        logical_path: str,
        artifact_format: str = "jsonl",
    ) -> ObjectStoreArtifactHandle:
        """Serialise lowered content rows into a JSONL artifact handle.

        ``rows`` may be Pydantic models (e.g. ``FileContentRow``) or plain
        ``dict`` payloads; each is normalised to a JSON object before being
        written.  The resulting handle uses the conventional
        ``CONTENT_ROWS_TABLE_ID`` so manifests/handles for derived content
        stay consistent across implementations.
        """
        ...

    def put_json(self, key: str, data: Any) -> str:
        """Serialise *data* as JSON and persist under *key*.

        Returns the storage URI of the written object.
        """
        ...

    def get_json(self, key: str) -> Any:
        """Read and deserialise a JSON object previously stored at *key*."""
        ...

    def exists(self, key: str) -> bool:
        """Return ``True`` if an object exists at *key*."""
        ...

    def delete(self, key: str) -> None:
        """Remove the object at *key* (no-op if absent)."""
        ...


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
        from unity.common.pipeline.row_streaming import iter_table_input_rows

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

    def materialize_content_rows(
        self,
        rows: Iterable[Any],
        *,
        logical_path: str,
        artifact_format: str = "jsonl",
    ) -> ObjectStoreArtifactHandle:
        """Serialise content rows as JSONL via ``materialize_table_input``.

        Mirrors the table materialisation flow but fixes the ``table_id`` to
        :data:`CONTENT_ROWS_TABLE_ID`.  Rows may be Pydantic models or
        dicts; non-dict inputs are coerced via ``model_dump(mode="json")``
        when available and otherwise wrapped into a single-field dict as a
        last resort.
        """
        serialised: list[dict[str, Any]] = []
        columns: list[str] = []
        for row in rows:
            payload: dict[str, Any]
            dump = getattr(row, "model_dump", None)
            if callable(dump):
                payload = dict(dump(mode="json", exclude_none=True))
            elif isinstance(row, dict):
                payload = {str(k): v for k, v in row.items()}
            else:
                payload = {"value": row}
            serialised.append(payload)
            if not columns:
                columns = [str(k) for k in payload.keys()]

        inline = InlineRowsHandle(
            rows=serialised,
            columns=columns,
            row_count=len(serialised),
        )
        return self.materialize_table_input(
            inline,
            logical_path=logical_path,
            table_id=CONTENT_ROWS_TABLE_ID,
            artifact_format=artifact_format,
        )

    # -- manifest CRUD -----------------------------------------------------

    def put_json(self, key: str, data: Any) -> str:
        target = self._key_path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(data, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return target.resolve().as_uri()

    def get_json(self, key: str) -> Any:
        target = self._key_path(key)
        if not target.exists():
            raise FileNotFoundError(f"Artifact not found: {key}")
        return json.loads(target.read_text(encoding="utf-8"))

    def exists(self, key: str) -> bool:
        return self._key_path(key).exists()

    def delete(self, key: str) -> None:
        target = self._key_path(key)
        if target.exists():
            target.unlink()

    # -- internal helpers ---------------------------------------------------

    def _key_path(self, key: str) -> Path:
        safe = key.lstrip("/")
        return self.root_dir / safe

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
