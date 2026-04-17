"""Row iteration for ``TableInputHandle`` variants.

This module is a thin dispatcher: each handle type delegates to the
native backend that knows how to stream its format efficiently
(Polars for CSV, openpyxl read-only for XLSX).  No file I/O logic
lives here -- it all lives in the backends.

All handle types are defined in ``unity.common.pipeline.types``, so
this module has **no dependency on FileManager** -- backend imports
are lazy so consumers that only use ``InlineRowsHandle`` or
``ObjectStoreArtifactHandle`` never trigger parser imports.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Dict
from urllib.parse import unquote, urlparse

from .types import (
    CsvFileHandle,
    InlineRowsHandle,
    ObjectStoreArtifactHandle,
    TableInputHandle,
    XlsxSheetHandle,
)

JsonObject = Dict[str, object]


def iter_table_input_rows(handle: TableInputHandle) -> Iterator[JsonObject]:
    """Yield table rows from any supported transport handle."""

    if isinstance(handle, InlineRowsHandle):
        yield from (dict(row) for row in handle.rows)
        return

    if isinstance(handle, CsvFileHandle):
        from unity.file_manager.file_parsers.implementations.native.backends.csv_backend import (
            NativeCsvBackend,
        )

        path = _resolve_local_path(
            source_local_path=handle.source_local_path,
            storage_uri=handle.storage_uri,
        )
        yield from NativeCsvBackend.iter_rows(
            path,
            delimiter=handle.delimiter,
            quotechar=handle.quotechar,
            has_header=handle.has_header,
            encoding=handle.encoding or "utf8-lossy",
            columns=handle.columns or None,
        )
        return

    if isinstance(handle, XlsxSheetHandle):
        from unity.file_manager.file_parsers.implementations.native.backends.excel_backend import (
            NativeExcelBackend,
        )

        path = _resolve_local_path(
            source_local_path=handle.source_local_path,
            storage_uri=handle.storage_uri,
        )
        yield from NativeExcelBackend.iter_rows(
            path,
            sheet_name=handle.sheet_name,
            has_header=handle.has_header,
            columns=handle.columns or None,
        )
        return

    if isinstance(handle, ObjectStoreArtifactHandle):
        yield from _iter_object_store_rows(handle)
        return

    raise TypeError(f"Unsupported table input handle: {type(handle)!r}")


def iter_table_input_row_batches(
    handle: TableInputHandle,
    batch_size: int,
) -> Iterator[list[JsonObject]]:
    """Yield bounded row batches from a table input handle."""

    size = max(int(batch_size or 0), 1)
    batch: list[JsonObject] = []
    for row in iter_table_input_rows(handle):
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# JSONL artifact handle
# ---------------------------------------------------------------------------


def _iter_object_store_rows(handle: ObjectStoreArtifactHandle) -> Iterator[JsonObject]:
    if handle.artifact_format != "jsonl":
        raise NotImplementedError(
            f"Artifact streaming is not implemented for {handle.artifact_format!r}",
        )

    path = _resolve_local_path(source_local_path="", storage_uri=handle.storage_uri)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                yield {str(key): value for key, value in loaded.items()}


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_local_path(*, source_local_path: str, storage_uri: str) -> Path:
    if source_local_path:
        return Path(source_local_path).expanduser().resolve()
    parsed = urlparse(storage_uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser().resolve()
    if parsed.scheme == "gs":
        raise NotImplementedError(
            f"gs:// URIs require a GCS adapter that downloads the object to a "
            f"local path before streaming.  Received: {storage_uri}  "
            f"Implement a GcsArtifactStore and call its download_to_local() "
            f"method in the worker bootstrap, then pass the local path via "
            f"source_local_path.",
        )
    raise ValueError(f"Cannot resolve local path from storage URI: {storage_uri}")
