from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Sequence

from unity.file_manager.types.config import FilePipelineConfig

from .ops import add_or_replace_file_row

if TYPE_CHECKING:
    from unity.file_manager.managers.file_manager import FileManager

logger = logging.getLogger(__name__)

_UNSET = object()


class AttachmentIngestionPool:
    """Manages a pool of background attachment ingestion workers.

    Wraps a ``ThreadPoolExecutor`` with deduplication so the same file path
    is never ingested concurrently more than once.  Instances are safe for
    use from multiple threads.
    """

    def __init__(self, *, max_workers: int = 2) -> None:
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="attachment-ingest",
        )
        self._active: dict[str, Future[None]] = {}

    def enqueue(
        self,
        file_manager: "FileManager",
        file_paths: str | Sequence[str],
        *,
        config: FilePipelineConfig | None = None,
    ) -> list[str]:
        queued: list[str] = []
        ingest_config = config or _default_attachment_ingest_config()

        for file_path in _normalize_paths(file_paths):
            with self._lock:
                active = self._active.get(file_path)
                if active is not None and not active.done():
                    queued.append(file_path)
                    continue

                _upsert_attachment_status(
                    file_manager,
                    file_path=file_path,
                    ingestion_status="queued",
                )
                future = self._executor.submit(
                    _run_attachment_ingest_job,
                    file_manager,
                    file_path=file_path,
                    config=ingest_config,
                )
                self._active[file_path] = future

            def _cleanup(done: Future[None], *, key: str = file_path) -> None:
                with self._lock:
                    if self._active.get(key) is done:
                        self._active.pop(key, None)

            future.add_done_callback(_cleanup)
            queued.append(file_path)

        return queued

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


_POOL: AttachmentIngestionPool | None = None
_POOL_LOCK = threading.Lock()


def _get_pool() -> AttachmentIngestionPool:
    global _POOL
    if _POOL is not None:
        return _POOL
    with _POOL_LOCK:
        if _POOL is None:
            from unity.settings import SETTINGS

            _POOL = AttachmentIngestionPool(
                max_workers=SETTINGS.file.ATTACHMENT_INGESTION_MAX_WORKERS,
            )
        return _POOL


def enqueue_attachment_ingestion(
    file_manager: "FileManager",
    file_paths: str | Sequence[str],
    *,
    config: FilePipelineConfig | None = None,
) -> list[str]:
    """Public entry point -- delegates to the module-level pool singleton."""
    return _get_pool().enqueue(file_manager, file_paths, config=config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_paths(file_paths: str | Sequence[str]) -> list[str]:
    if isinstance(file_paths, str):
        return [file_paths]
    return [path for path in file_paths if path]


def _default_attachment_ingest_config() -> FilePipelineConfig:
    config = FilePipelineConfig()
    config.execution.parallel_files = False
    return config


def _upsert_attachment_status(
    file_manager: "FileManager",
    *,
    file_path: str,
    ingestion_status: str,
    error: str | None = None,
    parse_status: str | None | object = _UNSET,
) -> None:
    try:
        entry: dict = {
            "file_path": file_path,
            "source_uri": file_manager.resolve_source_uri(file_path),
            "source_provider": file_manager.source_provider_name,
            "ingestion_status": ingestion_status,
            "error": error,
            "storage_id": "",
        }
        if parse_status is not _UNSET:
            entry["status"] = parse_status
        add_or_replace_file_row(file_manager, entry=entry)
    except Exception:
        logger.exception(
            "Failed to persist attachment ingestion status",
            extra={"file_path": file_path, "ingestion_status": ingestion_status},
        )


def _run_attachment_ingest_job(
    file_manager: "FileManager",
    *,
    file_path: str,
    config: FilePipelineConfig,
) -> None:
    try:
        _upsert_attachment_status(
            file_manager,
            file_path=file_path,
            ingestion_status="ingesting",
        )
        result = file_manager.ingest_files(file_path, config=config)
        file_result = getattr(result, "files", {}).get(file_path)
        if file_result is None:
            _upsert_attachment_status(
                file_manager,
                file_path=file_path,
                ingestion_status="error",
                error="attachment ingestion completed without a per-file result",
                parse_status="error",
            )
            return
        if getattr(file_result, "status", None) == "error":
            _upsert_attachment_status(
                file_manager,
                file_path=file_path,
                ingestion_status="error",
                error=getattr(file_result, "error", None) or "file could not be parsed",
                parse_status="error",
            )
            return
        _upsert_attachment_status(
            file_manager,
            file_path=file_path,
            ingestion_status="success",
        )
    except Exception as exc:
        logger.exception(
            "Attachment ingestion job failed",
            extra={"file_path": file_path},
        )
        _upsert_attachment_status(
            file_manager,
            file_path=file_path,
            ingestion_status="error",
            error=str(exc) or "attachment ingestion failed",
            parse_status="error",
        )
