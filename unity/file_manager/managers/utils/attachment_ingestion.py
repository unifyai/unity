from __future__ import annotations

import logging
import threading
import uuid
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

        if _pipeline_dispatch_enabled():
            for file_path in _normalize_paths(file_paths):
                _upsert_attachment_status(
                    file_manager,
                    file_path=file_path,
                    ingestion_status="queued",
                )
                try:
                    _dispatch_attachment_to_workers(
                        file_manager,
                        file_path=file_path,
                    )
                    queued.append(file_path)
                except Exception as exc:
                    logger.exception(
                        "Failed to dispatch attachment to workers",
                        extra={"file_path": file_path},
                    )
                    _upsert_attachment_status(
                        file_manager,
                        file_path=file_path,
                        ingestion_status="error",
                        error=str(exc) or "dispatch failed",
                        parse_status="error",
                    )
            return queued

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


# ---------------------------------------------------------------------------
# Worker dispatch (attachment -> GKE parse/ingest workers)
# ---------------------------------------------------------------------------


def _pipeline_dispatch_enabled() -> bool:
    """Return True when attachment ingestion should dispatch to GKE workers.

    Checks ``SETTINGS.file.PIPELINE_DISPATCH_ENABLED`` and verifies that a
    bucket and GCP project are configured; otherwise falls back to the
    in-process ``AttachmentIngestionPool``.
    """
    try:
        from unity.settings import SETTINGS
    except Exception:
        return False
    if not SETTINGS.file.PIPELINE_DISPATCH_ENABLED:
        return False
    if not SETTINGS.file.PIPELINE_ARTIFACT_BUCKET:
        return False
    if not SETTINGS.GCP_PROJECT_ID:
        return False
    return True


def _dispatch_attachment_to_workers(
    file_manager: "FileManager",
    *,
    file_path: str,
) -> None:
    """Upload attachment bytes to GCS and publish a ParseRequested envelope.

    Uses :func:`unity.common.pipeline.publish_parse_request` so the GCS
    upload + Pub/Sub publish logic (and the ``one file per
    ParseRequested`` invariant) stays in one place shared with operator
    dispatch scripts. The attachment path sets ``ingestion_mode="fm"``
    and an ``FmBinding`` so the ingest worker lands the file under
    ``Files/{alias}/{storage_id}/...`` via ``fm_process_plan``, not a
    bare ``DataManager`` context.

    The ingest worker publishes an ``attachment_ingestion_complete``
    event back to the per-assistant topic, which ``CommsManager``
    dispatches into the existing event broker.
    """
    from unity.common.pipeline import (
        DispatchTarget,
        publish_parse_request,
    )
    from unity.common.pipeline.types import AttachmentCallback, FmBinding
    from unity.session_details import SESSION_DETAILS, UNASSIGNED_USER_CONTEXT
    from unity.settings import SETTINGS

    bucket_name = SETTINGS.file.PIPELINE_ARTIFACT_BUCKET
    project_id = SETTINGS.GCP_PROJECT_ID
    env_suffix = SETTINGS.ENV_SUFFIX

    assistant_id = getattr(
        getattr(SESSION_DETAILS, "assistant", None),
        "agent_id",
        None,
    )
    if not assistant_id:
        raise RuntimeError(
            "PIPELINE_DISPATCH_ENABLED requires SESSION_DETAILS.assistant.agent_id",
        )
    user_id = (
        getattr(getattr(SESSION_DETAILS, "user", None), "id", None)
        or UNASSIGNED_USER_CONTEXT
    )

    data = file_manager._open_bytes_by_filepath(file_path)
    attachment_id = uuid.uuid4().hex
    job_id = f"attachment-{assistant_id}-{attachment_id}"

    callback = AttachmentCallback(
        assistant_id=str(assistant_id),
        env_suffix=env_suffix,
        display_name=file_path,
    )
    # Attachments always land under the "Local" alias -- this is the
    # alias exposed by ``LocalFileSystemAdapter`` and mirrors what the
    # ingest worker reconstructs via ``LocalFileSystemAdapter(root=None,
    # enable_sync=False)`` in :func:`_run_fm_mode`.
    fm_binding = FmBinding(
        user_id=str(user_id),
        assistant_id=str(assistant_id),
        fm_alias="Local",
        logical_path=file_path,
    )
    # No ``upload_prefix`` -- ``publish_parse_request`` composes every
    # source file under ``jobs/<job_id>/source/<basename>``. Because
    # ``job_id = f"attachment-{assistant_id}-{attachment_id}"`` below,
    # attachments remain scoped per-assistant-and-attachment, just
    # nested under the unified ``jobs/`` root that every other
    # pipeline artifact also lives under.
    target = DispatchTarget(
        project_id=project_id,
        bucket_name=bucket_name,
        env_suffix=env_suffix,
    )

    result = publish_parse_request(
        target=target,
        logical_path=file_path,
        ingestion_mode="fm",
        fm_binding=fm_binding,
        source_bytes=data,
        attachment_callback=callback,
        job_id=job_id,
        pubsub_attributes={"thread": "attachment_parse"},
    )

    _upsert_attachment_status(
        file_manager,
        file_path=file_path,
        ingestion_status="dispatched",
    )
    logger.info(
        "Dispatched attachment to parse worker: job=%s uri=%s message_id=%s",
        result.job_id,
        result.gs_uri,
        result.message_id,
    )


# ---------------------------------------------------------------------------
# Completion callback (invoked by CommsManager when the worker reports back)
# ---------------------------------------------------------------------------


def apply_attachment_completion(
    file_manager: "FileManager",
    *,
    display_name: str,
    status: str,
    error: str | None = None,
) -> None:
    """Update ``FileRecords`` for a completed worker-dispatched attachment.

    Called from ``CommsManager`` when a ``thread="attachment_ingestion_complete"``
    message arrives on the per-assistant Pub/Sub topic.
    """
    if status == "success":
        _upsert_attachment_status(
            file_manager,
            file_path=display_name,
            ingestion_status="success",
        )
    else:
        _upsert_attachment_status(
            file_manager,
            file_path=display_name,
            ingestion_status="error",
            error=error or "attachment ingestion failed",
            parse_status="error",
        )
