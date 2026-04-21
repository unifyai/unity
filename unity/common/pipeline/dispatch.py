"""Shared Pub/Sub publisher for ``ParseRequested`` messages.

Callers (attachment ingestion, operator scripts, ad-hoc dispatch CLI)
invoke :func:`publish_parse_request` to:

1. Ensure the source bytes live in GCS (uploading from memory or a local
   path when necessary, or passing through a pre-existing ``gs://`` URI).
2. Construct a :class:`ParseRequested` envelope with ``ingestion_mode``
   and the appropriate :class:`FmBinding` / :class:`DmBinding` populated.
3. Publish that envelope to the parse topic (``unity-parse{env_suffix}``)
   via Pub/Sub.

Contract:

* Exactly one file per :class:`ParseRequested`. This is load-bearing for
  Tier-2 parallelism (one pod = one message = one file). The function
  raises if callers try to circumvent this.
* ``ingestion_mode="fm"`` requires an ``fm_binding``; ``"dm"`` requires a
  ``dm_binding``. Mismatches raise ``ValueError``.
* Exactly one of ``source_local_path`` / ``source_bytes`` /
  ``source_gs_uri`` must be supplied. The first two cause a GCS upload;
  the third short-circuits when the bytes already live in GCS.

This module lives under :mod:`unity.common.pipeline` (where
:class:`ParseRequested` is already defined) so both live-assistant code
(``attachment_ingestion.py``) and deploy-side operator scripts can
consume it without creating circular package dependencies.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from .types import (
    AttachmentCallback,
    DmBinding,
    FmBinding,
    ParseRequested,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DispatchTarget:
    """Static routing info for a dispatch: GCP project, bucket, env suffix.

    ``env_suffix`` follows the cross-repo convention: empty string for
    production, ``"-{env}"`` otherwise (e.g. ``"-staging"``). It is
    appended to every environment-qualified resource name
    (``parse_topic_base + env_suffix``).

    Upload location is **not** configurable per-caller. Every dispatched
    source file lands at ``jobs/<job_id>/source/<basename>`` so all
    artifacts for a job share a single self-contained root directory and
    operators can ``gsutil ls gs://<bucket>/jobs/<job_id>/`` to see
    everything (source bytes, manifests, materialised tables, ledgers)
    for that job in one place.
    """

    project_id: str
    bucket_name: str
    env_suffix: str = ""
    parse_topic_base: str = "unity-parse"


@dataclass(frozen=True)
class PublishResult:
    """Outcome of a single dispatch."""

    job_id: str
    gs_uri: str
    topic: str
    message_id: str


def publish_parse_request(
    *,
    target: DispatchTarget,
    logical_path: str,
    ingestion_mode: Literal["fm", "dm"],
    fm_binding: Optional[FmBinding] = None,
    dm_binding: Optional[DmBinding] = None,
    source_local_path: Optional[str] = None,
    source_bytes: Optional[bytes] = None,
    source_gs_uri: Optional[str] = None,
    blob_key: Optional[str] = None,
    attachment_callback: Optional[AttachmentCallback] = None,
    deployment_id: str = "",
    job_id: Optional[str] = None,
    pubsub_attributes: Optional[dict[str, str]] = None,
    storage_client: Any | None = None,
    publisher_client: Any | None = None,
) -> PublishResult:
    """Upload the source bytes to GCS (if needed) and publish ``ParseRequested``.

    Returns a :class:`PublishResult` with the generated/assigned
    ``job_id``, the resolved ``gs_uri`` of the source bytes, the full
    topic path the message was published to, and the Pub/Sub
    ``message_id`` returned by the publisher.

    All heavy dependencies (``google.cloud.storage``, ``pubsub_v1``) are
    imported lazily so importing this module from test code or a worker
    pod that only publishes does not require them to be present.
    """
    _validate_inputs(
        target=target,
        logical_path=logical_path,
        ingestion_mode=ingestion_mode,
        fm_binding=fm_binding,
        dm_binding=dm_binding,
        source_local_path=source_local_path,
        source_bytes=source_bytes,
        source_gs_uri=source_gs_uri,
    )

    resolved_job_id = job_id or uuid.uuid4().hex

    gs_uri = _ensure_in_gcs(
        target=target,
        logical_path=logical_path,
        job_id=resolved_job_id,
        blob_key=blob_key,
        source_local_path=source_local_path,
        source_bytes=source_bytes,
        source_gs_uri=source_gs_uri,
        storage_client=storage_client,
    )

    parse_msg = ParseRequested(
        job_id=resolved_job_id,
        deployment_id=deployment_id,
        file_paths=[gs_uri],
        attachment_callback=attachment_callback,
        ingestion_mode=ingestion_mode,
        fm_binding=fm_binding,
        dm_binding=dm_binding,
    )

    topic_path, message_id = _publish(
        target=target,
        payload=parse_msg.model_dump(mode="json"),
        pubsub_attributes=pubsub_attributes,
        publisher_client=publisher_client,
    )

    logger.info(
        "Dispatched parse request: job=%s mode=%s uri=%s topic=%s message_id=%s",
        resolved_job_id,
        ingestion_mode,
        gs_uri,
        topic_path,
        message_id,
    )

    return PublishResult(
        job_id=resolved_job_id,
        gs_uri=gs_uri,
        topic=topic_path,
        message_id=message_id,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    *,
    target: DispatchTarget,
    logical_path: str,
    ingestion_mode: str,
    fm_binding: Optional[FmBinding],
    dm_binding: Optional[DmBinding],
    source_local_path: Optional[str],
    source_bytes: Optional[bytes],
    source_gs_uri: Optional[str],
) -> None:
    if not logical_path:
        raise ValueError("logical_path must be non-empty")

    if ingestion_mode not in ("fm", "dm"):
        raise ValueError(
            f"ingestion_mode must be 'fm' or 'dm', got {ingestion_mode!r}",
        )

    if ingestion_mode == "fm" and fm_binding is None:
        raise ValueError("ingestion_mode='fm' requires fm_binding")
    if ingestion_mode == "dm" and dm_binding is None:
        raise ValueError("ingestion_mode='dm' requires dm_binding")
    if ingestion_mode == "fm" and dm_binding is not None:
        raise ValueError("ingestion_mode='fm' must not supply dm_binding")
    if ingestion_mode == "dm" and fm_binding is not None:
        raise ValueError("ingestion_mode='dm' must not supply fm_binding")

    sources_supplied = sum(
        1 for s in (source_local_path, source_bytes, source_gs_uri) if s is not None
    )
    if sources_supplied != 1:
        raise ValueError(
            "Exactly one of source_local_path, source_bytes, source_gs_uri "
            f"must be supplied; got {sources_supplied}",
        )

    if source_gs_uri is not None and not source_gs_uri.startswith("gs://"):
        raise ValueError(
            f"source_gs_uri must start with 'gs://'; got {source_gs_uri!r}",
        )

    if not target.project_id:
        raise ValueError("DispatchTarget.project_id must be non-empty")
    if not target.bucket_name and source_gs_uri is None:
        raise ValueError(
            "DispatchTarget.bucket_name is required when uploading bytes",
        )


def _ensure_in_gcs(
    *,
    target: DispatchTarget,
    logical_path: str,
    job_id: str,
    blob_key: Optional[str],
    source_local_path: Optional[str],
    source_bytes: Optional[bytes],
    source_gs_uri: Optional[str],
    storage_client: Any | None,
) -> str:
    """Return a ``gs://`` URI for the source bytes, uploading if needed."""
    if source_gs_uri is not None:
        return source_gs_uri

    resolved_blob_key = blob_key or _default_blob_key(
        job_id=job_id,
        logical_path=logical_path,
    )

    from google.cloud import storage

    client = storage_client or storage.Client(project=target.project_id)
    bucket = client.bucket(target.bucket_name)
    blob = bucket.blob(resolved_blob_key)

    import os
    import time as _time

    upload_size = 0
    if source_bytes is not None:
        upload_size = len(source_bytes)
    elif source_local_path is not None:
        try:
            upload_size = os.path.getsize(source_local_path)
        except OSError:
            pass

    t0 = _time.perf_counter()
    if source_bytes is not None:
        blob.upload_from_string(source_bytes)
    else:
        assert source_local_path is not None  # narrowed by _validate_inputs
        blob.upload_from_filename(source_local_path)
    elapsed = _time.perf_counter() - t0

    gs_uri = f"gs://{target.bucket_name}/{resolved_blob_key}"
    mb = upload_size / (1024 * 1024)
    rate = mb / elapsed if elapsed > 0 else 0
    logger.info(
        "Uploaded %.1f MB to %s in %.1fs (%.1f MB/s)",
        mb,
        gs_uri,
        elapsed,
        rate,
    )
    return gs_uri


def _default_blob_key(*, job_id: str, logical_path: str) -> str:
    """Compose ``jobs/<job_id>/source/<basename(logical_path)>``.

    This is the single authoritative location for a dispatched source
    file. Co-locating the source bytes with the downstream manifests,
    materialised artifacts, and ledgers under the same ``jobs/<job_id>/``
    root means one directory contains everything operators need to
    inspect or purge a single job, and makes cross-environment leaks
    structurally impossible (the enclosing bucket is env-scoped).
    """
    basename = Path(logical_path).name or "unnamed"
    return f"jobs/{job_id}/source/{basename}"


def _publish(
    *,
    target: DispatchTarget,
    payload: dict,
    pubsub_attributes: Optional[dict[str, str]],
    publisher_client: Any | None,
) -> tuple[str, str]:
    """Publish ``payload`` as JSON; return ``(topic_path, message_id)``."""
    from google.cloud import pubsub_v1

    publisher = publisher_client or pubsub_v1.PublisherClient()
    topic_name = f"{target.parse_topic_base}{target.env_suffix}"
    topic_path = publisher.topic_path(target.project_id, topic_name)

    data = json.dumps(payload, default=str).encode("utf-8")
    attrs = dict(pubsub_attributes or {})
    future = publisher.publish(topic_path, data=data, **attrs)
    message_id: str = future.result(timeout=30)
    return topic_path, message_id
