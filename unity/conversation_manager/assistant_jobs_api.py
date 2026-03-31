"""Low-level AssistantJobs operations: records, labels, and VMs.

Every function accepts explicit parameters (auth keys, URLs) so it can
be called from any context — the Unity container or the job-watcher
operator — without importing Unity-specific packages.

Orchestra log operations use the ``unify`` SDK (which reads
``ORCHESTRA_URL`` from the environment automatically).  Comms-service
operations (labels, VMs) use ``requests`` directly.
"""

from __future__ import annotations

import json
import logging
import time
import traceback

import requests
import unify

log = logging.getLogger(__name__)

VM_RELEASE_ATTEMPTS = 3

PROJECT_NAME = "AssistantJobs"
CONTEXT = "startup_events"


# ---------------------------------------------------------------------------
# Project management
# ---------------------------------------------------------------------------


def ensure_project_exists(api_key: str) -> None:
    """Create the AssistantJobs project if it doesn't already exist."""
    try:
        unify.create_project(PROJECT_NAME, api_key=api_key)
    except Exception:
        log.exception("Error ensuring project exists")


# ---------------------------------------------------------------------------
# Record queries
# ---------------------------------------------------------------------------


def get_assistant_logs(
    api_key: str,
    filter_expr: str,
    limit: int = 100,
) -> list:
    """Fetch AssistantJobs log entries matching *filter_expr*.

    Returns a list of ``unify.Log`` objects.
    """
    return unify.get_logs(
        project=PROJECT_NAME,
        context=CONTEXT,
        filter=filter_expr,
        limit=limit,
        api_key=api_key,
    )


def create_assistant_log(api_key: str, **entries) -> "unify.Log":
    """Create a new AssistantJobs audit log entry."""
    return unify.log(
        project=PROJECT_NAME,
        context=CONTEXT,
        api_key=api_key,
        **entries,
    )


# ---------------------------------------------------------------------------
# K8s label operations (via comms service)
# ---------------------------------------------------------------------------


def patch_job_label(
    comms_url: str,
    admin_key: str,
    job_name: str,
    status: str,
    assistant_id: str | None = None,
    ack_ts: str | None = None,
    timeout: float = 30,
    retries: int = 0,
) -> bool:
    """Patch the K8s Job ``unity-status`` label.  Returns True on success."""
    labels: dict[str, str] = {"unity-status": status}
    if assistant_id is not None:
        labels["assistant-id"] = str(assistant_id).lower().replace("_", "-")
    if ack_ts is not None:
        labels["unity-startup-ack"] = ack_ts
    for attempt in range(1 + retries):
        try:
            resp = requests.patch(
                f"{comms_url}/infra/job/labels",
                data={
                    "job_name": job_name,
                    "labels": json.dumps(labels),
                },
                headers={"Authorization": f"Bearer {admin_key}"},
                timeout=timeout,
            )
            if resp.ok:
                return True
            log.warning(
                "Label patch for %s returned %s (attempt %d/%d)",
                job_name,
                resp.status_code,
                attempt + 1,
                1 + retries,
            )
        except Exception:
            log.exception(
                "Error patching label for %s (attempt %d/%d)",
                job_name,
                attempt + 1,
                1 + retries,
            )
    return False


# ---------------------------------------------------------------------------
# VM operations (via comms service)
# ---------------------------------------------------------------------------


def detach_disk(
    comms_url: str,
    admin_key: str,
    assistant_id: str,
) -> None:
    """Best-effort detach of the assistant's persistent disk."""
    headers = {"Authorization": f"Bearer {admin_key}"}
    try:
        resp = requests.post(
            f"{comms_url}/infra/vm/pool/disk/detach/{assistant_id}",
            headers=headers,
            timeout=60,
        )
        if resp.ok:
            log.info("Disk detached for %s: %s", assistant_id, resp.json())
        else:
            log.error(
                "Failed to detach disk for %s: %d %s",
                assistant_id,
                resp.status_code,
                resp.text,
            )
    except Exception:
        log.exception("Error detaching disk for %s", assistant_id)
        traceback.print_exc()


def release_pool_vm(
    comms_url: str,
    admin_key: str,
    assistant_id: str,
    max_attempts: int = VM_RELEASE_ATTEMPTS,
) -> None:
    """Release the pool VM assigned to *assistant_id* (with retries)."""
    headers = {"Authorization": f"Bearer {admin_key}"}

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(
                f"{comms_url}/infra/vm/pool/release",
                json={"assistant_id": assistant_id},
                headers=headers,
                timeout=60,
            )
            if resp.ok:
                body = resp.json()
                if body.get("released"):
                    log.info("Pool VM released for %s: %s", assistant_id, body)
                    return

                log.warning(
                    "Release returned released=false for %s: %s — "
                    "attempting disk detach",
                    assistant_id,
                    body,
                )
                detach_disk(comms_url, admin_key, assistant_id)
                return

            if resp.status_code >= 500 and attempt < max_attempts:
                log.warning(
                    "Pool VM release got %d for %s, retrying (%d/%d)",
                    resp.status_code,
                    assistant_id,
                    attempt,
                    max_attempts,
                )
                time.sleep(attempt)
                continue

            log.error(
                "Failed to release pool VM for %s: %d %s",
                assistant_id,
                resp.status_code,
                resp.text,
            )
            return
        except Exception:
            log.exception("Error releasing pool VM for %s", assistant_id)
            traceback.print_exc()
            return
