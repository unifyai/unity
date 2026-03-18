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

import requests
import unify

log = logging.getLogger(__name__)

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


def get_running_count(api_key: str) -> int:
    """Return the number of records where ``running == 'true'``."""
    logs = get_assistant_logs(api_key, "running == 'true'")
    return len(logs)


# ---------------------------------------------------------------------------
# Record mutations
# ---------------------------------------------------------------------------


def expire_assistant_records(api_key: str, assistant_id: str) -> None:
    """Set ``running=False`` on all records for *assistant_id*."""
    try:
        logs = get_assistant_logs(
            api_key,
            f"assistant_id == '{assistant_id}' and running == 'true'",
        )
        if not logs:
            log.info("No running records for %s — already clean", assistant_id)
            return

        for entry in logs:
            entry.update_entries(running=False)
        log.info("Expired %d record(s) for assistant %s", len(logs), assistant_id)
    except Exception:
        log.exception("Error expiring records for %s", assistant_id)


# ---------------------------------------------------------------------------
# K8s label operations (via comms service)
# ---------------------------------------------------------------------------


def patch_job_label(
    comms_url: str,
    admin_key: str,
    job_name: str,
    status: str,
    assistant_id: str | None = None,
    timeout: float = 30,
    retries: int = 0,
) -> bool:
    """Patch the K8s Job ``unity-status`` label.  Returns True on success."""
    labels: dict[str, str] = {"unity-status": status}
    if assistant_id is not None:
        labels["assistant-id"] = str(assistant_id).lower().replace("_", "-")
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


def release_pool_vm(
    comms_url: str,
    admin_key: str,
    assistant_id: str,
) -> None:
    """Fire-and-forget request to release the pool VM for *assistant_id*.

    The communication service handles the actual GCP release. The
    job-watcher provides crash-safe coverage if this request is lost.
    """
    try:
        requests.post(
            f"{comms_url}/infra/vm/pool/release",
            json={"assistant_id": assistant_id},
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=0.1,
        )
        log.info("Pool VM release dispatched for %s", assistant_id)
    except requests.exceptions.Timeout:
        log.info("Pool VM release dispatched for %s (timeout)", assistant_id)
    except Exception:
        log.exception("Error dispatching pool VM release for %s", assistant_id)
