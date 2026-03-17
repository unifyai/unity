#!/usr/bin/env python3
"""Watch Unity job terminations and run exit cleanup.

Uses kopf (Kubernetes Operator Pythonic Framework) to watch Jobs labelled
``app=unity``.  When a Job reaches a terminal condition (*Complete* or
*Failed*), the handler:

1. Marks ``running=False`` on the assistant's ``AssistantJobs`` record in
   Orchestra so the reverse-lookup directory stays accurate.
2. Releases any pool VM still assigned to the assistant (prevents leaked
   VMs lingering in "assigned" state after a crash).

The ``assistant-id`` label is set on the Job (not the Pod) by
``mark_job_label`` during startup, so we must watch Jobs to read it.

On graceful exit, ``mark_job_done`` in the Unity container performs the
same idempotent cleanup.  The job-watcher covers crash scenarios where
in-container cleanup never runs.

Cleanup logic lives in ``assistant_jobs_api.py`` (shared with the
Unity codebase) and is copied into this container at build time.

kopf manages the watch stream lifecycle, reconnection, error isolation,
retries, deduplication, and liveness probes.
"""

from __future__ import annotations

import datetime
import logging
import os

import kopf

from assistant_jobs_api import expire_assistant_records, release_pool_vm

COMMS_URL = os.environ["UNITY_COMMS_URL"]
SHARED_UNIFY_KEY = os.environ["SHARED_UNIFY_KEY"]
ADMIN_KEY = os.environ["ORCHESTRA_ADMIN_KEY"]
MAX_EVENT_AGE = datetime.timedelta(minutes=5)

_events_processed = 0


@kopf.on.startup()
def configure(settings: kopf.OperatorSettings, **_):
    settings.posting.enabled = False
    settings.persistence.finalizer = None

    logging.getLogger("kopf").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


def _parse_k8s_timestamp(ts: str) -> datetime.datetime | None:
    """Parse a K8s ISO-8601 timestamp (e.g. ``2026-03-16T12:00:00Z``)."""
    try:
        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


@kopf.on.event("batch", "v1", "jobs", labels={"app": "unity"})
def on_job_event(event, **_):
    """React to every Job event; filter for terminal conditions."""
    global _events_processed

    job = event.get("object", {})
    status = job.get("status", {})
    print(
        f"job_name: {job.get('metadata', {}).get('name', 'unknown')} status: {status}"
    )

    if status.get("active", 0) >= 1:
        return

    conditions = status.get("conditions", [])

    terminal = next(
        (
            c
            for c in conditions
            if c.get("type") in ("Complete", "Failed") and c.get("status") == "True"
        ),
        None,
    )
    if terminal is None:
        return

    transition_time = _parse_k8s_timestamp(terminal.get("lastTransitionTime", ""))
    if transition_time is not None:
        age = datetime.datetime.now(datetime.timezone.utc) - transition_time
        if age > MAX_EVENT_AGE:
            return

    metadata = job.get("metadata", {})
    labels = metadata.get("labels", {})
    job_name = metadata.get("name", "unknown")
    assistant_id = labels.get("assistant-id")

    print(
        f"Job {job_name} terminal (condition={terminal['type']}, assistant-id={assistant_id})"
    )
    _events_processed += 1

    if not assistant_id:
        print(f"No assistant-id label on {job_name} — skipping cleanup")
        return

    expire_assistant_records(SHARED_UNIFY_KEY, assistant_id)
    release_pool_vm(COMMS_URL, ADMIN_KEY, assistant_id)


@kopf.on.probe(id="health")
def health_probe(**_):
    return {
        "events_processed": _events_processed,
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
