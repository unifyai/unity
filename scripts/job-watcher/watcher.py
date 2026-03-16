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

log = logging.getLogger("job-watcher")

COMMS_URL = os.environ["UNITY_COMMS_URL"]
SHARED_UNIFY_KEY = os.environ["SHARED_UNIFY_KEY"]
ADMIN_KEY = os.environ["ORCHESTRA_ADMIN_KEY"]

_events_processed = 0


@kopf.on.event("batch", "v1", "jobs", labels={"app": "unity"})
def on_job_event(event, **_):
    """React to every Job event; filter for terminal conditions."""
    global _events_processed

    job = event.get("object", {})
    conditions = job.get("status", {}).get("conditions", [])

    terminal = next(
        (c for c in conditions
         if c.get("type") in ("Complete", "Failed") and c.get("status") == "True"),
        None,
    )
    if terminal is None:
        return

    metadata = job.get("metadata", {})
    labels = metadata.get("labels", {})
    job_name = metadata.get("name", "unknown")
    assistant_id = labels.get("assistant-id")

    log.info(
        "Job %s terminal (condition=%s, assistant-id=%s)",
        job_name, terminal["type"], assistant_id,
    )
    _events_processed += 1

    if not assistant_id:
        log.info("No assistant-id label on %s — skipping cleanup", job_name)
        return

    expire_assistant_records(SHARED_UNIFY_KEY, assistant_id)
    release_pool_vm(COMMS_URL, ADMIN_KEY, assistant_id)


@kopf.on.probe(id="health")
def health_probe(**_):
    return {
        "events_processed": _events_processed,
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
