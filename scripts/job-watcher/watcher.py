#!/usr/bin/env python3
"""Watch Unity pod terminations and run exit cleanup.

Uses kopf (Kubernetes Operator Pythonic Framework) to watch pods labelled
``app=unity``.  When a pod transitions to *Succeeded* or *Failed*, the
handler:

1. Marks ``running=False`` on the assistant's ``AssistantJobs`` record in
   Orchestra so the reverse-lookup directory stays accurate.
2. Releases any pool VM still assigned to the assistant (prevents leaked
   VMs lingering in "assigned" state after a crash).

This is the **sole owner** of VM release and AssistantJobs record cleanup.
The Unity container's ``mark_job_done`` only handles session duration
metrics and the K8s label patch.  By running externally, cleanup is
guaranteed regardless of how the Unity container exits (graceful, OOM,
segfault, node failure, etc.).

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


@kopf.on.event("", "v1", "pods", labels={"app": "unity"})
def on_pod_event(event, **_):
    """React to every pod event; filter for terminal phases."""
    global _events_processed

    pod = event.get("object", {})
    phase = pod.get("status", {}).get("phase")
    if phase not in ("Succeeded", "Failed"):
        return

    metadata = pod.get("metadata", {})
    labels = metadata.get("labels", {})
    pod_name = metadata.get("name", "unknown")
    assistant_id = labels.get("assistant-id")

    log.info(
        "Pod %s terminated (phase=%s, assistant-id=%s)", pod_name, phase, assistant_id
    )
    _events_processed += 1

    if not assistant_id:
        log.info("No assistant-id label on %s — skipping cleanup", pod_name)
        return

    expire_assistant_records(SHARED_UNIFY_KEY, assistant_id)
    release_pool_vm(COMMS_URL, ADMIN_KEY, assistant_id)


@kopf.on.probe(id="health")
def health_probe(**_):
    return {
        "events_processed": _events_processed,
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
