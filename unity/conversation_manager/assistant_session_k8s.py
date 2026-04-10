from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import logging
import time

from kubernetes import client as k8s_client, config as k8s_config, watch
from kubernetes.client.rest import ApiException

from unity.settings import SETTINGS

logger = logging.getLogger(__name__)

SESSION_REF_LABEL = "assistantsession.unify.ai/name"
SESSION_REF_ANNOTATION = "assistantsession.unify.ai/name"
BINDING_ID_LABEL = "assistantsession.unify.ai/binding-id"
BINDING_ID_ANNOTATION = "assistantsession.unify.ai/binding-id"
ACTIVATION_ID_ANNOTATION = "assistantsession.unify.ai/activation-id"
CONTAINER_READY_ANNOTATION = "assistantsession.unify.ai/container-ready"

_batch_api: k8s_client.BatchV1Api | None = None
_core_api: k8s_client.CoreV1Api | None = None
_custom_api: k8s_client.CustomObjectsApi | None = None


@dataclass(frozen=True)
class JobAssignmentRecord:
    """Binding-scoped session identity stamped onto a Unity Job."""

    session_name: str
    binding_id: str


@dataclass(frozen=True)
class BootstrapSecretRecord:
    """Bootstrap payload plus the owner annotations recorded on the Secret."""

    name: str
    payload: dict[str, Any]
    owner_session_name: str
    owner_activation_id: str


def _load_clients() -> None:
    global _batch_api, _core_api, _custom_api
    if _batch_api and _core_api and _custom_api:
        return
    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()
    api_client = k8s_client.ApiClient()
    _batch_api = k8s_client.BatchV1Api(api_client)
    _core_api = k8s_client.CoreV1Api(api_client)
    _custom_api = k8s_client.CustomObjectsApi(api_client)


def _namespace() -> str:
    ns_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
    if ns_path.exists():
        return ns_path.read_text().strip()
    return SETTINGS.DEPLOY_ENV


def _session_name_from_job(job) -> str | None:
    labels = job.metadata.labels or {}
    annotations = job.metadata.annotations or {}
    return labels.get(SESSION_REF_LABEL) or annotations.get(SESSION_REF_ANNOTATION)


def _binding_id_from_job(job) -> str | None:
    labels = job.metadata.labels or {}
    annotations = job.metadata.annotations or {}
    return labels.get(BINDING_ID_LABEL) or annotations.get(BINDING_ID_ANNOTATION)


def wait_for_assistant_session_name(job_name: str) -> str:
    _load_clients()
    assert _batch_api is not None

    namespace = _namespace()
    logger.info(
        "assistant-session wait start for job %s in namespace %s",
        job_name,
        namespace,
    )
    while True:
        job = _batch_api.read_namespaced_job(name=job_name, namespace=namespace)
        session_name = _session_name_from_job(job)
        if session_name:
            logger.info(
                "assistant-session found on job %s via direct read: %s",
                job_name,
                session_name,
            )
            return session_name

        watcher = watch.Watch()
        try:
            logger.info(
                "assistant-session ref missing on %s, watching for label/annotation update",
                job_name,
            )
            for event in watcher.stream(
                _batch_api.list_namespaced_job,
                namespace=namespace,
                field_selector=f"metadata.name={job_name}",
                timeout_seconds=30,
            ):
                session_name = _session_name_from_job(event["object"])
                if session_name:
                    watcher.stop()
                    logger.info(
                        "assistant-session found on job %s via watch event: %s",
                        job_name,
                        session_name,
                    )
                    return session_name
        except Exception:
            logger.exception(
                "assistant-session watch failed for job %s, retrying",
                job_name,
            )
            continue


def read_job_assignment_record(job_name: str) -> JobAssignmentRecord:
    """Read the controller-owned session and binding identity from a Job."""

    _load_clients()
    assert _batch_api is not None
    job = _batch_api.read_namespaced_job(name=job_name, namespace=_namespace())
    return JobAssignmentRecord(
        session_name=str(_session_name_from_job(job) or ""),
        binding_id=str(_binding_id_from_job(job) or ""),
    )


def read_assistant_session(session_name: str) -> dict[str, Any]:
    _load_clients()
    assert _custom_api is not None
    return _custom_api.get_namespaced_custom_object(
        group=SETTINGS.conversation.ASSISTANT_SESSION_GROUP,
        version=SETTINGS.conversation.ASSISTANT_SESSION_VERSION,
        namespace=_namespace(),
        plural=SETTINGS.conversation.ASSISTANT_SESSION_PLURAL,
        name=session_name,
    )


def read_session_bootstrap_secret_record(secret_name: str) -> BootstrapSecretRecord:
    """Read a bootstrap Secret together with its owner annotations."""

    _load_clients()
    assert _core_api is not None
    secret = _core_api.read_namespaced_secret(name=secret_name, namespace=_namespace())
    data = secret.data or {}
    raw = data.get("startup.json", "")
    metadata = getattr(secret, "metadata", None)
    annotations = getattr(metadata, "annotations", None) or {}
    payload = json.loads(base64.b64decode(raw).decode("utf-8")) if raw else {}
    return BootstrapSecretRecord(
        name=str(getattr(metadata, "name", "") or secret_name),
        payload=payload,
        owner_session_name=str(annotations.get(SESSION_REF_ANNOTATION, "") or ""),
        owner_activation_id=str(annotations.get(ACTIVATION_ID_ANNOTATION, "") or ""),
    )


def read_session_bootstrap_secret(secret_name: str) -> dict[str, Any]:
    """Read only the bootstrap payload stored in the Secret."""

    return read_session_bootstrap_secret_record(secret_name).payload


def mark_job_container_ready(job_name: str, max_retries: int = 3) -> None:
    """Patch the container-ready annotation on the owning Job.

    Retries on 409 Conflict (stale resourceVersion) by re-reading the
    Job and re-applying the patch.  Without this retry, a concurrent
    controller or label patch would silently prevent the annotation
    from being set, leaving the session stuck in ContainerAssigned.
    """
    _load_clients()
    assert _batch_api is not None
    namespace = _namespace()

    for attempt in range(max_retries + 1):
        job = _batch_api.read_namespaced_job(name=job_name, namespace=namespace)
        annotations = dict(job.metadata.annotations or {})
        if annotations.get(CONTAINER_READY_ANNOTATION) == "true":
            logger.info("container-ready already set on job %s", job_name)
            return
        annotations[CONTAINER_READY_ANNOTATION] = "true"
        body = {
            "metadata": {
                "annotations": annotations,
                "resourceVersion": job.metadata.resource_version,
            },
        }
        try:
            _batch_api.patch_namespaced_job(
                name=job_name,
                namespace=namespace,
                body=body,
            )
            logger.info(
                "container-ready patched on job %s (attempt %d/%d)",
                job_name,
                attempt + 1,
                max_retries + 1,
            )
            return
        except ApiException as exc:
            if exc.status == 409 and attempt < max_retries:
                logger.warning(
                    "container-ready patch conflict on %s (attempt %d/%d), retrying",
                    job_name,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(0.2 * (attempt + 1))
                continue
            logger.exception(
                "container-ready patch failed on job %s after %d attempt(s)",
                job_name,
                attempt + 1,
            )
            raise


def collect_shutdown_diagnostics(job_name: str) -> dict[str, Any]:
    """Collect job/session/pod/event state at shutdown time.

    Called from Unity's graceful-shutdown path after SIGTERM is received but
    before the process exits. This captures the K8s state needed to attribute
    *why* the pod is being terminated (job suspend, pod deletion, kubelet kill,
    etc.).
    """
    _load_clients()
    assert _batch_api is not None
    assert _core_api is not None

    namespace = _namespace()
    pod_name = os.environ.get("HOSTNAME", "")
    diagnostics: dict[str, Any] = {
        "job_name": job_name,
        "pod_name": pod_name or None,
        "namespace": namespace,
    }

    try:
        job = _batch_api.read_namespaced_job(name=job_name, namespace=namespace)
        diagnostics["job"] = {
            "name": job.metadata.name,
            "resource_version": job.metadata.resource_version,
            "deletion_timestamp": str(job.metadata.deletion_timestamp or ""),
            "labels": dict(job.metadata.labels or {}),
            "annotations": dict(job.metadata.annotations or {}),
            "suspend": bool(getattr(job.spec, "suspend", False)),
            "active": int(job.status.active or 0),
            "succeeded": int(job.status.succeeded or 0),
            "failed": int(job.status.failed or 0),
        }
        session_name = _session_name_from_job(job)
        if session_name:
            diagnostics["session_name"] = session_name
            try:
                session = read_assistant_session(session_name)
                diagnostics["assistant_session"] = {
                    "name": ((session.get("metadata") or {}).get("name") or ""),
                    "activation_id": (
                        (session.get("spec") or {}).get("activationId") or ""
                    ),
                    "desired_state": (
                        (session.get("spec") or {}).get("desiredState") or ""
                    ),
                    "phase": ((session.get("status") or {}).get("phase") or ""),
                    "last_error": (
                        (session.get("status") or {}).get("lastError") or ""
                    ),
                    "binding": ((session.get("status") or {}).get("binding") or {}),
                }
            except Exception as exc:
                diagnostics["assistant_session_error"] = str(exc)
    except Exception as exc:
        diagnostics["job_error"] = str(exc)

    if pod_name:
        try:
            pod = _core_api.read_namespaced_pod(name=pod_name, namespace=namespace)
            diagnostics["pod"] = {
                "name": pod.metadata.name,
                "phase": str(pod.status.phase or ""),
                "node_name": str(pod.spec.node_name or ""),
                "deletion_timestamp": str(pod.metadata.deletion_timestamp or ""),
                "reason": str(getattr(pod.status, "reason", "") or ""),
                "message": str(getattr(pod.status, "message", "") or ""),
                "conditions": [
                    {
                        "type": str(cond.type or ""),
                        "status": str(cond.status or ""),
                        "reason": str(getattr(cond, "reason", "") or ""),
                        "message": str(getattr(cond, "message", "") or ""),
                    }
                    for cond in (pod.status.conditions or [])
                ],
            }
        except Exception as exc:
            diagnostics["pod_error"] = str(exc)

        try:
            events = _core_api.list_namespaced_event(
                namespace=namespace,
                field_selector=f"involvedObject.kind=Pod,involvedObject.name={pod_name}",
            )
            recent_events = sorted(
                events.items,
                key=lambda e: (
                    getattr(getattr(e, "last_timestamp", None), "timestamp", None)
                    or str(getattr(e, "last_timestamp", "") or "")
                    or str(getattr(e.metadata, "creation_timestamp", "") or "")
                ),
            )
            diagnostics["pod_events"] = [
                {
                    "reason": str(getattr(event, "reason", "") or ""),
                    "message": str(getattr(event, "message", "") or ""),
                    "type": str(getattr(event, "type", "") or ""),
                    "count": int(getattr(event, "count", 0) or 0),
                    "last_timestamp": str(getattr(event, "last_timestamp", "") or ""),
                }
                for event in recent_events[-12:]
            ]
        except Exception as exc:
            diagnostics["pod_events_error"] = str(exc)

    return diagnostics
