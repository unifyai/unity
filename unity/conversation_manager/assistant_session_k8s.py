from __future__ import annotations

import base64
import json
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
CONTAINER_READY_ANNOTATION = "assistantsession.unify.ai/container-ready"

_batch_api: k8s_client.BatchV1Api | None = None
_core_api: k8s_client.CoreV1Api | None = None
_custom_api: k8s_client.CustomObjectsApi | None = None


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


def read_session_bootstrap_secret(secret_name: str) -> dict[str, Any]:
    _load_clients()
    assert _core_api is not None
    secret = _core_api.read_namespaced_secret(name=secret_name, namespace=_namespace())
    data = secret.data or {}
    raw = data.get("startup.json", "")
    if not raw:
        return {}
    return json.loads(base64.b64decode(raw).decode("utf-8"))


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
