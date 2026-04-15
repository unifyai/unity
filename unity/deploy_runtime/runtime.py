from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobAssignmentRecord:
    session_name: str
    binding_id: str


@dataclass(frozen=True)
class BootstrapSecretRecord:
    name: str
    payload: dict[str, Any]
    owner_session_name: str
    owner_activation_id: str


class SessionAssignmentBackend(Protocol):
    def wait_for_assistant_session_name(self, job_name: str) -> str: ...

    def read_job_assignment_record(self, job_name: str) -> JobAssignmentRecord: ...

    def read_assistant_session(self, session_name: str) -> dict[str, Any]: ...

    def read_session_bootstrap_secret_record(
        self,
        secret_name: str,
    ) -> BootstrapSecretRecord: ...

    def mark_job_container_ready(self, job_name: str, max_retries: int = 3) -> None: ...

    def collect_shutdown_diagnostics(self, job_name: str) -> dict[str, Any]: ...


class AssistantJobsBackend(Protocol):
    def mark_job_label(
        self,
        job_name: str,
        status: str,
        assistant_id: str | None = None,
        ack_ts: str | None = None,
        timeout: float = 30,
        retries: int = 0,
    ) -> bool: ...

    def log_job_startup(
        self,
        job_name: str,
        user_id: str,
        assistant_id: str,
        medium: str = "",
    ) -> None: ...

    def update_liveview_url(
        self,
        assistant_id: str,
        user_id: str,
        liveview_url: str,
    ) -> None: ...

    def mark_job_done(
        self,
        job_name: str,
        inactivity_timeout: float = 0.0,
        shutdown_reason: str | None = None,
    ) -> None: ...


class MetricsBackend(Protocol):
    def init_metrics(self) -> None: ...

    async def shutdown_metrics(self) -> None: ...


class ShutdownLogBackend(Protocol):
    def upload_shutdown_logs(self) -> bool: ...


class MissingHostedSessionBackend:
    def _unsupported(self, operation: str) -> None:
        raise RuntimeError(
            f"{operation} requires a hosted deploy runtime backend. "
            "This local-first Unity runtime does not configure one by default.",
        )

    def wait_for_assistant_session_name(self, job_name: str) -> str:
        self._unsupported("wait_for_assistant_session_name")

    def read_job_assignment_record(self, job_name: str) -> JobAssignmentRecord:
        self._unsupported("read_job_assignment_record")

    def read_assistant_session(self, session_name: str) -> dict[str, Any]:
        self._unsupported("read_assistant_session")

    def read_session_bootstrap_secret_record(
        self,
        secret_name: str,
    ) -> BootstrapSecretRecord:
        self._unsupported("read_session_bootstrap_secret_record")

    def mark_job_container_ready(self, job_name: str, max_retries: int = 3) -> None:
        self._unsupported("mark_job_container_ready")

    def collect_shutdown_diagnostics(self, job_name: str) -> dict[str, Any]:
        return {
            "job_name": job_name,
            "backend_available": False,
            "reason": "no hosted session backend configured",
        }


class NoopAssistantJobsBackend:
    def mark_job_label(
        self,
        job_name: str,
        status: str,
        assistant_id: str | None = None,
        ack_ts: str | None = None,
        timeout: float = 30,
        retries: int = 0,
    ) -> bool:
        return False

    def log_job_startup(
        self,
        job_name: str,
        user_id: str,
        assistant_id: str,
        medium: str = "",
    ) -> None:
        return None

    def update_liveview_url(
        self,
        assistant_id: str,
        user_id: str,
        liveview_url: str,
    ) -> None:
        return None

    def mark_job_done(
        self,
        job_name: str,
        inactivity_timeout: float = 0.0,
        shutdown_reason: str | None = None,
    ) -> None:
        return None


class NoopMetricsBackend:
    def init_metrics(self) -> None:
        return None

    async def shutdown_metrics(self) -> None:
        return None


class NoopShutdownLogBackend:
    def upload_shutdown_logs(self) -> bool:
        return False


@dataclass(frozen=True)
class DeployRuntimeBackends:
    session: SessionAssignmentBackend = field(
        default_factory=MissingHostedSessionBackend,
    )
    jobs: AssistantJobsBackend = field(default_factory=NoopAssistantJobsBackend)
    metrics: MetricsBackend = field(default_factory=NoopMetricsBackend)
    logs: ShutdownLogBackend = field(default_factory=NoopShutdownLogBackend)


_BACKENDS = DeployRuntimeBackends()
_LOAD_ATTEMPTED = False


def register_deploy_runtime(
    backends: DeployRuntimeBackends | None = None,
    *,
    session: SessionAssignmentBackend | None = None,
    jobs: AssistantJobsBackend | None = None,
    metrics: MetricsBackend | None = None,
    logs: ShutdownLogBackend | None = None,
) -> DeployRuntimeBackends:
    global _BACKENDS
    if backends is not None:
        _BACKENDS = backends
    else:
        _BACKENDS = replace(
            _BACKENDS,
            session=session or _BACKENDS.session,
            jobs=jobs or _BACKENDS.jobs,
            metrics=metrics or _BACKENDS.metrics,
            logs=logs or _BACKENDS.logs,
        )
    return _BACKENDS


def reset_deploy_runtime() -> None:
    global _BACKENDS, _LOAD_ATTEMPTED
    _BACKENDS = DeployRuntimeBackends()
    _LOAD_ATTEMPTED = False


def _apply_runtime_overrides(overrides: Any) -> None:
    if overrides is None:
        return
    if isinstance(overrides, DeployRuntimeBackends):
        register_deploy_runtime(overrides)
        return
    if isinstance(overrides, dict):
        register_deploy_runtime(
            session=overrides.get("session"),
            jobs=overrides.get("jobs"),
            metrics=overrides.get("metrics"),
            logs=overrides.get("logs"),
        )
        return
    raise TypeError(
        "Runtime backend registration must return DeployRuntimeBackends or a "
        "dict containing session/jobs/metrics/logs overrides.",
    )


def ensure_deploy_runtime_loaded() -> None:
    global _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return
    _LOAD_ATTEMPTED = True

    if not os.environ.get("_UNITY_STARTUP_HOOK_GROUP"):
        return

    group = os.environ.get("_UNITY_RUNTIME_HOOK_GROUP", "unity.deploy_runtime")
    package_filter = os.environ.get("_UNITY_STARTUP_HOOK_PACKAGE")

    try:
        from importlib.metadata import entry_points

        matched = False
        for ep in entry_points(group=group):
            if (
                package_filter
                and getattr(ep, "dist", None)
                and ep.dist.name != package_filter
            ):
                continue
            matched = True
            LOGGER.info("Loading deploy runtime backend: %s", ep.name)
            register_fn = ep.load()
            _apply_runtime_overrides(register_fn())

        if not matched:
            LOGGER.info(
                "No deploy runtime backends discovered for entry point group %s",
                group,
            )
    except Exception as exc:
        LOGGER.warning("Deploy runtime backend discovery failed: %s", exc)


def get_deploy_runtime() -> DeployRuntimeBackends:
    ensure_deploy_runtime_loaded()
    return _BACKENDS


def wait_for_assistant_session_name(job_name: str) -> str:
    return get_deploy_runtime().session.wait_for_assistant_session_name(job_name)


def read_job_assignment_record(job_name: str) -> JobAssignmentRecord:
    return get_deploy_runtime().session.read_job_assignment_record(job_name)


def read_assistant_session(session_name: str) -> dict[str, Any]:
    return get_deploy_runtime().session.read_assistant_session(session_name)


def read_session_bootstrap_secret_record(secret_name: str) -> BootstrapSecretRecord:
    return get_deploy_runtime().session.read_session_bootstrap_secret_record(
        secret_name,
    )


def mark_job_container_ready(job_name: str, max_retries: int = 3) -> None:
    get_deploy_runtime().session.mark_job_container_ready(
        job_name,
        max_retries=max_retries,
    )


def collect_shutdown_diagnostics(job_name: str) -> dict[str, Any]:
    return get_deploy_runtime().session.collect_shutdown_diagnostics(job_name)


def mark_job_label(
    job_name: str,
    status: str,
    assistant_id: str | None = None,
    ack_ts: str | None = None,
    timeout: float = 30,
    retries: int = 0,
) -> bool:
    return get_deploy_runtime().jobs.mark_job_label(
        job_name,
        status,
        assistant_id=assistant_id,
        ack_ts=ack_ts,
        timeout=timeout,
        retries=retries,
    )


def log_job_startup(
    job_name: str,
    user_id: str,
    assistant_id: str,
    medium: str = "",
) -> None:
    get_deploy_runtime().jobs.log_job_startup(
        job_name,
        user_id,
        assistant_id,
        medium=medium,
    )


def update_liveview_url(assistant_id: str, user_id: str, liveview_url: str) -> None:
    get_deploy_runtime().jobs.update_liveview_url(
        assistant_id,
        user_id,
        liveview_url,
    )


def mark_job_done(
    job_name: str,
    inactivity_timeout: float = 0.0,
    shutdown_reason: str | None = None,
) -> None:
    get_deploy_runtime().jobs.mark_job_done(
        job_name,
        inactivity_timeout=inactivity_timeout,
        shutdown_reason=shutdown_reason,
    )


def init_metrics() -> None:
    get_deploy_runtime().metrics.init_metrics()


async def shutdown_metrics() -> None:
    await get_deploy_runtime().metrics.shutdown_metrics()


def upload_shutdown_logs() -> bool:
    return get_deploy_runtime().logs.upload_shutdown_logs()
