"""Public assistant-job lifecycle surface for Unity.

The open-source runtime uses this module as a stable API, while the concrete
hosted implementation is supplied via ``unity.deploy_runtime`` when the private
deployment package is installed and enabled.
"""

from unity.deploy_runtime import (
    log_job_startup as _log_job_startup,
    mark_job_done as _mark_job_done,
    mark_job_label as _mark_job_label,
    update_liveview_url as _update_liveview_url,
)


def mark_job_label(
    job_name: str,
    status: str,
    assistant_id: str | None = None,
    ack_ts: str | None = None,
    timeout: float = 30,
    retries: int = 0,
) -> bool:
    return _mark_job_label(
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
    _log_job_startup(job_name, user_id, assistant_id, medium=medium)


def update_liveview_url(assistant_id: str, user_id: str, liveview_url: str) -> None:
    _update_liveview_url(assistant_id, user_id, liveview_url)


def mark_job_done(
    job_name: str,
    inactivity_timeout: float = 0.0,
    shutdown_reason: str | None = None,
) -> None:
    _mark_job_done(
        job_name,
        inactivity_timeout=inactivity_timeout,
        shutdown_reason=shutdown_reason,
    )
