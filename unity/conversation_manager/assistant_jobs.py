"""AssistantJobs lifecycle helpers for the Unity container.

Thin wrapper around ``assistant_jobs_api`` that reads session-specific
values (``SESSION_DETAILS``, ``SETTINGS``) and records Prometheus
metrics.  All actual HTTP operations live in ``assistant_jobs_api.py``
which is shared with the job-watcher operator.

On graceful exit, ``mark_job_done`` also runs record expiry and VM
release for immediate cleanup.  The job-watcher operator
(``scripts/job-watcher/``) repeats the same idempotent operations
externally to cover crash scenarios.
"""

from dotenv import load_dotenv

load_dotenv()
import time
import traceback
from datetime import datetime, timezone

import unify

from unity.logger import LOGGER
from unity.common.hierarchical_logger import ICONS
from unity.conversation_manager.assistant_jobs_api import (
    ensure_project_exists,
    expire_assistant_records,
    get_assistant_logs,
    get_running_count,
    patch_job_label,
    release_pool_vm,
)
from unity.conversation_manager.metrics import (
    running_job_count as _m_running_jobs,
    session_duration as _m_session_dur,
)
from unity.session_details import SESSION_DETAILS
from unity.settings import SETTINGS

# Track whether AssistantJobs project has been verified/created
_project_verified = False

# Session start time (perf_counter), set by log_job_startup, read by mark_job_done
_session_start_perf: float | None = None


def _ensure_project_exists(api_key: str) -> None:
    """Lazily ensure the AssistantJobs project exists."""
    global _project_verified
    if _project_verified or not api_key:
        return
    try:
        ensure_project_exists(api_key)
        _project_verified = True
    except Exception as e:
        LOGGER.error(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Could not verify/create AssistantJobs project: {e}",
        )


def _record_running_job_count(api_key: str) -> None:
    """Query running jobs and record the count as a metric (best-effort)."""
    try:
        count = get_running_count(api_key)
        _m_running_jobs.set(count)
        LOGGER.debug(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Running job count: {count}",
        )
    except Exception as exc:
        LOGGER.error(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Failed to record running job count: {exc}",
        )


def mark_job_label(
    job_name: str,
    status: str,
    assistant_id: str | None = None,
    timeout: float = 30,
    retries: int = 0,
):
    """Patch the K8s Job unity-status label via the communication service."""
    comms_url = SETTINGS.conversation.COMMS_URL.rstrip("/")
    admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()
    if not comms_url or not admin_key:
        LOGGER.debug(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Skipping label update: COMMS_URL or admin key not configured",
        )
        return
    ok = patch_job_label(
        comms_url,
        admin_key,
        job_name,
        status,
        assistant_id,
        timeout=timeout,
        retries=retries,
    )
    if ok:
        LOGGER.debug(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Marked job as {status}: {job_name}",
        )
    else:
        LOGGER.warning(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Failed to mark job as {status}: {job_name}",
        )


def log_job_startup(job_name: str, user_id: str, assistant_id: str):
    """Create or update the running job record with job_name.

    If a running record already exists, updates it with the current
    job_name.  Otherwise creates a fresh record with all available
    session metadata from ``SESSION_DETAILS``.

    The liveview_url is set later by ``update_liveview_url`` when the
    ``AssistantDesktopReady`` event arrives.
    """
    api_key = SESSION_DETAILS.shared_unify_key or None
    if not api_key:
        LOGGER.debug(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Skipping log_job_startup: no shared API key available",
        )
        return

    _ensure_project_exists(api_key)

    try:
        LOGGER.debug(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Getting existing logs for user_id={user_id}, assistant_id={assistant_id}",
        )
        existing_logs = get_assistant_logs(
            api_key,
            f"user_id == '{user_id}' and "
            f"assistant_id == '{assistant_id}' and "
            f"running == 'true'",
        )
        LOGGER.debug(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Found {len(existing_logs)} running records",
        )

        if existing_logs:
            existing_logs[0].update_entries(job_name=job_name)
            LOGGER.debug(
                f"{ICONS['assistant_jobs']} [assistant_jobs] Updated existing record with job_name={job_name}",
            )
        else:
            LOGGER.warning(
                f"{ICONS['assistant_jobs']} [assistant_jobs] No running record found for "
                f"user_id={user_id}, assistant_id={assistant_id}. "
                f"Creating record from container.",
            )
            unify.log(
                project="AssistantJobs",
                context="startup_events",
                api_key=api_key,
                user_id=user_id,
                assistant_id=assistant_id,
                job_name=job_name,
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                running=True,
                assistant_name=SESSION_DETAILS.assistant.name,
                user_name=SESSION_DETAILS.user.name,
                user_number=SESSION_DETAILS.user.number,
                assistant_number=SESSION_DETAILS.assistant.number,
                user_email=SESSION_DETAILS.user.email,
                assistant_email=SESSION_DETAILS.assistant.email,
            )

        # X1: record running job count right after the record is updated
        _record_running_job_count(api_key)

        # Mark session start for U9 duration measurement
        global _session_start_perf
        _session_start_perf = time.perf_counter()
    except Exception as e:
        LOGGER.error(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Error in log_job_startup: {e}",
        )
        traceback.print_exc()


def update_liveview_url(assistant_id: str, user_id: str, liveview_url: str) -> None:
    """Update the AssistantJobs record with the resolved liveview_url.

    Called by the ``AssistantDesktopReady`` event handler once the VM is
    confirmed ready.
    """
    api_key = SESSION_DETAILS.shared_unify_key or None
    if not api_key:
        return

    _ensure_project_exists(api_key)

    try:
        existing_logs = get_assistant_logs(
            api_key,
            f"user_id == '{user_id}' and "
            f"assistant_id == '{assistant_id}' and "
            f"running == 'true'",
        )
        if existing_logs:
            existing_logs[0].update_entries(liveview_url=liveview_url)
            LOGGER.debug(
                f"{ICONS['assistant_jobs']} [assistant_jobs] Updated record with liveview_url={liveview_url}",
            )
    except Exception as e:
        LOGGER.error(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Error updating liveview_url: {e}",
        )


def mark_job_done(job_name: str, inactivity_timeout: float = 0.0):
    """Mark a job as done, expire records, release VM, and record metrics.

    The job-watcher operator repeats the same expire/release calls
    externally (crash-safe).  Both paths are idempotent so running
    them here as well gives immediate cleanup on graceful exit.
    """
    mark_job_label(job_name, "done")

    assistant_id = str(SESSION_DETAILS.assistant.agent_id)
    api_key = SESSION_DETAILS.shared_unify_key or None
    comms_url = SETTINGS.conversation.COMMS_URL.rstrip("/")
    admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()

    # Release pool VM first so the VM is freed even if later steps are slow
    if (
        comms_url
        and admin_key
        and SESSION_DETAILS.assistant.desktop_mode in ("windows", "ubuntu")
    ):
        release_pool_vm(comms_url, admin_key, assistant_id)

    if api_key:
        expire_assistant_records(api_key, assistant_id)
        _record_running_job_count(api_key)

    if _session_start_perf is not None:
        total_dur = time.perf_counter() - _session_start_perf
        active_dur = max(0.0, total_dur - inactivity_timeout)
        _m_session_dur.record(active_dur)
        LOGGER.debug(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Session duration: "
            f"{total_dur:.1f}s total, {inactivity_timeout:.1f}s idle, {active_dur:.1f}s active",
        )
