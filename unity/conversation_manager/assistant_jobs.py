"""AssistantJobs lifecycle helpers for the Unity container.

Thin wrapper around ``assistant_jobs_api`` that reads session-specific
values (``SESSION_DETAILS``, ``SETTINGS``) and records Prometheus
metrics.  All actual HTTP operations live in ``assistant_jobs_api.py``
which is shared with the job-watcher operator.

``log_job_startup`` creates the AssistantJobs audit record with all
assistant/user info from ``SESSION_DETAILS`` plus the container-specific
``job_name``.  ``update_liveview_url`` may later add the desktop URL.
The job-watcher operator handles crash-safe VM release independently.
"""

from dotenv import load_dotenv

load_dotenv()
import time
import traceback
from datetime import datetime, timezone

from unity.logger import LOGGER
from unity.common.hierarchical_logger import ICONS
from unity.conversation_manager.assistant_jobs_api import (
    create_assistant_log,
    ensure_project_exists,
    get_assistant_logs,
    patch_job_label,
    release_pool_vm,
)
from unity.conversation_manager.metrics import (
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


def log_job_startup(
    job_name: str,
    user_id: str,
    assistant_id: str,
    medium: str = "",
):
    """Create an AssistantJobs audit record for this container session.

    Logs all available assistant/user info from ``SESSION_DETAILS`` plus
    the container-specific ``job_name``.  ``update_liveview_url`` may
    later add the desktop URL.
    """
    api_key = SESSION_DETAILS.shared_unify_key or None
    if not api_key:
        LOGGER.debug(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Skipping log_job_startup: no shared API key available",
        )
        return

    _ensure_project_exists(api_key)

    try:
        sd = SESSION_DETAILS
        create_assistant_log(
            api_key,
            user_id=user_id,
            assistant_id=assistant_id,
            job_name=job_name,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            medium=medium,
            user_name=f"{sd.user.first_name} {sd.user.surname}".strip(),
            assistant_name=f"{sd.assistant.first_name} {sd.assistant.surname}".strip(),
            user_number=sd.user.number,
            assistant_number=sd.assistant.number,
            user_email=sd.user.email,
            assistant_email=sd.assistant.email,
        )
        LOGGER.debug(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Created audit record: "
            f"job_name={job_name}, assistant_id={assistant_id}",
        )

        global _session_start_perf
        _session_start_perf = time.perf_counter()
    except Exception as e:
        LOGGER.error(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Error creating job record: {e}",
        )
        traceback.print_exc()


def update_liveview_url(assistant_id: str, user_id: str, liveview_url: str) -> None:
    """Update the AssistantJobs record with the resolved liveview_url.

    Called by the ``AssistantDesktopReady`` event handler once the VM is
    confirmed ready.  Finds the record by ``assistant_id`` + ``job_name``
    (unique to this container session).
    """
    api_key = SESSION_DETAILS.shared_unify_key or None
    if not api_key:
        return

    job_name = SETTINGS.conversation.JOB_NAME
    if not job_name:
        return

    _ensure_project_exists(api_key)

    try:
        existing_logs = get_assistant_logs(
            api_key,
            f"assistant_id == '{assistant_id}' and " f"job_name == '{job_name}'",
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
    """Mark a job as done, release VM, and record session duration.

    The job-watcher operator performs crash-safe VM release independently.
    """
    mark_job_label(job_name, "done")

    assistant_id = str(SESSION_DETAILS.assistant.agent_id)
    comms_url = SETTINGS.conversation.COMMS_URL.rstrip("/")
    admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()

    # U9: session duration (log_job_startup -> mark_job_done), excluding idle tail
    if _session_start_perf is not None:
        total_dur = time.perf_counter() - _session_start_perf
        active_dur = max(0.0, total_dur - inactivity_timeout)
        _m_session_dur.record(active_dur)
        LOGGER.debug(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Session duration: "
            f"{total_dur:.1f}s total, {inactivity_timeout:.1f}s idle, {active_dur:.1f}s active",
        )

    # Release pool VM if applicable (managed VM, not user's own desktop)
    if (
        comms_url
        and admin_key
        and SESSION_DETAILS.assistant.desktop_mode in ("windows", "ubuntu")
    ):
        release_pool_vm(comms_url, admin_key, assistant_id)
