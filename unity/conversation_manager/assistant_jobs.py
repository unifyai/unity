"""AssistantJobs lifecycle: log job startup, mark done, query running count.

Manages records in the ``AssistantJobs`` Unify project that track which
assistant containers are currently running.
"""

from dotenv import load_dotenv

load_dotenv()
import json
import time
import traceback
import requests
import unify

from unity.logger import LOGGER
from unity.common.hierarchical_logger import DEFAULT_ICON, ICONS
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
        unify.create_project("AssistantJobs", api_key=api_key)
        _project_verified = True
    except Exception as e:
        LOGGER.error(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Could not verify/create AssistantJobs project: {e}",
        )


def _is_managed_vm() -> bool:
    """Check if running on a managed VM.

    Returns True when desktop_mode is "windows" or "ubuntu".
    """
    return SESSION_DETAILS.assistant.desktop_mode in ("windows", "ubuntu")


def _record_running_job_count(api_key: str) -> None:
    """Query running jobs and record the count as a metric (best-effort)."""
    try:
        logs = unify.get_logs(
            project="AssistantJobs",
            context="startup_events",
            filter="running == 'true'",
            limit=100,
            api_key=api_key,
        )
        _m_running_jobs.set(len(logs))
        LOGGER.debug(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Running job count: {len(logs)}",
        )
    except Exception as exc:
        LOGGER.error(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Failed to record running job count: {exc}",
        )


def mark_job_label(job_name: str, status: str):
    """Patch the K8s Job unity-status label via the communication service."""
    comms_url = SETTINGS.conversation.COMMS_URL.rstrip("/")
    admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()
    if not comms_url or not admin_key:
        LOGGER.debug(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Skipping label update: COMMS_URL or admin key not configured",
        )
        return
    try:
        resp = requests.patch(
            f"{comms_url}/infra/job/labels",
            data={
                "job_name": job_name,
                "labels": json.dumps({"unity-status": status}),
            },
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=30,
        )
        if resp.ok:
            LOGGER.debug(
                f"{ICONS['assistant_jobs']} [assistant_jobs] Marked job as {status}: {job_name}",
            )
        else:
            LOGGER.warning(
                f"{ICONS['assistant_jobs']} [assistant_jobs] Failed to mark job as {status} "
                f"(status {resp.status_code}): {resp.text}",
            )
    except Exception as e:
        LOGGER.warning(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Error marking job as {status}: {e}",
        )


def log_job_startup(job_name: str, user_id: str, assistant_id: str):
    """Update the running job record with job_name.

    The adapter already created the running=True record with all assistant info.
    This function adds the container-specific job_name.  The liveview_url is
    set later by ``update_liveview_url`` when the ``AssistantDesktopReady``
    event arrives.
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
        existing_logs = unify.get_logs(
            project="AssistantJobs",
            context="startup_events",
            filter=(
                f"user_id == '{user_id}' and "
                f"assistant_id == '{assistant_id}' and "
                f"running == 'true'"
            ),
            limit=100,
            api_key=api_key,
        )
        LOGGER.debug(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Found {len(existing_logs)} running records",
        )

        if existing_logs:
            log = existing_logs[0]
            log.update_entries(job_name=job_name)
            LOGGER.debug(
                f"{ICONS['assistant_jobs']} [assistant_jobs] Updated record with job_name={job_name}",
            )

            # X1: record running job count right after the record is updated
            _record_running_job_count(api_key)

            # Mark session start for U9 duration measurement
            global _session_start_perf
            _session_start_perf = time.perf_counter()
        else:
            # No record found - adapter's mark_job_running() must have failed
            # Log warning but don't fail; liveview just won't be tracked
            LOGGER.error(
                f"{ICONS['assistant_jobs']} [assistant_jobs] WARNING: No running record found for "
                f"user_id={user_id}, assistant_id={assistant_id}. "
                f"Adapter may have failed to create the record.",
            )
    except Exception as e:
        LOGGER.error(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Error updating job record: {e}",
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
        existing_logs = unify.get_logs(
            project="AssistantJobs",
            context="startup_events",
            filter=(
                f"user_id == '{user_id}' and "
                f"assistant_id == '{assistant_id}' and "
                f"running == 'true'"
            ),
            limit=100,
            api_key=api_key,
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


def _release_vm(assistant_id: str, _max_attempts: int = 3) -> None:
    """Release the pool VM assigned to this assistant back to the pool.

    The pool release endpoint finds the VM by its ``assistant_id`` label
    and transitions it from "assigned" back to "available".  Retries on
    transient 5xx errors to guard against Cloud Run cold-start 503s.

    If the release response indicates no VM is assigned (labels already
    cleared but disk still attached), falls back to an explicit disk
    detach call.
    """
    comms_url = SETTINGS.conversation.COMMS_URL.rstrip("/")
    admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()
    if not comms_url or not admin_key:
        LOGGER.debug(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Skipping VM release: "
            "COMMS_URL or admin key not configured",
        )
        return

    headers = {"Authorization": f"Bearer {admin_key}"}

    for attempt in range(1, _max_attempts + 1):
        try:
            response = requests.post(
                f"{comms_url}/infra/vm/pool/release",
                json={"assistant_id": assistant_id},
                headers=headers,
                timeout=60,
            )
            if response.ok:
                body = response.json()
                if body.get("released"):
                    LOGGER.info(
                        f"{ICONS['assistant_jobs']} [assistant_jobs] Pool VM released for assistant "
                        f"{assistant_id}: {body}",
                    )
                    return
                LOGGER.warning(
                    f"{ICONS['assistant_jobs']} [assistant_jobs] Release returned released=false "
                    f"for {assistant_id}: {body}",
                )
                _detach_disk(assistant_id, comms_url, headers)
                return
            if response.status_code >= 500 and attempt < _max_attempts:
                LOGGER.warning(
                    f"{ICONS['assistant_jobs']} [assistant_jobs] Pool VM release got "
                    f"{response.status_code}, retrying ({attempt}/{_max_attempts})…",
                )
                time.sleep(attempt)
                continue
            LOGGER.error(
                f"{ICONS['assistant_jobs']} [assistant_jobs] Failed to release pool VM: "
                f"{response.status_code} {response.text}",
            )
            return
        except Exception as e:
            LOGGER.error(
                f"{ICONS['assistant_jobs']} [assistant_jobs] Error releasing pool VM: {e}",
            )
            traceback.print_exc()
            return


def _detach_disk(assistant_id: str, comms_url: str, headers: dict) -> None:
    """Best-effort detach of the assistant's persistent disk."""
    try:
        response = requests.post(
            f"{comms_url}/infra/vm/pool/disk/detach/{assistant_id}",
            headers=headers,
            timeout=60,
        )
        if response.ok:
            LOGGER.info(
                f"{ICONS['assistant_jobs']} [assistant_jobs] Disk detached for assistant "
                f"{assistant_id}: {response.json()}",
            )
        else:
            LOGGER.error(
                f"{ICONS['assistant_jobs']} [assistant_jobs] Failed to detach disk: "
                f"{response.status_code} {response.text}",
            )
    except Exception as e:
        LOGGER.error(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Error detaching disk: {e}",
        )
        traceback.print_exc()


def mark_job_done(job_name: str, inactivity_timeout: float = 0.0):
    """Mark a job as done and record session-end metrics."""
    mark_job_label(job_name, "done")

    api_key = SESSION_DETAILS.shared_unify_key or None
    if not api_key:
        LOGGER.debug(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Skipping mark_job_done: no shared API key available",
        )
        return

    # mark job done in the logs
    try:
        job_log = unify.get_logs(
            project="AssistantJobs",
            context="startup_events",
            filter=f"job_name == '{job_name}'",
            limit=100,
            api_key=api_key,
        )[0]
        job_log.update_entries(running=False)
        LOGGER.info(f"{DEFAULT_ICON} Job marked done {job_name}")

        # X1: record running job count right after the record is updated
        _record_running_job_count(api_key)
    except Exception as e:
        LOGGER.error(f"{DEFAULT_ICON} Error finding job: {e}")
        traceback.print_exc()

    # U9: session duration (log_job_startup → mark_job_done), excluding idle tail
    if _session_start_perf is not None:
        total_dur = time.perf_counter() - _session_start_perf
        active_dur = max(0.0, total_dur - inactivity_timeout)
        _m_session_dur.record(active_dur)
        LOGGER.debug(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Session duration: "
            f"{total_dur:.1f}s total, {inactivity_timeout:.1f}s idle, {active_dur:.1f}s active",
        )

    # Release pool VM if applicable (managed VM, not user's own desktop)
    if _is_managed_vm():
        _release_vm(str(SESSION_DETAILS.assistant.agent_id))
