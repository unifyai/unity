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


def _calc_wait_from_ready_at(vm_ready_at: str | None) -> int:
    """Calculate seconds to wait based on vm_ready_at timestamp.

    Returns wait time in seconds, minimum 5 (no maximum).
    """
    if not vm_ready_at:
        return 10  # Default if no timestamp

    try:
        from datetime import datetime, timezone

        # Parse ISO timestamp (e.g., "2025-01-16T14:02:00.000-08:00")
        ready_dt = datetime.fromisoformat(vm_ready_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = (ready_dt - now).total_seconds()
        # Minimum 5 seconds, no maximum
        return max(5, int(delta))
    except Exception:
        return 10  # Fallback on parse error


def _resolve_vm_liveview(assistant_id: str, vm_type: str) -> str | None:
    """Resolve liveview URL for a VM by polling /infra/vm/status endpoint.

    Args:
        assistant_id: The assistant ID to check status for.
        vm_type: VM type ("windows" or "ubuntu").

    Returns the desktop_url when vm_ready=True, or None if resolution fails.
    """
    comms_url = SETTINGS.conversation.COMMS_URL.rstrip("/")
    admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()
    if not comms_url or not admin_key:
        LOGGER.debug(
            f"{ICONS['liveview']} [Liveview] Skipping: COMMS_URL or admin key not configured",
        )
        return None

    max_retries = 5
    for attempt in range(max_retries):
        LOGGER.debug(
            f"{ICONS['liveview']} [Liveview] Attempt {attempt + 1} to get {vm_type} VM status "
            f"for assistant {assistant_id}",
        )
        try:
            resp = requests.get(
                f"{comms_url}/infra/vm/status/{assistant_id}",
                params={"vm_type": vm_type},
                headers={"Authorization": f"Bearer {admin_key}"},
                timeout=30,
            )
            if resp.ok:
                data = resp.json() or {}
                vm_ready = data.get("vm_ready", False)
                vm_ready_at = data.get("vm_ready_at")
                desktop_url = data.get("desktop_url")
                status = data.get("status", "UNKNOWN")

                LOGGER.debug(
                    f"{ICONS['liveview']} [Liveview] VM Status: {status}, Ready: {vm_ready}",
                )

                if vm_ready and desktop_url:
                    LOGGER.info(
                        f"{ICONS['liveview']} [Liveview] {vm_type.capitalize()} VM is ready!",
                    )
                    LOGGER.info(
                        f"{ICONS['liveview']} [Liveview] URL: {desktop_url}/desktop/custom.html",
                    )
                    return f"{desktop_url}/desktop/custom.html"

                # Calculate wait time from vm_ready_at timestamp (no max clamp)
                wait_time = _calc_wait_from_ready_at(vm_ready_at)
                if wait_time > 60:
                    mins, secs = divmod(wait_time, 60)
                    LOGGER.debug(
                        f"{ICONS['liveview']} [Liveview] Waiting {mins}m {secs}s...",
                    )
                else:
                    LOGGER.debug(
                        f"{ICONS['liveview']} [Liveview] Waiting {wait_time}s...",
                    )
                time.sleep(wait_time)
            else:
                LOGGER.error(
                    f"{ICONS['liveview']} [Liveview] Request failed: {resp.status_code} {resp.text}",
                )
                time.sleep(10)
        except Exception as e:
            LOGGER.error(f"{ICONS['liveview']} [Liveview] Error: {e}")
            time.sleep(10)

    return None


def _record_running_job_count(api_key: str) -> None:
    """Query running jobs and record the count as a metric (best-effort)."""
    try:
        logs = unify.get_logs(
            project="AssistantJobs",
            context="startup_events",
            filter="running == 'true'",
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


def _mark_job_label(job_name: str, status: str):
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
            timeout=10,
        )
        if resp.ok:
            LOGGER.info(
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
    """Update the running job record with job_name and liveview_url.

    The adapter already created the running=True record with all assistant info.
    This function just adds the container-specific details: job_name and liveview_url.
    """
    _mark_job_label(job_name, "running")

    api_key = SESSION_DETAILS.shared_unify_key or None
    if not api_key:
        LOGGER.debug(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Skipping log_job_startup: no shared API key available",
        )
        return

    _ensure_project_exists(api_key)

    # Update the existing record (created by adapter) with job_name and liveview_url
    existing_logs = []
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
            api_key=api_key,
        )
        LOGGER.debug(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Found {len(existing_logs)} running records",
        )

        if existing_logs:
            log = existing_logs[0]
            log.update_entries(job_name=job_name)
            LOGGER.info(
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

    # Resolve liveview URL and attach it to the record (this can take a while).
    # Only proceed if we successfully obtained a log record above.
    if existing_logs:
        try:
            if _is_managed_vm():
                vm_type = SESSION_DETAILS.assistant.desktop_mode
                liveview_url = _resolve_vm_liveview(assistant_id, vm_type)
            else:
                # User's own desktop - no liveview URL to resolve
                liveview_url = None
        except Exception as e:
            LOGGER.error(
                f"{ICONS['liveview']} [Liveview] Error resolving liveview URL: {e}",
            )
            traceback.print_exc()
            liveview_url = None
        log.update_entries(liveview_url=liveview_url)
        LOGGER.info(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Updated record with liveview_url={liveview_url}",
        )


def _stop_vm(assistant_id: str, vm_type: str) -> None:
    """Stop the VM for the given assistant.

    Called when a job is marked done and the assistant was running on a
    managed VM (desktop_mode in windows/ubuntu).

    Args:
        assistant_id: The assistant ID whose VM to stop.
        vm_type: VM type ("windows" or "ubuntu").
    """
    try:
        comms_url = SETTINGS.conversation.COMMS_URL.rstrip("/")
        admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()
        if not comms_url or not admin_key:
            LOGGER.debug(
                f"{ICONS['assistant_jobs']} [assistant_jobs] Skipping VM stop: "
                "COMMS_URL or admin key not configured",
            )
            return

        response = requests.post(
            f"{comms_url}/infra/vm/stop",
            json={"assistant_id": assistant_id, "vm_type": vm_type},
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=60,
        )
        if response.ok:
            LOGGER.info(
                f"{ICONS['assistant_jobs']} [assistant_jobs] {vm_type.capitalize()} VM stopped for assistant "
                f"{assistant_id}: {response.json()}",
            )
        else:
            LOGGER.error(
                f"{ICONS['assistant_jobs']} [assistant_jobs] Failed to stop {vm_type} VM: "
                f"{response.status_code} {response.text}",
            )
    except requests.exceptions.Timeout:
        LOGGER.error(
            f"{ICONS['assistant_jobs']} [assistant_jobs] {vm_type.capitalize()} VM stop request timed out",
        )
    except Exception as e:
        LOGGER.error(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Error stopping {vm_type} VM: {e}",
        )
        traceback.print_exc()


def mark_job_done(job_name: str):
    """Mark a job as done and record session-end metrics."""
    _mark_job_label(job_name, "done")

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
            api_key=api_key,
        )[0]
        job_log.update_entries(running=False)
        LOGGER.info(f"{DEFAULT_ICON} Job marked done {job_name}")

        # X1: record running job count right after the record is updated
        _record_running_job_count(api_key)
    except Exception as e:
        LOGGER.error(f"{DEFAULT_ICON} Error finding job: {e}")
        traceback.print_exc()

    # U9: session duration (log_job_startup → mark_job_done)
    if _session_start_perf is not None:
        dur = time.perf_counter() - _session_start_perf
        _m_session_dur.record(dur)
        LOGGER.info(
            f"{ICONS['assistant_jobs']} [assistant_jobs] Session duration: {dur:.1f}s",
        )

    # Stop VM if applicable (managed VM, not user's own desktop)
    if _is_managed_vm():
        vm_type = SESSION_DETAILS.assistant.desktop_mode
        _stop_vm(SESSION_DETAILS.assistant.id, vm_type)
