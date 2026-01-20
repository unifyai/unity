from dotenv import load_dotenv

load_dotenv()
import time
import traceback
import requests
import unify

from unity.session_details import SESSION_DETAILS
from unity.settings import SETTINGS


# Track whether AssistantJobs project has been verified/created
_project_verified = False


def _ensure_project_exists(api_key: str) -> None:
    """Lazily ensure the AssistantJobs project exists."""
    global _project_verified
    if _project_verified or not api_key:
        return
    try:
        unify.create_project("AssistantJobs", api_key=api_key)
        _project_verified = True
    except Exception as e:
        print(f"[debug_logger] Could not verify/create AssistantJobs project: {e}")


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


def _resolve_windows_vm_liveview(assistant_id: str) -> str | None:
    """Resolve liveview URL for a Windows VM by polling /infra/vm/status endpoint.

    Returns the desktop_url when vm_ready=True, or None if resolution fails.
    """
    comms_url = SETTINGS.conversation.COMMS_URL.rstrip("/")
    admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()
    if not comms_url or not admin_key:
        print("[Liveview] Skipping: COMMS_URL or admin key not configured")
        return None

    max_retries = 10  # More retries for VM boot (can take longer)
    for attempt in range(max_retries):
        print(
            f"\n\n[Liveview] Attempt {attempt + 1} to get Windows VM status "
            f"for assistant {assistant_id}",
        )
        try:
            resp = requests.get(
                f"{comms_url}/infra/vm/status/{assistant_id}",
                headers={"Authorization": f"Bearer {admin_key}"},
                timeout=30,
            )
            if resp.ok:
                data = resp.json() or {}
                vm_ready = data.get("vm_ready", False)
                vm_ready_at = data.get("vm_ready_at")
                desktop_url = data.get("desktop_url")
                status = data.get("status", "UNKNOWN")

                print(f"[Liveview] VM Status: {status}, Ready: {vm_ready}")

                if vm_ready and desktop_url:
                    print("[Liveview] ✅ Windows VM is ready!")
                    print(f"[Liveview] URL: {desktop_url}/desktop/custom.html")
                    return f"{desktop_url}/desktop/custom.html"

                # Calculate wait time from vm_ready_at timestamp (no max clamp)
                wait_time = _calc_wait_from_ready_at(vm_ready_at)
                if wait_time > 60:
                    mins, secs = divmod(wait_time, 60)
                    print(f"[Liveview] Waiting {mins}m {secs}s...")
                else:
                    print(f"[Liveview] Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"[Liveview] Request failed: {resp.status_code} {resp.text}")
                time.sleep(10)
        except Exception as e:
            print(f"[Liveview] Error: {e}")
            time.sleep(10)

    return None


def _resolve_k8s_liveview(job_name: str) -> str | None:
    """Resolve liveview URL for a K8s job by polling /infra/job/service/ip endpoint.

    Returns the liveview URL when ready, or None if resolution fails.
    """
    comms_url = SETTINGS.conversation.COMMS_URL.rstrip("/")
    admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()
    if not comms_url or not admin_key:
        print("[Liveview] Skipping: COMMS_URL or admin key not configured")
        return None

    max_retries = 5  # Cap retries to avoid infinite loops
    svc = f"unity-svc-{job_name}"
    for attempt in range(max_retries):
        print(
            f"\n\n[Liveview] Attempt {attempt + 1} to get liveview URL for job {job_name}",
        )
        try:
            resp = requests.get(
                f"{comms_url}/infra/job/service/ip",
                params={"service_name": svc},
                headers={"Authorization": f"Bearer {admin_key}"},
                timeout=30,
            )
            if resp.ok:
                data = resp.json() or {}
                external = data.get("external", {})
                ready = external.get("ready", False)
                checks = external.get("checks", {})
                seconds_left = checks.get("seconds_until_ready", 0)

                print(f"[Liveview] Ready: {ready}")
                print(
                    f"[Liveview]  - service_exists: {checks.get('service_exists')}",
                )
                print(
                    f"[Liveview]  - gce_lb_wait_passed: {checks.get('gce_lb_wait_passed')}",
                )
                if seconds_left > 0:
                    mins, secs = divmod(seconds_left, 60)
                    print(f"[Liveview]  - time_until_ready: {mins}m {secs}s")

                if ready:
                    liveview_url = external.get("url")
                    liveview_url = f"{liveview_url}/custom.html"
                    print("[Liveview] ✅ Service is ready!")
                    print(f"[Liveview] URL: {liveview_url}")
                    return liveview_url

                # Wait for seconds_left, minimum 5 seconds
                wait_time = max(seconds_left - 10, 5) if seconds_left > 0 else 5
                print(f"[Liveview] Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(
                    f"[Liveview] Request failed with status {resp.status_code}: {resp.text}",
                )
                # Wait a bit before retrying on failure
                time.sleep(5)
        except Exception as e:
            print(f"[Liveview] Error: {e}")
            time.sleep(5)

    return None


def log_job_startup(job_name: str, user_id: str, assistant_id: str):
    """Update the running job record with job_name and liveview_url.

    The adapter already created the running=True record with all assistant info.
    This function just adds the container-specific details: job_name and liveview_url.
    """
    api_key = SESSION_DETAILS.shared_unify_key or None
    if not api_key:
        print("[debug_logger] Skipping log_job_startup: no shared API key available")
        return

    _ensure_project_exists(api_key)

    # Resolve liveview URL first (this can take a while)
    try:
        is_windows_vm = (
            not SESSION_DETAILS.assistant.is_user_desktop
            and SESSION_DETAILS.assistant.desktop_mode == "windows"
        )

        if is_windows_vm:
            liveview_url = _resolve_windows_vm_liveview(assistant_id)
        else:
            liveview_url = _resolve_k8s_liveview(job_name)
    except Exception as e:
        print(f"[Liveview] Error resolving liveview URL: {e}")
        traceback.print_exc()
        liveview_url = None

    # Update the existing record (created by adapter) with job_name and liveview_url
    try:
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
        print(f"[debug_logger] Found {len(existing_logs)} running records")

        if existing_logs:
            log = existing_logs[0]
            log.update_entries(job_name=job_name, liveview_url=liveview_url)
            print(
                f"[debug_logger] Updated record with job_name={job_name}, "
                f"liveview_url={liveview_url}"
            )
        else:
            # No record found - adapter's mark_job_running() must have failed
            # Log warning but don't fail; liveview just won't be tracked
            print(
                f"[debug_logger] WARNING: No running record found for "
                f"user_id={user_id}, assistant_id={assistant_id}. "
                f"Adapter may have failed to create the record."
            )
    except Exception as e:
        print(f"[debug_logger] Error updating job record: {e}")
        traceback.print_exc()


def _stop_windows_vm(assistant_id: str) -> None:
    """Stop the Windows VM for the given assistant.

    Called when a job is marked done and the assistant was running on a
    non-user Windows VM (is_user_desktop=False and desktop_mode=windows).
    """
    try:
        comms_url = SETTINGS.conversation.COMMS_URL.rstrip("/")
        admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()
        if not comms_url or not admin_key:
            print(
                "[debug_logger] Skipping Windows VM stop: "
                "COMMS_URL or admin key not configured",
            )
            return

        response = requests.post(
            f"{comms_url}/infra/vm/stop",
            json={"assistant_id": assistant_id},
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=60,
        )
        if response.ok:
            print(
                f"[debug_logger] Windows VM stopped for assistant {assistant_id}: "
                f"{response.json()}",
            )
        else:
            print(
                f"[debug_logger] Failed to stop Windows VM: "
                f"{response.status_code} {response.text}",
            )
    except requests.exceptions.Timeout:
        print("[debug_logger] Windows VM stop request timed out")
    except Exception as e:
        print(f"[debug_logger] Error stopping Windows VM: {e}")
        traceback.print_exc()


def mark_job_done(job_name: str):
    api_key = SESSION_DETAILS.shared_unify_key or None
    if not api_key:
        print("[debug_logger] Skipping mark_job_done: no shared API key available")
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
        print("Job marked done", job_name)
    except Exception as e:
        print(f"Error finding job: {e}")
        traceback.print_exc()

    # Stop Windows VM if applicable (non-user desktop running Windows)
    if (
        not SESSION_DETAILS.assistant.is_user_desktop
        and SESSION_DETAILS.assistant.desktop_mode == "windows"
    ):
        _stop_windows_vm(SESSION_DETAILS.assistant.id)

    # delete the job service
    try:
        comms_url = SETTINGS.conversation.COMMS_URL.rstrip("/")
        admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()
        svc = f"unity-svc-{job_name}"
        response = requests.delete(
            f"{comms_url}/infra/job/service",
            data={"service_name": svc},
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=3,
        )
        print(f"Job service deleted: {response.text}")
    except requests.exceptions.Timeout:
        print("Job service deletion (timed out)")
    except Exception as e:
        print(f"Error deleting job service: {e}")
        traceback.print_exc()
