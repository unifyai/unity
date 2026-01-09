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


def log_job_startup(
    job_name: str,
    timestamp: str,
    medium: str,
    user_id: str,
    assistant_id: str,
    user_name: str,
    assistant_name: str,
    user_number: str,
    assistant_number: str,
    user_email: str,
    assistant_email: str,
):
    api_key = SESSION_DETAILS.shared_unify_key or None
    if not api_key:
        print("[debug_logger] Skipping log_job_startup: no shared API key available")
        return

    _ensure_project_exists(api_key)

    try:
        # Create startup event log and get log instance
        unify.create_logs(
            project="AssistantJobs",
            context="startup_events",
            entries=[
                {
                    "job_name": job_name,
                    "timestamp": timestamp,
                    "medium": medium,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "user_name": user_name,
                    "assistant_name": assistant_name,
                    "user_number": user_number,
                    "assistant_number": assistant_number,
                    "user_email": user_email,
                    "assistant_email": assistant_email,
                    "running": True,
                },
            ],
            api_key=api_key,
        )
        log = unify.get_logs(
            project="AssistantJobs",
            context="startup_events",
            filter=f"job_name == '{job_name}'",
            api_key=api_key,
        )
        print("Debug Logger - Logs:", log)
        log = log[0]
        print("Logged Startup Event", job_name)
    except Exception as e:
        print(f"Error logging startup event: {e}")
        traceback.print_exc()

    try:
        # Resolve liveview URL via comms infra service
        liveview_url = None
        max_retries = 5  # Cap retries to avoid infinite loops
        comms_url = SETTINGS.conversation.COMMS_URL.rstrip("/")
        admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY
        if comms_url and admin_key and job_name:
            svc = f"unity-svc-{job_name}"
            for attempt in range(max_retries):
                print(
                    f"\n\n[Liveview] Attempt {attempt + 1} to get liveview URL for job {job_name}",
                )
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
                        print(f"[Liveview] ✅ Service is ready!")
                        print(f"[Liveview] URL: {liveview_url}")
                        break

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
        log.update_entries(liveview_url=liveview_url)
        print("[Liveview] Updated log with liveview URL:", job_name)
    except Exception as e:
        print(f"[Liveview] Error resolving liveview URL: {e}")
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

    # delete the job service
    try:
        comms_url = SETTINGS.conversation.COMMS_URL.rstrip("/")
        admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY
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
