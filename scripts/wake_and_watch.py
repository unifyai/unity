#!/usr/bin/env python3
"""
End-to-end test helper: refresh idle containers, wake up assistant, stream logs.

Combines three steps into a single workflow for testing local changes
after a cloud build completes (~10 minutes):

  1. Refreshes idle GKE containers to use the latest Docker image
  2. Wakes up the assistant (assigns a fresh idle container)
  3. Streams the assistant's container logs

Usage:
    python scripts/wake_and_watch.py                        # staging (default)
    python scripts/wake_and_watch.py --env production      # production
    python scripts/wake_and_watch.py --env preview         # preview
    python scripts/wake_and_watch.py --assistant-id 464    # explicit assistant
    python scripts/wake_and_watch.py --skip-refresh        # skip idle container refresh

Environment:
    UNIFY_KEY          Required. Resolves caller identity and assistant.
    SHARED_UNIFY_KEY   Required. Queries the AssistantJobs project.
"""

import argparse
import os
import sys
import time

# ─── Constants ────────────────────────────────────────────────────────────────

ORCHESTRA_URLS = {
    "production": "https://api.unify.ai/v0",
    "staging": "https://api.staging.internal.saas.unify.ai/v0",
    "preview": "https://api.staging.internal.saas.unify.ai/v0",
}

ADAPTERS_URLS = {
    "production": "https://unity-adapters-1021024874437.us-central1.run.app",
    "staging": "https://unity-adapters-staging-ky4ja5fxna-uc.a.run.app",
    "preview": "https://unity-adapters-preview-ky4ja5fxna-uc.a.run.app",
}

# ─── Early env setup (before imports that read ORCHESTRA_URL) ─────────────────


def _parse_env_early() -> str:
    for i, arg in enumerate(sys.argv):
        if arg == "--env" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "staging"


ENV = _parse_env_early()
os.environ["ORCHESTRA_URL"] = ORCHESTRA_URLS[ENV]

from dotenv import load_dotenv

load_dotenv()
os.environ["ORCHESTRA_URL"] = ORCHESTRA_URLS[ENV]

import requests
import unify

SHARED_UNIFY_KEY = os.environ["SHARED_UNIFY_KEY"]

# ─── Colours ──────────────────────────────────────────────────────────────────

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
NC = "\033[0m"


def info(msg):
    print(f"{CYAN}ℹ{NC}  {msg}", flush=True)


def warn(msg):
    print(f"{YELLOW}⚠{NC}  {msg}", flush=True)


def error(msg):
    print(f"{RED}✗{NC}  {msg}", file=sys.stderr, flush=True)


def success(msg):
    print(f"{GREEN}✓{NC}  {msg}", flush=True)


# ─── Assistant resolution ─────────────────────────────────────────────────────


def resolve_assistant_id() -> str:
    unify_key = os.environ.get("UNIFY_KEY")
    if not unify_key:
        error("UNIFY_KEY required to auto-detect assistant.")
        print("  Set it in .env or pass --assistant-id.")
        sys.exit(1)

    info("Resolving assistant from UNIFY_KEY...")
    user_info = unify.get_user_basic_info(api_key=unify_key)
    info(
        f"Authenticated as {user_info['first']} {user_info['last']} ({user_info['email']})",
    )

    assistants = unify.list_assistants(api_key=unify_key)
    if not assistants:
        error("No assistants found for this account.")
        sys.exit(1)

    if len(assistants) == 1:
        a = assistants[0]
        aid = str(a["agent_id"])
        success(f"Found assistant: {a['first_name']} {a['surname']} (ID {aid})")
        return aid

    print("\n  Multiple assistants found:")
    for a in assistants:
        print(f"    {a['agent_id']}: {a['first_name']} {a['surname']}")
    error("Please specify --assistant-id")
    sys.exit(1)


def resolve_user_email() -> str:
    unify_key = os.environ.get("UNIFY_KEY")
    user_info = unify.get_user_basic_info(api_key=unify_key)
    return user_info["email"]


# ─── Step 1: Refresh idle containers ─────────────────────────────────────────


def refresh_idle_containers(adapters_url: str, delay: int = 30):
    create_url = f"{adapters_url}/scheduled/jobs/create"
    cleanup_url = f"{adapters_url}/scheduled/jobs/cleanup"

    for i in range(1, 3):
        info(f"[{i}/2] Creating idle job...")
        try:
            resp = requests.post(create_url, timeout=120)
            resp.raise_for_status()
            print(f"       {resp.json()}", flush=True)
        except Exception as e:
            error(f"Failed to create idle job: {e}")

    info(f"Waiting {delay}s for new jobs to register as idle...")
    time.sleep(delay)

    info("Cleaning up stale idle jobs...")
    try:
        resp = requests.post(cleanup_url, timeout=120)
        resp.raise_for_status()
        print(f"       {resp.json()}", flush=True)
    except Exception as e:
        error(f"Failed to clean up stale jobs: {e}")

    success("Idle container refresh complete.")


# ─── Step 2: Wake up assistant ────────────────────────────────────────────────


def wake_up_assistant(adapters_url: str, assistant_id: str):
    wakeup_url = f"{adapters_url}/assistant/wakeup"
    info(f"Sending wakeup for assistant {assistant_id}...")
    try:
        resp = requests.post(
            wakeup_url,
            data={"assistant_id": assistant_id},
            timeout=120,
        )
        resp.raise_for_status()
        success(f"Wakeup accepted (HTTP {resp.status_code}).")
    except Exception as e:
        error(f"Wakeup failed: {e}")
        sys.exit(1)


# ─── Step 3: Wait for job, then stream logs ──────────────────────────────────


def get_existing_job_names(user_email: str) -> set[str]:
    """Snapshot current job names so we can detect a new one after wakeup."""
    namespace_suffix = f"-{ENV}"
    names = set()
    try:
        logs = unify.get_logs(
            project="AssistantJobs",
            context="startup_events",
            filter=f"user_email == '{user_email}'",
            api_key=SHARED_UNIFY_KEY,
            limit=20,
        )
        for log in logs:
            jn = log.entries.get("job_name")
            if jn and jn.endswith(namespace_suffix):
                names.add(jn)
    except Exception:
        pass
    return names


def wait_for_new_job(
    user_email: str,
    old_job_names: set[str],
    timeout: int = 180,
) -> str | None:
    """Poll AssistantJobs until a new job_name appears that we haven't seen.

    The AssistantJobs record is created by Unity's ``log_job_startup``
    after the container starts, so the mere appearance of a new job_name
    means the container is alive.
    """
    namespace_suffix = f"-{ENV}"
    start = time.time()

    while time.time() - start < timeout:
        elapsed = int(time.time() - start)
        info(f"Waiting for container to start... ({elapsed}s)")

        try:
            logs = unify.get_logs(
                project="AssistantJobs",
                context="startup_events",
                filter=f"user_email == '{user_email}'",
                api_key=SHARED_UNIFY_KEY,
                limit=10,
            )
            for log in logs:
                jn = log.entries.get("job_name")
                if (
                    jn
                    and jn.endswith(namespace_suffix)
                    and jn not in old_job_names
                ):
                    return jn
        except Exception as e:
            warn(f"Poll error: {e}")

        time.sleep(10)

    return None


def stream_logs(job_name: str | None):
    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "job_logs",
        "stream_logs.py",
    )
    argv = [sys.executable, script, "--env", ENV]
    if job_name:
        argv += ["--job", job_name]
    info(f"Handing off to stream_logs.py...")
    print(flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvp(sys.executable, argv)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Refresh containers, wake assistant, stream logs.",
    )
    parser.add_argument(
        "--env",
        choices=["production", "staging", "preview"],
        default="staging",
        help="Target deploy environment (default: staging)",
    )
    parser.add_argument(
        "--assistant-id",
        default=None,
        help="Assistant ID (auto-detected from UNIFY_KEY if omitted)",
    )
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Skip the idle container refresh (step 1)",
    )
    parser.add_argument(
        "--refresh-delay",
        type=int,
        default=30,
        help="Seconds between job creation and cleanup (default: 30)",
    )
    parser.add_argument(
        "--job-timeout",
        type=int,
        default=180,
        help="Max seconds to wait for job registration (default: 180)",
    )
    args = parser.parse_args()

    adapters_url = ADAPTERS_URLS[ENV]
    assistant_id = args.assistant_id or resolve_assistant_id()
    user_email = resolve_user_email()

    print(flush=True)
    print(f"{BOLD}{'═' * 56}{NC}", flush=True)
    print(f"{BOLD}  Wake & Watch{NC}", flush=True)
    print(f"  Environment : {CYAN}{ENV}{NC}", flush=True)
    print(f"  Assistant   : {CYAN}{assistant_id}{NC}", flush=True)
    print(f"  User        : {CYAN}{user_email}{NC}", flush=True)
    print(f"{BOLD}{'═' * 56}{NC}", flush=True)
    print(flush=True)

    # Snapshot existing jobs before any changes
    old_job_names = get_existing_job_names(user_email)
    if old_job_names:
        info(f"Existing jobs: {', '.join(sorted(old_job_names))}")
    else:
        info("No existing jobs found.")
    print(flush=True)

    # Step 1: Refresh idle containers
    if not args.skip_refresh:
        info(f"{BOLD}Step 1/3: Refreshing idle containers{NC}")
        refresh_idle_containers(adapters_url, args.refresh_delay)
    else:
        info(f"{BOLD}Step 1/3: Skipped (--skip-refresh){NC}")
    print(flush=True)

    # Step 2: Wake up assistant
    info(f"{BOLD}Step 2/3: Waking up assistant{NC}")
    wake_up_assistant(adapters_url, assistant_id)
    print(flush=True)

    # Step 3: Wait for job, then stream logs
    info(f"{BOLD}Step 3/3: Streaming logs{NC}")
    job_name = wait_for_new_job(user_email, old_job_names, args.job_timeout)
    if job_name:
        success(f"New job registered: {job_name}")
    else:
        warn(f"Timed out after {args.job_timeout}s. Trying logs anyway...")
    print(flush=True)

    stream_logs(job_name)


if __name__ == "__main__":
    main()
