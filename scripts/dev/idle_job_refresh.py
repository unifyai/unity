#!/usr/bin/env python3
"""
Refresh idle Unity jobs by creating new ones and cleaning up stale ones.

Intended to run after a Unity Cloud Build completes, replacing the
hourly Cloud Scheduler cron with an event-driven trigger.

Usage:
    python scripts/dev/idle_job_refresh.py                 # prod (default, lists jobs)
    python scripts/dev/idle_job_refresh.py --staging       # staging
    python scripts/dev/idle_job_refresh.py --no-list-jobs  # skip job listing
    python scripts/dev/idle_job_refresh.py --delay 45
"""

from dotenv import load_dotenv
import argparse
import os
import sys
import time

import requests

load_dotenv()

ADAPTERS_URLS = {
    "prod": "https://unity-adapters-1021024874437.us-central1.run.app",
    "staging": "https://unity-adapters-staging-ky4ja5fxna-uc.a.run.app",
}
COMMS_URLS = {
    "prod": "https://unity-comms-app-262420637606.us-central1.run.app",
    "staging": "https://unity-comms-app-staging-262420637606.us-central1.run.app",
}


def list_jobs(comms_url: str, admin_key: str, label: str):
    """Fetch and print current jobs from the comms /infra/jobs endpoint."""
    headers = {"Authorization": f"Bearer {admin_key}"}
    try:
        resp = requests.get(
            f"{comms_url}/infra/jobs",
            params={"label_selector": "app=unity"},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        jobs = data.get("jobs", [])
        names = [job["job_name"] for job in jobs]
        print(f"\n--- {label} ({len(names)} jobs) ---")
        for name in names:
            print(f"  {name}")
        if not names:
            print("  (none)")
        print()
    except Exception as e:
        print(f"  Failed to list jobs: {e}", file=sys.stderr)


def refresh_idle_jobs(
    adapters_url: str,
    delay: int = 30,
    comms_url: str | None = None,
    admin_key: str | None = None,
):
    show_jobs = comms_url and admin_key
    create_url = f"{adapters_url}/scheduled/jobs/create"
    cleanup_url = f"{adapters_url}/scheduled/jobs/cleanup"

    if show_jobs:
        list_jobs(comms_url, admin_key, "Before job creation")

    for i in range(1, 3):
        print(f"[{i}/2] Creating idle job via {create_url}")
        try:
            resp = requests.post(create_url, timeout=120)
            resp.raise_for_status()
            print(f"       Response: {resp.json()}")
        except Exception as e:
            print(f"       Failed: {e}", file=sys.stderr)

    if show_jobs:
        list_jobs(comms_url, admin_key, "After job creation")

    print(f"Waiting {delay}s for jobs to register as idle...")
    time.sleep(delay)

    print(f"Cleaning up stale idle jobs via {cleanup_url}")
    try:
        resp = requests.post(cleanup_url, timeout=120)
        resp.raise_for_status()
        print(f"       Response: {resp.json()}")
    except Exception as e:
        print(f"       Failed: {e}", file=sys.stderr)

    if show_jobs:
        list_jobs(comms_url, admin_key, "After cleanup")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Create fresh idle jobs and clean up stale ones.",
    )
    parser.add_argument(
        "--staging",
        action="store_true",
        help="Target the staging environment (default: prod)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=30,
        help="Seconds to wait between job creation and cleanup (default: 30)",
    )
    parser.add_argument(
        "--no-list-jobs",
        action="store_true",
        help="Disable job listing at each step",
    )
    args = parser.parse_args()

    env = "staging" if args.staging else "prod"
    adapters_url = ADAPTERS_URLS[env]
    print(f"Environment: {env}")

    comms_url = None
    admin_key = None
    if not args.no_list_jobs:
        comms_url = COMMS_URLS[env]
        admin_key = os.getenv("ORCHESTRA_ADMIN_KEY")

    refresh_idle_jobs(adapters_url, args.delay, comms_url, admin_key)

if __name__ == "__main__":
    main()
