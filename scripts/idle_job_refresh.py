#!/usr/bin/env python3
"""
Refresh idle Unity jobs by creating new ones and cleaning up stale ones.

Intended to run after a Unity Cloud Build completes, replacing the
hourly Cloud Scheduler cron with an event-driven trigger.

Usage:
    python scripts/idle_job_refresh.py                 # prod (default)
    python scripts/idle_job_refresh.py --staging       # staging
    python scripts/idle_job_refresh.py --delay 45
"""

import argparse
import os
import sys
import time

import requests

ADAPTERS_URLS = {
    "prod": "https://unity-adapters-1021024874437.us-central1.run.app",
    "staging": "https://unity-adapters-staging-ky4ja5fxna-uc.a.run.app",
}


def refresh_idle_jobs(adapters_url: str, delay: int = 30):
    create_url = f"{adapters_url}/scheduled/jobs/create"
    cleanup_url = f"{adapters_url}/scheduled/jobs/cleanup"

    for i in range(1, 3):
        print(f"[{i}/2] Creating idle job via {create_url}")
        try:
            resp = requests.post(create_url, timeout=120)
            resp.raise_for_status()
            print(f"       Response: {resp.json()}")
        except Exception as e:
            print(f"       Failed: {e}", file=sys.stderr)

    print(f"Waiting {delay}s for jobs to register as idle...")
    time.sleep(delay)

    print(f"Cleaning up stale idle jobs via {cleanup_url}")
    try:
        resp = requests.post(cleanup_url, timeout=120)
        resp.raise_for_status()
        print(f"       Response: {resp.json()}")
    except Exception as e:
        print(f"       Failed: {e}", file=sys.stderr)

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
        "--adapters-url",
        default=None,
        help="Override the adapters URL (ignores --staging)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=30,
        help="Seconds to wait between job creation and cleanup (default: 30)",
    )
    args = parser.parse_args()

    if args.adapters_url:
        url = args.adapters_url
    else:
        env = "staging" if args.staging else "prod"
        url = os.getenv("UNITY_ADAPTERS_URL", ADAPTERS_URLS[env])
        print(f"Environment: {env}")

    refresh_idle_jobs(url, args.delay)


if __name__ == "__main__":
    main()
