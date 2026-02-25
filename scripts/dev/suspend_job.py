#!/usr/bin/env python3
"""
Suspend a running Unity Kubernetes job by name.

Usage:
    python scripts/dev/suspend_job.py unity-2026-02-25-12-00-00
    python scripts/dev/suspend_job.py unity-2026-02-25-12-00-00 --staging
    python scripts/dev/suspend_job.py unity-2026-02-25-12-00-00 --namespace staging
"""

import argparse
import os
import sys

import requests

COMMS_URLS = {
    "prod": os.getenv("UNITY_COMMS_URL", ""),
    "staging": os.getenv("UNITY_COMMS_URL", ""),
}

DEFAULT_NAMESPACES = {
    "prod": "production",
    "staging": "staging",
}


def suspend_job(comms_url: str, admin_key: str, job_name: str, namespace: str):
    headers = {"Authorization": f"Bearer {admin_key}"}
    print(f"Suspending job '{job_name}' in namespace '{namespace}'...")
    try:
        resp = requests.post(
            f"{comms_url}/infra/job/stop",
            data={"job_name": job_name, "namespace": namespace},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        print(f"  {data.get('message', data)}")
    except requests.exceptions.HTTPError as e:
        print(f"  Failed ({e.response.status_code}): {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"  Failed: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Suspend a running Unity Kubernetes job.",
    )
    parser.add_argument(
        "job_name",
        help="Name of the K8s job to suspend (e.g. unity-2026-02-25-12-00-00)",
    )
    parser.add_argument(
        "--staging",
        action="store_true",
        help="Target the staging environment (default: prod)",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="K8s namespace (default: 'production' for prod, 'staging' for staging)",
    )
    parser.add_argument(
        "--comms-url",
        default=None,
        help="Override the comms service URL",
    )
    parser.add_argument(
        "--admin-key",
        default=None,
        help="Override the admin key",
    )
    args = parser.parse_args()

    env = "staging" if args.staging else "prod"
    comms_url = args.comms_url or os.getenv("UNITY_COMMS_URL")
    admin_key = args.admin_key or os.getenv("ORCHESTRA_ADMIN_KEY")
    namespace = args.namespace or DEFAULT_NAMESPACES[env]

    if not comms_url or not admin_key:
        parser.error(
            "Requires UNITY_COMMS_URL and ORCHESTRA_ADMIN_KEY env vars "
            "(or --comms-url and --admin-key)"
        )

    print(f"Environment: {env}")
    suspend_job(comms_url, admin_key, args.job_name, namespace)


if __name__ == "__main__":
    main()
