#!/usr/bin/env python3
"""
Suspend a running Unity Kubernetes job.

Usage:
    python scripts/dev/suspend_job.py                                           # auto-detect latest running staging job
    python scripts/dev/suspend_job.py unity-2026-02-25-12-00-00                 # explicit job, staging (default)
    python scripts/dev/suspend_job.py --env production                          # auto-detect latest running production job
    python scripts/dev/suspend_job.py --env preview                             # auto-detect latest running preview job
    python scripts/dev/suspend_job.py unity-2026-02-25-12-00-00 --namespace my-ns
"""

import os
import sys

from job_utils import ORCHESTRA_URLS


def _parse_namespace_early() -> str:
    for i, arg in enumerate(sys.argv):
        if arg == "--env" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "staging"


os.environ["ORCHESTRA_URL"] = ORCHESTRA_URLS[_parse_namespace_early()]

from dotenv import load_dotenv

load_dotenv()

import argparse

import requests

from job_utils import resolve_latest_job

COMMS_URLS = {
    "production": "https://unity-comms-app-262420637606.us-central1.run.app",
    "staging": "https://unity-comms-app-staging-262420637606.us-central1.run.app",
    "preview": "https://unity-comms-app-preview-262420637606.us-central1.run.app",
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
        print(
            f"  Failed ({e.response.status_code}): {e.response.text}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"  Failed: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Suspend a running Unity Kubernetes job.",
        epilog=(
            "When no job name is provided, the script auto-detects the latest\n"
            "running job by resolving your identity from UNIFY_KEY and searching\n"
            "AssistantJobs for your most recent active session.\n"
            "\n"
            "Examples:\n"
            "  python scripts/dev/suspend_job.py                          # latest running staging job\n"
            "  python scripts/dev/suspend_job.py --env production         # latest running production job\n"
            "  python scripts/dev/suspend_job.py --env preview            # latest running preview job\n"
            "  python scripts/dev/suspend_job.py unity-2026-02-25-12-00-00-staging"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "job_name",
        nargs="?",
        default=None,
        help="Name of the K8s job to suspend. If omitted, auto-detects the latest running job for your account.",
    )
    parser.add_argument(
        "--env",
        choices=["production", "staging", "preview"],
        default="staging",
        help="Target deploy environment (default: staging)",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="K8s namespace (defaults to the value of --env)",
    )
    args = parser.parse_args()

    env = args.env
    namespace = args.namespace or env
    comms_url = COMMS_URLS[env]
    admin_key = os.getenv("ORCHESTRA_ADMIN_KEY")

    job_name = args.job_name or resolve_latest_job(namespace, running_only=True)

    print(f"Environment: {env}")
    suspend_job(comms_url, admin_key, job_name, namespace)


if __name__ == "__main__":
    main()
