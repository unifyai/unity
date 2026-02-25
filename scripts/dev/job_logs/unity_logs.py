#!/usr/bin/env python3
"""
unity_logs.py — View logs for a Unity GKE job.

Usage:
    python unity_logs.py                       # auto-detect latest staging job
    python unity_logs.py --job <job_name>      # explicit job, staging namespace
    python unity_logs.py --namespace production # auto-detect latest production job

Behaviour:
    1. If --job is omitted, resolves the caller's email from UNIFY_KEY and finds
       the most recent job for that email in the AssistantJobs project.
    2. Queries the AssistantJobs Unify project to check if the job is running.
    3. If running  → prints all existing logs AND streams new ones via kubectl -f.
    4. If not running → prints historical logs via gcloud.

Environment:
    UNIFY_KEY          Required for auto-detection (resolves caller identity).
    SHARED_UNIFY_KEY   Required. Shared API key for the AssistantJobs project.
"""

import os
import sys

# The unify SDK reads ORCHESTRA_URL at import time, and .env sets it to
# localhost for local development. This script needs the real backend, so
# derive the URL from --namespace before importing anything else.
_ORCHESTRA_URLS = {
    "staging": "https://orchestra-staging-lz5fmz6i7q-ew.a.run.app/v0",
    "production": "https://api.unify.ai/v0",
}


def _parse_namespace_early() -> str:
    for i, arg in enumerate(sys.argv):
        if arg == "--namespace" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--namespace="):
            return arg.split("=", 1)[1]
    return "staging"


os.environ["ORCHESTRA_URL"] = _ORCHESTRA_URLS[_parse_namespace_early()]

from dotenv import load_dotenv

load_dotenv()
import argparse
import shutil
import subprocess

import unify

# ─── Configuration ───────────────────────────────────────────────────────────

GCP_PROJECT = "responsive-city-458413-a2"
GKE_CLUSTER = "unity"
GKE_REGION = "us-central1"
SHARED_UNIFY_KEY = os.environ["SHARED_UNIFY_KEY"]


# ─── Colours ─────────────────────────────────────────────────────────────────

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
DIM = "\033[2m"
NC = "\033[0m"


def info(msg):
    print(f"{CYAN}[INFO]{NC} {msg}")


def warn(msg):
    print(f"{YELLOW}[WARN]{NC} {msg}")


def error(msg):
    print(f"{RED}[ERROR]{NC} {msg}", file=sys.stderr)


def success(msg):
    print(f"{GREEN}[OK]{NC} {msg}")


# ─── Prerequisite checks ────────────────────────────────────────────────────


def check_prerequisites():
    """Verify that required CLI tools and env vars are available."""
    missing = []
    for tool in ("gcloud", "kubectl"):
        if shutil.which(tool) is None:
            missing.append(tool)

    if missing:
        error(f"Missing required tools: {', '.join(missing)}")
        print("  Run the auth setup script first:  ./setup_auth.sh")
        sys.exit(1)

    if not os.environ.get("SHARED_UNIFY_KEY"):
        error("SHARED_UNIFY_KEY environment variable is not set.")
        print()
        print("  This key is needed to query the AssistantJobs project.")
        print("  Ask a team member for the shared Unify API key, then:")
        print()
        print("    export SHARED_UNIFY_KEY='your_key_here'")
        print()
        print("  (Add to ~/.zshrc or ~/.bashrc to persist across sessions.)")
        sys.exit(1)


def ensure_gke_credentials():
    """Silently refresh GKE cluster credentials."""
    result = subprocess.run(
        [
            "gcloud",
            "container",
            "clusters",
            "get-credentials",
            GKE_CLUSTER,
            "--region",
            GKE_REGION,
            "--project",
            GCP_PROJECT,
            "--quiet",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        error("Failed to get GKE cluster credentials.")
        print("  Run ./setup_auth.sh to configure GCP authentication.")
        sys.exit(1)


# ─── Job auto-detection ──────────────────────────────────────────────────────


def resolve_latest_job(namespace: str) -> str:
    """Resolve the most recent job for the current user in the given namespace.

    Uses UNIFY_KEY to identify the caller, then queries AssistantJobs for
    their most recent job whose name ends with ``-{namespace}``.
    """
    unify_key = os.environ.get("UNIFY_KEY")
    if not unify_key:
        error("UNIFY_KEY is required to auto-detect jobs.")
        print("  Set it in .env or pass --job explicitly.")
        sys.exit(1)

    info("Resolving identity from UNIFY_KEY...")
    user_info = unify.get_user_basic_info(api_key=unify_key)
    email = user_info["email"]
    info(f"Authenticated as {user_info['first']} {user_info['last']} ({email})")

    info(f"Searching for latest '{namespace}' job...")
    logs = unify.get_logs(
        project="AssistantJobs",
        context="startup_events",
        filter=f"user_email == '{email}'",
        api_key=SHARED_UNIFY_KEY,
        limit=20,
    )

    suffix = f"-{namespace}"
    for log in logs:
        job_name = log.entries.get("job_name")
        if job_name and job_name.endswith(suffix):
            running = str(log.entries.get("running", "false")).lower() == "true"
            status = f"{GREEN}running{NC}" if running else f"{YELLOW}completed{NC}"
            success(f"Found job: {job_name} ({status})")
            return job_name

    error(f"No jobs found for {email} in namespace '{namespace}'.")
    sys.exit(1)


# ─── AssistantJobs query ─────────────────────────────────────────────────────


def query_assistant_jobs(job_name: str) -> dict | None:
    """Query the AssistantJobs project for this job.

    Returns the log entries dict if found, None otherwise.
    Uses the same unify.get_logs() pattern as debug_logger.py.
    """
    try:
        logs = unify.get_logs(
            project="AssistantJobs",
            context="startup_events",
            filter=f"job_name == '{job_name}'",
            api_key=SHARED_UNIFY_KEY,
        )
        if logs:
            return logs[0].entries
    except Exception as e:
        warn(f"Could not query AssistantJobs: {e}")

    return None


def query_job_status(job_name: str) -> tuple[bool | None, dict | None]:
    """Query AssistantJobs for the job's running status.

    Returns (is_running, metadata_dict_or_none).
    is_running is None when no AssistantJobs record exists.
    """
    info(f"Querying AssistantJobs for job '{job_name}'...")

    entry = query_assistant_jobs(job_name)

    if entry is not None:
        # Print session metadata.
        assistant = entry.get("assistant_name", "unknown")
        assistant_id = entry.get("assistant_id", "?")
        user = entry.get("user_name", "unknown")
        medium = entry.get("medium", "unknown")
        timestamp = entry.get("timestamp", "unknown")

        print(f"\n  Assistant : {assistant}  (ID {assistant_id})")
        print(f"  User      : {user}")
        print(f"  Medium    : {medium}")
        print(f"  Started   : {timestamp}\n")

        running = str(entry.get("running", "false")).lower() == "true"
        if running:
            success(f"Job is currently {GREEN}running{NC}.")
        else:
            info(f"Job is {YELLOW}not running{NC} (completed/suspended).")
        return running, entry

    warn("No AssistantJobs record found.")
    return None, None


# ─── Log output ──────────────────────────────────────────────────────────────


def stream_logs(job_name: str, namespace: str):
    """Print all existing logs and stream new ones via kubectl -f.

    kubectl logs job/<name> -f --tail=-1 prints the full log history
    and then continues streaming as new lines arrive.
    """
    info(f"Streaming logs (existing + live) for '{job_name}'...")
    print(f"  {DIM}(Press Ctrl+C to stop){NC}")
    print()

    try:
        subprocess.run(
            [
                "kubectl",
                "logs",
                f"job/{job_name}",
                "-n",
                namespace,
                "-f",
                "--tail=-1",
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        warn("kubectl could not attach. Retrying in 5s...")
        import time

        time.sleep(5)
        subprocess.run(
            [
                "kubectl",
                "logs",
                f"job/{job_name}",
                "-n",
                namespace,
                "-f",
                "--tail=-1",
            ],
        )
    except KeyboardInterrupt:
        print()
        info("Stopped streaming.")


def _gcloud_logging_read(log_filter: str):
    """Run gcloud logging read with the given filter."""
    subprocess.run(
        [
            "gcloud",
            "logging",
            "read",
            log_filter,
            f"--project={GCP_PROJECT}",
            "--freshness=3650d",
            "--format=value(textPayload)",
            "--order=asc",
        ],
    )


def fetch_historical_logs(job_name: str, namespace: str):
    """Fetch historical logs from Cloud Logging.

    Called when the running flag is False, meaning the pod is gone.
    Goes straight to gcloud — no kubectl attempt needed.
    """
    info(f"Fetching historical logs from Cloud Logging for '{job_name}'...")
    print()

    log_filter = (
        f'resource.type="k8s_container"'
        f' AND resource.labels.cluster_name="{GKE_CLUSTER}"'
        f' AND resource.labels.namespace_name="{namespace}"'
        f' AND labels."k8s-pod/batch_kubernetes_io/job-name"="{job_name}"'
    )
    _gcloud_logging_read(log_filter)


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="View logs for a Unity GKE job.",
        epilog=(
            "When --job is omitted, the script auto-detects the latest job\n"
            "by resolving your identity from UNIFY_KEY and searching\n"
            "AssistantJobs for your most recent session.\n"
            "\n"
            "Examples:\n"
            "  python unity_logs.py                        # latest staging job\n"
            "  python unity_logs.py --namespace production  # latest production job\n"
            "  python unity_logs.py --job unity-2026-02-10-17-30-53-staging"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--job",
        default=None,
        help="Name of the GKE job. If omitted, auto-detects the latest job for your account.",
    )
    parser.add_argument(
        "--namespace",
        default="staging",
        help="Kubernetes namespace (default: staging)",
    )
    args = parser.parse_args()

    namespace = args.namespace
    job_name = args.job or resolve_latest_job(namespace)

    check_prerequisites()
    ensure_gke_credentials()

    print()
    print(f"{BOLD}{'═' * 56}{NC}")
    print(f"{BOLD}  Unity GKE Job Logs{NC}")
    print(f"  Job:       {CYAN}{job_name}{NC}")
    print(f"  Namespace: {CYAN}{namespace}{NC}")
    print(f"{BOLD}{'═' * 56}{NC}")
    print()

    running, _ = query_job_status(job_name)

    print()
    if running is True:
        # AssistantJobs says running → pod exists, stream live.
        stream_logs(job_name, namespace)
    elif running is False:
        # AssistantJobs says not running → pod is gone, use gcloud.
        fetch_historical_logs(job_name, namespace)
    else:
        # No AssistantJobs record. Try streaming; if kubectl fails, fall
        # back to gcloud historical logs.
        warn("Unknown job status. Trying kubectl stream first...")
        print()
        stream_logs(job_name, namespace)


if __name__ == "__main__":
    main()
