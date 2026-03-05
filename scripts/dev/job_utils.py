"""Shared utilities for Unity GKE job management scripts."""

import os
import sys

ORCHESTRA_URLS = {
    "staging": "https://api.staging.internal.saas.unify.ai/v0",
    "production": "https://api.unify.ai/v0",
}

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


def _find_latest_job_entry(namespace: str, *, running_only: bool = False):
    """Return the log entry for the latest matching job, or exit."""
    import unify

    shared_key = os.environ.get("SHARED_UNIFY_KEY")
    if not shared_key:
        error("SHARED_UNIFY_KEY environment variable is not set.")
        sys.exit(1)

    unify_key = os.environ.get("UNIFY_KEY")
    if not unify_key:
        error("UNIFY_KEY is required to auto-detect jobs.")
        print("  Set it in .env or pass the job name explicitly.")
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
        api_key=shared_key,
        limit=20,
    )

    suffix = f"-{namespace}"
    for log in logs:
        job_name = log.entries.get("job_name")
        if not (job_name and job_name.endswith(suffix)):
            continue
        running = str(log.entries.get("running", "false")).lower() == "true"
        if running_only and not running:
            continue
        status = f"{GREEN}running{NC}" if running else f"{YELLOW}completed{NC}"
        success(f"Found job: {job_name} ({status})")
        return log

    qualifier = "running " if running_only else ""
    error(f"No {qualifier}jobs found for {email} in namespace '{namespace}'.")
    sys.exit(1)


def resolve_latest_job(namespace: str, *, running_only: bool = False) -> str:
    """Resolve the most recent job name for the current user.

    Uses UNIFY_KEY to identify the caller, then queries AssistantJobs for
    their most recent job whose name ends with ``-{namespace}``.

    When *running_only* is True, only jobs with running status are
    considered — useful for suspension where completed jobs are irrelevant.
    """
    return _find_latest_job_entry(namespace, running_only=running_only).entries[
        "job_name"
    ]


def resolve_latest_assistant_id(namespace: str) -> str:
    """Resolve the assistant_id for the most recent running job.

    Uses UNIFY_KEY to identify the caller, then queries AssistantJobs for
    their most recent *running* job and returns its assistant_id.
    """
    entry = _find_latest_job_entry(namespace, running_only=True).entries
    assistant_id = entry.get("assistant_id")
    if not assistant_id:
        error(f"Job '{entry.get('job_name')}' has no assistant_id in metadata.")
        sys.exit(1)
    return str(assistant_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Job utilities CLI")
    parser.add_argument(
        "command",
        choices=["assistant-id"],
    )
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()

    namespace = "production" if args.production else "staging"
    os.environ["ORCHESTRA_URL"] = ORCHESTRA_URLS[namespace]

    from dotenv import load_dotenv

    load_dotenv()

    if args.command == "assistant-id":
        print(resolve_latest_assistant_id(namespace))
