#!/usr/bin/env python3
"""
stream_logs.py — View logs for a Unity GKE job.

Usage:
    python stream_logs.py                       # auto-detect latest staging job
    python stream_logs.py --job <job_name>      # explicit job, staging namespace
    python stream_logs.py --production          # auto-detect latest production job

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
import re
import sys
from pathlib import Path

# Add scripts/dev/ to path for shared job_utils import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from job_utils import (
    ORCHESTRA_URLS,
    BOLD,
    CYAN,
    DIM,
    GREEN,
    NC,
    YELLOW,
    error,
    info,
    resolve_latest_job,
    success,
    warn,
)

# The unify SDK reads ORCHESTRA_URL at import time, and .env sets it to
# localhost for local development. This script needs the real backend, so
# derive the URL from --production before importing anything else.


def _parse_namespace_early() -> str:
    if "--production" in sys.argv:
        return "production"
    return "staging"


os.environ["ORCHESTRA_URL"] = ORCHESTRA_URLS[_parse_namespace_early()]

from dotenv import load_dotenv

load_dotenv()
import argparse
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import unify

from unity.syntax_highlight import (
    MARKDOWN_CLOSING_RE,
    MARKDOWN_OPENING_RE,
    highlight_code_blocks,
)

# ─── Configuration ───────────────────────────────────────────────────────────

GCP_PROJECT = "responsive-city-458413-a2"
GKE_CLUSTER = "unity"
GKE_REGION = "us-central1"
SHARED_UNIFY_KEY = os.environ["SHARED_UNIFY_KEY"]

# Resolve full paths for CLI tools (handles .cmd on Windows)
GCLOUD = shutil.which("gcloud") or "gcloud"
KUBECTL = shutil.which("kubectl") or "kubectl"


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
            GCLOUD,
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


# ─── Pod resolution ───────────────────────────────────────────────────────────


def resolve_pod_name(job_name: str, namespace: str) -> str | None:
    """Resolve the pod name for a job. Returns None if no pod is found."""
    result = subprocess.run(
        [
            KUBECTL,
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            f"job-name={job_name}",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        capture_output=True,
        text=True,
    )
    pod = result.stdout.strip()
    return pod if result.returncode == 0 and pod else None


# ─── Log mirroring ────────────────────────────────────────────────────────────

_CONTAINER_LOG_PATH_RE = re.compile(r"/var/log/(unillm|unify|unity)/\S+")

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
MIRROR_BASE = WORKSPACE_ROOT / "logs" / "prod_logs"


def _mirror_dir(job_name: str, base: Path = MIRROR_BASE) -> Path:
    d = base / job_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch_file(
    pod_name: str,
    namespace: str,
    container_path: str,
    local_path: Path,
) -> None:
    """Download a single file from the pod via kubectl exec -- cat."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [KUBECTL, "exec", pod_name, "-n", namespace, "--", "cat", container_path],
            capture_output=True,
        )
        if result.returncode == 0:
            local_path.write_bytes(result.stdout)
    except Exception:
        pass


_CONTAINER_LOG_DIRS = ("/var/log/unillm", "/var/log/unify", "/var/log/unity")


def _sync_all_logs(pod_name: str, namespace: str, mirror_root: Path) -> None:
    """Bulk-copy all container log directories to the local mirror."""
    info("Syncing all container logs...")
    for container_dir in _CONTAINER_LOG_DIRS:
        subdir = container_dir.split("/")[-1]  # unillm, unify, unity
        local_dir = mirror_root / subdir
        local_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                KUBECTL,
                "cp",
                f"{namespace}/{pod_name}:{container_dir}/.",
                str(local_dir),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            success(f"  {container_dir} → {local_dir}")
        else:
            warn(f"  {container_dir}: {result.stderr.strip() or 'copy failed'}")


def _hyperlink(uri: str, text: str) -> str:
    """Wrap *text* in an OSC 8 terminal hyperlink pointing to *uri*.

    Supported by iTerm2, VS Code terminal, Kitty, GNOME Terminal, Windows
    Terminal, and most modern VTE-based emulators. Terminals that don't
    understand OSC 8 silently ignore the escape sequences and show plain text.
    """
    return f"\033]8;;{uri}\007{text}\033]8;;\007"


def _rewrite_line(
    line: str,
    mirror_root: Path,
    pod_name: str,
    namespace: str,
    executor: ThreadPoolExecutor,
) -> str:
    """If the line contains a /var/log/... path, rewrite it to a clickable
    local hyperlink and schedule a background download."""
    match = _CONTAINER_LOG_PATH_RE.search(line)
    if not match:
        return line

    container_path = match.group(0)
    # /var/log/unillm/file.txt → unillm/file.txt
    relative = container_path[len("/var/log/") :]
    local_path = mirror_root / relative

    executor.submit(_fetch_file, pod_name, namespace, container_path, local_path)

    local_str = str(local_path)
    link = _hyperlink(local_path.as_uri(), local_str)
    return line[: match.start()] + link + line[match.end() :]


# ─── Log output ──────────────────────────────────────────────────────────────

_IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _emit_lines(
    proc: subprocess.Popen,
    *,
    rewrite_fn=None,
) -> None:
    """Read lines from *proc* stdout, apply code-block highlighting, and emit.

    When the output stream is a TTY, markdown-fenced code blocks are
    buffered and syntax-highlighted via Pygments before being written.

    *rewrite_fn*, if provided, is called on each decoded line before
    highlight processing (used for log-file hyperlink rewriting).
    """
    code_buf: list[str] = []
    in_code_block = False

    def _flush_code_block() -> None:
        nonlocal in_code_block
        block = "".join(code_buf)
        if _IS_TTY:
            block = highlight_code_blocks(block)
        sys.stdout.write(block)
        sys.stdout.flush()
        code_buf.clear()
        in_code_block = False

    for raw_line in proc.stdout:
        line = raw_line.decode("utf-8", errors="replace")
        if rewrite_fn:
            line = rewrite_fn(line)

        if in_code_block:
            code_buf.append(line)
            if MARKDOWN_CLOSING_RE.search(line) and not MARKDOWN_OPENING_RE.search(
                line,
            ):
                _flush_code_block()
            continue

        if MARKDOWN_OPENING_RE.search(line):
            in_code_block = True
            code_buf.append(line)
            continue

        sys.stdout.write(line)
        sys.stdout.flush()

    if code_buf:
        _flush_code_block()


def stream_logs(
    job_name: str,
    namespace: str,
    *,
    mirror: bool = True,
    mirror_base: Path = MIRROR_BASE,
    sync_all: bool = False,
):
    """Print all existing logs and stream new ones via kubectl -f.

    When *mirror* is True, container log-file paths (``/var/log/...``) are
    rewritten to local paths and the files are downloaded in the background.

    When *sync_all* is True, a bulk copy of all container log directories
    is performed when the stream ends (Ctrl+C or pod exit).
    """
    pod_name: str | None = None
    mirror_root: Path | None = None

    if mirror or sync_all:
        pod_name = resolve_pod_name(job_name, namespace)
        if pod_name:
            mirror_root = _mirror_dir(job_name, mirror_base)
            info(f"Mirroring container logs → {mirror_root}")
            if sync_all:
                info("Full log sync will run when streaming ends.")
        else:
            warn("Could not resolve pod name — log mirroring disabled.")
            mirror = False
            sync_all = False

    info(f"Streaming logs (existing + live) for '{job_name}'...")
    print(f"  {DIM}(Press Ctrl+C to stop){NC}")
    print()

    kubectl_cmd = [
        KUBECTL,
        "logs",
        f"job/{job_name}",
        "-n",
        namespace,
        "-f",
        "--tail=-1",
    ]

    def _run_stream(cmd: list[str]) -> None:
        with ThreadPoolExecutor(max_workers=4) as executor:
            rewrite_fn = None
            if mirror and mirror_root and pod_name:
                rewrite_fn = lambda line: _rewrite_line(
                    line,
                    mirror_root,
                    pod_name,
                    namespace,
                    executor,
                )

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                _emit_lines(proc, rewrite_fn=rewrite_fn)
                proc.wait()
                if proc.returncode != 0:
                    stderr = proc.stderr.read().decode("utf-8", errors="replace")
                    if stderr.strip():
                        warn(stderr.strip())
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                print()
                info("Stopped streaming.")
            finally:
                executor.shutdown(wait=True)
                if sync_all and pod_name and mirror_root:
                    _sync_all_logs(pod_name, namespace, mirror_root)

    try:
        _run_stream(kubectl_cmd)
    except Exception:
        warn("kubectl could not attach. Retrying in 5s...")
        import time

        time.sleep(5)
        _run_stream(kubectl_cmd)


def _gcloud_logging_read(log_filter: str):
    """Run gcloud logging read with the given filter, applying highlighting."""
    proc = subprocess.Popen(
        [
            GCLOUD,
            "logging",
            "read",
            log_filter,
            f"--project={GCP_PROJECT}",
            "--freshness=3650d",
            "--format=value(textPayload)",
            "--order=asc",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _emit_lines(proc)
    proc.wait()


def fetch_historical_logs(job_name: str, namespace: str):
    """Fetch historical logs from Cloud Logging.

    Called when the running flag is False, meaning the pod is gone.
    Goes straight to gcloud — no kubectl attempt needed.
    """
    info(f"Fetching historical logs from Cloud Logging for '{job_name}'...")
    gcs_path = f"gs://unity-pod-logs/{namespace}/{job_name}/"
    print(
        f"\n  {DIM}Cloud Logging only has INFO+ (terminal output).{NC}"
        f"\n  {DIM}Full DEBUG logs (if uploaded on shutdown): {CYAN}{gcs_path}{NC}"
        f"\n  {DIM}Download: gcloud storage cp --recursive {gcs_path} ./pod-logs/{NC}\n",
    )
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
            "  python stream_logs.py                        # latest staging job\n"
            "  python stream_logs.py --production            # latest production job\n"
            "  python stream_logs.py --job unity-2026-02-10-17-30-53-staging"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--job",
        default=None,
        help="Name of the GKE job. If omitted, auto-detects the latest job for your account.",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Target the production environment (default: staging)",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        default=False,
        help="Disable automatic log file mirroring (pass-through mode).",
    )
    parser.add_argument(
        "--mirror-dir",
        default=None,
        help=f"Local directory for mirrored log files (default: {MIRROR_BASE}).",
    )
    parser.add_argument(
        "--sync-all-logs",
        action="store_true",
        default=True,
        help="Bulk-copy all container log directories on stream exit (Ctrl+C). On by default.",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        default=False,
        help="Disable automatic log sync on stream exit.",
    )
    args = parser.parse_args()

    namespace = "production" if args.production else "staging"
    mirror = not args.no_mirror
    sync_all = args.sync_all_logs and not args.no_sync
    mirror_base = Path(args.mirror_dir).resolve() if args.mirror_dir else MIRROR_BASE
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
        stream_logs(
            job_name,
            namespace,
            mirror=mirror,
            mirror_base=mirror_base,
            sync_all=sync_all,
        )
    elif running is False:
        # AssistantJobs says not running → pod is gone, use gcloud.
        fetch_historical_logs(job_name, namespace)
    else:
        # No AssistantJobs record. Try streaming; if kubectl fails, fall
        # back to gcloud historical logs.
        warn("Unknown job status. Trying kubectl stream first...")
        print()
        stream_logs(
            job_name,
            namespace,
            mirror=mirror,
            mirror_base=mirror_base,
            sync_all=sync_all,
        )


if __name__ == "__main__":
    main()
