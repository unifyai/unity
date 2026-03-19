#!/usr/bin/env python3
"""
Download all session data for a Unity organization or assistant.

Discovers assistants via the admin API, then fetches guidance, functions,
file records, sessions, cloud logs, and pod log archives.

Usage:
    cd unity/
    uv run python scripts/download_session_data.py --org-id 2
    uv run python scripts/download_session_data.py --org-id 6
    uv run python scripts/download_session_data.py --assistant-id 84
    uv run python scripts/download_session_data.py --org-id 2 --name colliers
    uv run python scripts/download_session_data.py --org-id 2 --output-dir /tmp/data
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import requests

# ---------------------------------------------------------------------------
# GCP constants
# ---------------------------------------------------------------------------

GCP_PROJECT = "responsive-city-458413-a2"
GKE_CLUSTER = "unity"
GCS_BUCKET = "gs://unity-pod-logs/unknown"
ORCHESTRA_URL = "https://api.unify.ai/v0"
GCLOUD = shutil.which("gcloud") or "gcloud"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def info(msg: str) -> None:
    print(f"\033[0;36m[INFO]\033[0m {msg}")


def ok(msg: str) -> None:
    print(f"\033[0;32m[ OK ]\033[0m {msg}")


def warn(msg: str) -> None:
    print(f"\033[1;33m[WARN]\033[0m {msg}")


def err(msg: str) -> None:
    print(f"\033[0;31m[ ERR]\033[0m {msg}", file=sys.stderr)


def _env(name: str) -> str:
    val = os.environ.get(name, "").strip().strip('"').strip("'")
    if not val:
        err(f"{name} not set. Add it to .env or export it.")
        sys.exit(1)
    return val


def _slugify(name: str) -> str:
    """Convert a name to a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "unknown"


def _fetch_org_name(org_id: int, admin_key: str) -> str:
    """Fetch organization name from admin API."""
    try:
        resp = requests.get(
            f"{ORCHESTRA_URL}/admin/organizations",
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        for org in resp.json().get("organizations", []):
            if org.get("id") == org_id:
                return org.get("name", "")
    except Exception:
        pass
    return ""


def _resolve_output_dir(args, admin_key: str) -> tuple[Path, str]:
    """Determine the output directory and display name.

    Returns (output_path, display_name).
    """
    base = Path(__file__).resolve().parent.parent

    if args.output_dir:
        out = Path(args.output_dir).resolve()
        name = args.name or out.name.removesuffix("_data") or "data"
        return out, name

    if args.name:
        return base / f"{args.name}_data", args.name

    # Auto-derive name from org or assistant
    if args.org_id:
        org_name = _fetch_org_name(args.org_id, admin_key)
        slug = _slugify(org_name) if org_name else f"org_{args.org_id}"
    else:
        all_assts = _fetch_all_assistants(admin_key)
        match = [
            a for a in all_assts if str(a.get("agent_id")) == str(args.assistant_id)
        ]
        if match:
            name = f"{match[0].get('first_name', '')} {match[0].get('surname', '')}".strip()
            slug = (
                f"{_slugify(name)}_{args.assistant_id}"
                if name
                else f"assistant_{args.assistant_id}"
            )
        else:
            slug = f"assistant_{args.assistant_id}"

    candidate = base / f"{slug}_data"

    # Collision check: if dir exists but belongs to a different entity, bump the number
    if candidate.exists():
        org_json = candidate / "metadata" / "org.json"
        if org_json.exists():
            try:
                existing = json.loads(org_json.read_text())
                existing_org = existing.get("org_id")
                target_org = args.org_id
                if not target_org and match:
                    target_org = match[0].get("organization_id")
                if existing_org == target_org:
                    return candidate, slug  # same entity, reuse
            except Exception:
                pass
            # Different entity — find a free name
            for i in range(2, 100):
                alt = base / f"{slug}_{i}_data"
                if not alt.exists():
                    return alt, f"{slug}_{i}"
        # No org.json yet — safe to reuse (first run left an empty dir, or same entity)
    return candidate, slug


# ---------------------------------------------------------------------------
# Step 1: Discover org assistants via admin API
# ---------------------------------------------------------------------------


def _fetch_all_assistants(admin_key: str) -> list[dict]:
    resp = requests.get(
        f"{ORCHESTRA_URL}/admin/assistant",
        headers={"Authorization": f"Bearer {admin_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("info", [])


def discover_assistants_by_org(org_id: int, admin_key: str) -> list[dict]:
    info(f"Discovering assistants for org_id={org_id}...")
    all_assistants = _fetch_all_assistants(admin_key)
    org_assistants = [a for a in all_assistants if a.get("organization_id") == org_id]
    ok(f"Found {len(org_assistants)} assistant(s) in org {org_id}")
    for a in org_assistants:
        name = f"{a.get('first_name', '')} {a.get('surname', '')}".strip()
        owner = f"{a.get('user_first_name', '')} {a.get('user_last_name', '')}".strip()
        print(
            f"    assistant_id={a['agent_id']}  {name}  (owner: {owner}, {a.get('user_email', '')})",
        )
    return org_assistants


def discover_assistant_by_id(assistant_id: str, admin_key: str) -> list[dict]:
    info(f"Fetching assistant_id={assistant_id}...")
    all_assistants = _fetch_all_assistants(admin_key)
    matches = [a for a in all_assistants if str(a.get("agent_id")) == str(assistant_id)]
    if not matches:
        return []
    a = matches[0]
    name = f"{a.get('first_name', '')} {a.get('surname', '')}".strip()
    owner = f"{a.get('user_first_name', '')} {a.get('user_last_name', '')}".strip()
    ok(
        f"Found assistant {a['agent_id']}  {name}  (owner: {owner}, {a.get('user_email', '')})",
    )
    return matches


def save_org_metadata(
    org_id: int,
    org_name: str,
    assistants: list[dict],
    out: Path,
) -> None:
    meta_dir = out / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    org_info = {
        "org_id": org_id,
        "org_name": org_name,
        "assistants": [],
    }
    for a in assistants:
        org_info["assistants"].append(
            {
                "assistant_id": a["agent_id"],
                "name": f"{a.get('first_name', '')} {a.get('surname', '')}".strip(),
                "owner_name": f"{a.get('user_first_name', '')} {a.get('user_last_name', '')}".strip(),
                "owner_email": a.get("user_email", ""),
                "user_id": a.get("user_id", ""),
            },
        )

    (meta_dir / "org.json").write_text(json.dumps(org_info, indent=2))
    ok(f"Saved org metadata to {meta_dir / 'org.json'}")


# ---------------------------------------------------------------------------
# Step 2: Download assistant data (guidance, functions, file records)
# ---------------------------------------------------------------------------


def _fetch_context_logs(api_key: str, context: str, limit: int = 500) -> list[dict]:
    resp = requests.get(
        f"{ORCHESTRA_URL}/logs",
        params={"project_name": "Assistants", "context": context, "limit": limit},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    return resp.json().get("logs", [])


def _inject_ts(log_entry: dict) -> dict:
    """Inject the log-level timestamp into the entries dict."""
    entry = log_entry.get("entries", {})
    entry["_created_at"] = log_entry.get("ts", "")
    return entry


GCS_ATTACHMENTS_BUCKET = "gs://assistant-message-attachments-production"


def download_assistant_files(aid: str, asst_dir: Path) -> None:
    """Download attachment files from GCS for this assistant."""
    files_dir = asst_dir / "files"
    if files_dir.exists() and any(files_dir.iterdir()):
        existing = sum(1 for f in files_dir.iterdir() if f.is_file())
        ok(f"  Files: {existing} already on disk (skipping)")
        return

    gcs_prefix = f"{GCS_ATTACHMENTS_BUCKET}/{aid}/"
    result = subprocess.run(
        [GCLOUD, "storage", "ls", gcs_prefix, f"--project={GCP_PROJECT}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0 or not result.stdout.strip():
        warn(f"  No files in GCS for assistant {aid}")
        return

    file_count = len(result.stdout.strip().splitlines())
    info(f"  Downloading {file_count} files from GCS...")
    files_dir.mkdir(parents=True, exist_ok=True)

    dl = subprocess.run(
        [
            GCLOUD,
            "storage",
            "cp",
            f"{gcs_prefix}*",
            str(files_dir),
            f"--project={GCP_PROJECT}",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if dl.returncode != 0:
        warn(f"  File download failed: {dl.stderr[:200]}")
        return

    downloaded = sum(1 for f in files_dir.iterdir() if f.is_file())
    ok(f"  Files: {downloaded} downloaded")


def download_assistant_data(assistant: dict, out: Path) -> None:
    aid = assistant["agent_id"]
    uid = assistant["user_id"]
    api_key = assistant.get("api_key", "")
    name = f"{assistant.get('first_name', '')} {assistant.get('surname', '')}".strip()

    info(f"Downloading data for assistant {aid} ({name})...")
    asst_dir = out / "assistants" / str(aid)
    asst_dir.mkdir(parents=True, exist_ok=True)

    # Save assistant info
    asst_info = {
        "assistant_id": aid,
        "name": name,
        "owner_name": f"{assistant.get('user_first_name', '')} {assistant.get('user_last_name', '')}".strip(),
        "owner_email": assistant.get("user_email", ""),
        "user_id": uid,
        "about": assistant.get("about", ""),
        "phone": assistant.get("phone", ""),
        "email": assistant.get("email", ""),
        "created_at": assistant.get("created_at", ""),
    }
    (asst_dir / "info.json").write_text(json.dumps(asst_info, indent=2))

    if not api_key:
        warn(f"  No api_key for assistant {aid} -- skipping guidance/functions/files")
        return

    # Guidance (preserve timestamps)
    logs = _fetch_context_logs(api_key, f"{uid}/{aid}/Guidance")
    entries = [_inject_ts(lg) for lg in logs]
    (asst_dir / "guidance.json").write_text(json.dumps(entries, indent=2, default=str))
    ok(f"  Guidance: {len(entries)} entries")

    # Functions (preserve timestamps)
    logs = _fetch_context_logs(api_key, f"{uid}/{aid}/Functions/Compositional")
    entries = [_inject_ts(lg) for lg in logs]
    (asst_dir / "functions.json").write_text(json.dumps(entries, indent=2, default=str))
    ok(f"  Functions: {len(entries)} entries")

    # File records (preserve timestamps)
    logs = _fetch_context_logs(api_key, f"{uid}/{aid}/FileRecords/Local")
    entries = [_inject_ts(lg) for lg in logs]
    (asst_dir / "file_records.json").write_text(
        json.dumps(entries, indent=2, default=str),
    )
    ok(f"  File records: {len(entries)} entries")

    # Download actual files from GCS
    download_assistant_files(aid, asst_dir)


# ---------------------------------------------------------------------------
# Step 3: Fetch sessions from AssistantJobs
# ---------------------------------------------------------------------------


def fetch_sessions(assistant_ids: list[str], shared_key: str) -> list[dict]:
    info(f"Fetching sessions for {len(assistant_ids)} assistant(s)...")
    all_sessions: list[dict] = []

    for aid in assistant_ids:
        resp = requests.get(
            f"{ORCHESTRA_URL}/logs",
            params={
                "project_name": "AssistantJobs",
                "context": "startup_events",
                "filter_expr": f"assistant_id == '{aid}'",
                "limit": 500,
            },
            headers={"Authorization": f"Bearer {shared_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        logs = resp.json().get("logs", [])
        all_sessions.extend(logs)
        ok(f"  assistant {aid}: {len(logs)} sessions")

    ok(f"Total sessions: {len(all_sessions)}")
    return all_sessions


# ---------------------------------------------------------------------------
# Step 4: Download cloud logging
# ---------------------------------------------------------------------------


def download_cloud_logging(job_name: str, session_dir: Path) -> bool:
    dst = session_dir / "cloud_logging.txt"
    if dst.exists() and dst.stat().st_size > 100:
        return True

    session_dir.mkdir(parents=True, exist_ok=True)
    log_filter = (
        f'resource.type="k8s_container"'
        f' AND resource.labels.cluster_name="{GKE_CLUSTER}"'
        f' AND labels."k8s-pod/batch_kubernetes_io/job-name"="{job_name}"'
    )
    try:
        result = subprocess.run(
            [
                GCLOUD,
                "logging",
                "read",
                log_filter,
                f"--project={GCP_PROJECT}",
                "--freshness=90d",
                "--format=value(textPayload)",
                "--order=asc",
                "--limit=5000",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            warn(f"  gcloud failed for {job_name}: {result.stderr[:200]}")
            return False
        text = result.stdout.strip()
        if not text:
            warn(f"  No cloud logging for {job_name}")
            dst.write_text("")
            return False
        dst.write_text(text, encoding="utf-8")
        ok(f"  Cloud Logging: {job_name} ({text.count(chr(10)) + 1} lines)")
        return True
    except subprocess.TimeoutExpired:
        warn(f"  Timeout for {job_name}")
        return False
    except Exception as e:
        warn(f"  Error for {job_name}: {e}")
        return False


def download_all_cloud_logging(job_names: list[str], out: Path) -> dict[str, bool]:
    info(f"Downloading Cloud Logging for {len(job_names)} jobs...")
    results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}
        for jn in job_names:
            sd = out / "sessions" / jn
            futures[pool.submit(download_cloud_logging, jn, sd)] = jn
        for fut in as_completed(futures):
            jn = futures[fut]
            try:
                results[jn] = fut.result()
            except Exception as e:
                warn(f"  Exception for {jn}: {e}")
                results[jn] = False
    succeeded = sum(1 for v in results.values() if v)
    ok(f"Cloud Logging: {succeeded}/{len(job_names)} downloaded")
    return results


# ---------------------------------------------------------------------------
# Step 5: Download GCS pod logs
# ---------------------------------------------------------------------------


def list_gcs_pod_logs(job_names: list[str]) -> dict[str, str | None]:
    info("Checking GCS for pod log archives...")
    result = subprocess.run(
        [GCLOUD, "storage", "ls", f"{GCS_BUCKET}/", f"--project={GCP_PROJECT}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        warn(f"Failed to list GCS bucket: {result.stderr[:200]}")
        return {jn: None for jn in job_names}

    available = set(result.stdout.strip().splitlines())
    mapping: dict[str, str | None] = {}
    for jn in job_names:
        prefix = f"{GCS_BUCKET}/{jn}/"
        mapping[jn] = prefix if prefix in available else None

    found = sum(1 for v in mapping.values() if v)
    ok(f"GCS pod logs available for {found}/{len(job_names)} jobs")
    return mapping


def download_gcs_pod_log(job_name: str, gcs_prefix: str, session_dir: Path) -> bool:
    pod_dir = session_dir / "pod_logs"
    if pod_dir.exists() and any(pod_dir.iterdir()):
        return True

    session_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [GCLOUD, "storage", "ls", gcs_prefix, f"--project={GCP_PROJECT}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return False

    tar_files = [
        l.strip()
        for l in result.stdout.strip().splitlines()
        if l.strip().endswith(".tar.gz")
    ]
    if not tar_files:
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        for tar_url in tar_files:
            tar_name = tar_url.rsplit("/", 1)[-1]
            local_tar = Path(tmpdir) / tar_name
            dl = subprocess.run(
                [
                    GCLOUD,
                    "storage",
                    "cp",
                    tar_url,
                    str(local_tar),
                    f"--project={GCP_PROJECT}",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if dl.returncode != 0:
                continue
            pod_dir.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(local_tar, "r:gz") as tf:
                    tf.extractall(path=pod_dir, filter="data")
                ok(f"  Pod logs: {job_name}")
            except Exception as e:
                warn(f"  Extract failed for {tar_name}: {e}")
                return False
    return True


def download_all_gcs_pod_logs(
    gcs_mapping: dict[str, str | None],
    out: Path,
) -> dict[str, bool]:
    info("Downloading GCS pod log archives...")
    results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}
        for jn, prefix in gcs_mapping.items():
            if prefix is None:
                results[jn] = False
                continue
            sd = out / "sessions" / jn
            futures[pool.submit(download_gcs_pod_log, jn, prefix, sd)] = jn
        for fut in as_completed(futures):
            jn = futures[fut]
            try:
                results[jn] = fut.result()
            except Exception as e:
                warn(f"  Exception for {jn}: {e}")
                results[jn] = False
    succeeded = sum(1 for v in results.values() if v)
    total = sum(1 for v in gcs_mapping.values() if v is not None)
    ok(f"GCS pod logs: {succeeded}/{total} downloaded")
    return results


# ---------------------------------------------------------------------------
# Step 6: Generate index
# ---------------------------------------------------------------------------


def generate_index(
    sessions: list[dict],
    cl_results: dict[str, bool],
    gcs_results: dict[str, bool],
    out: Path,
) -> None:
    info("Generating index...")
    index_entries = []
    for session in sessions:
        e = session.get("entries", {})
        jn = e.get("job_name", "")
        sd = out / "sessions" / jn
        cl_file = sd / "cloud_logging.txt"
        pod_dir = sd / "pod_logs"

        cl_lines = 0
        if cl_file.exists():
            cl_lines = cl_file.read_text(encoding="utf-8", errors="replace").count("\n")

        pod_files: list[str] = []
        pod_size = 0
        if pod_dir.exists():
            for f in pod_dir.rglob("*"):
                if f.is_file():
                    pod_files.append(str(f.relative_to(pod_dir)))
                    pod_size += f.stat().st_size

        index_entries.append(
            {
                "job_name": jn,
                "assistant_id": e.get("assistant_id", ""),
                "assistant_name": e.get("assistant_name", ""),
                "timestamp": e.get("timestamp", ""),
                "medium": e.get("medium", ""),
                "user_name": e.get("user_name", ""),
                "user_email": e.get("user_email", ""),
                "log_id": session.get("id"),
                "cloud_logging_downloaded": cl_results.get(jn, False),
                "cloud_logging_lines": cl_lines,
                "pod_logs_downloaded": gcs_results.get(jn, False),
                "pod_log_files": pod_files,
                "pod_log_total_bytes": pod_size,
            },
        )

    index_entries.sort(key=lambda x: x.get("timestamp", ""))
    (out / "index.json").write_text(json.dumps(index_entries, indent=2, default=str))
    ok(f"Index: {len(index_entries)} sessions")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Download Unity session data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--org-id",
        type=int,
        help="Organization ID (downloads all assistants in org)",
    )
    group.add_argument("--assistant-id", type=str, help="Single assistant ID")
    parser.add_argument(
        "--name",
        default=None,
        help="Short name for output dir (auto-derived if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Explicit output directory path",
    )
    args = parser.parse_args()

    admin_key = _env("ORCHESTRA_ADMIN_KEY")
    shared_key = _env("SHARED_UNIFY_KEY")

    out, display_name = _resolve_output_dir(args, admin_key)
    mode = (
        f"org_id={args.org_id}" if args.org_id else f"assistant_id={args.assistant_id}"
    )

    print()
    print("=" * 60)
    print(f"  Download: {display_name} ({mode})")
    print(f"  Output:   {out}")
    print("=" * 60)
    print()

    # Load existing index for incremental mode
    existing_jobs: set[str] = set()
    idx_path = out / "index.json"
    if idx_path.exists():
        try:
            existing_jobs = {
                e["job_name"]
                for e in json.loads(idx_path.read_text())
                if e.get("job_name")
            }
            info(f"Found {len(existing_jobs)} existing sessions on disk")
        except Exception:
            pass

    # Step 1: discover assistants
    if args.org_id:
        assistants = discover_assistants_by_org(args.org_id, admin_key)
    else:
        assistants = discover_assistant_by_id(args.assistant_id, admin_key)

    if not assistants:
        err(f"No assistants found for {mode}")
        sys.exit(1)

    org_id = args.org_id or assistants[0].get("organization_id")
    save_org_metadata(org_id, display_name, assistants, out)
    print()

    # Step 2: download assistant data (always refresh)
    for a in assistants:
        download_assistant_data(a, out)
    print()

    # Step 3: fetch sessions
    assistant_ids = [a["agent_id"] for a in assistants]
    sessions = fetch_sessions(assistant_ids, shared_key)

    meta_dir = out / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "sessions.json").write_text(json.dumps(sessions, indent=2, default=str))

    job_names = [
        s.get("entries", {}).get("job_name", "")
        for s in sessions
        if s.get("entries", {}).get("job_name")
    ]
    new_jobs = [jn for jn in job_names if jn not in existing_jobs]
    skipped = len(job_names) - len(new_jobs)

    if new_jobs:
        info(f"{len(new_jobs)} new session(s), {skipped} already on disk")
    else:
        info(f"All {len(job_names)} sessions already downloaded -- checking for gaps")
        new_jobs = job_names
    print()

    # Step 4: cloud logging
    cl_results = download_all_cloud_logging(new_jobs, out)
    print()

    # Step 5: GCS pod logs
    gcs_mapping = list_gcs_pod_logs(new_jobs)
    gcs_results = download_all_gcs_pod_logs(gcs_mapping, out)
    print()

    # Fill in results for existing sessions
    for jn in job_names:
        if jn not in cl_results:
            cl_file = out / "sessions" / jn / "cloud_logging.txt"
            cl_results[jn] = cl_file.exists() and cl_file.stat().st_size > 100
        if jn not in gcs_results:
            pod_dir = out / "sessions" / jn / "pod_logs"
            gcs_results[jn] = (
                pod_dir.exists() and any(pod_dir.iterdir())
                if pod_dir.exists()
                else False
            )

    # Step 6: index
    generate_index(sessions, cl_results, gcs_results, out)

    # Summary
    print()
    print("=" * 60)
    total_size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    total_files = sum(1 for f in out.rglob("*") if f.is_file())
    print(f"  Done! {display_name} ({mode})")
    print(f"  Location:   {out}")
    print(f"  Assistants: {len(assistants)}")
    print(f"  Sessions:   {len(job_names)}")
    print(f"  Files:      {total_files}")
    print(f"  Size:       {total_size / 1024 / 1024:.1f} MB")
    if new_jobs and new_jobs != job_names:
        print(f"  New:        {len(new_jobs)}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
