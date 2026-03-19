#!/usr/bin/env python3
"""
Upload pod log directories to GCS on shutdown.

Called from main.py's shutdown sequence (via subprocess) so it runs on all exit
paths — both SIGTERM from Kubernetes and self-initiated inactivity shutdown.
Compresses /var/log/{unity,unify,unillm,magnitude} into a tar.gz and uploads
to gs://unity-pod-logs/{namespace}/{job_name}/.

The bucket has a 7-day lifecycle policy — logs are auto-deleted after a week.

Usage:
    python3 scripts/upload_pod_logs.py
    python3 scripts/upload_pod_logs.py --dry-run

Environment:
    UNITY_CONVERSATION_JOB_NAME  Required. The K8s job name (e.g., unity-2026-02-28-12-00-09-staging).
    GCS_LOG_BUCKET               Optional. Override bucket name (default: unity-pod-logs).
"""

import os
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

BUCKET_NAME = os.environ.get("GCS_LOG_BUCKET", "unity-pod-logs")
LOG_DIRS = ["/var/log/unity", "/var/log/unify", "/var/log/unillm", "/var/log/magnitude"]
JOB_NAME = os.environ.get("UNITY_CONVERSATION_JOB_NAME", "")


def _derive_namespace(job_name: str) -> str:
    for env in ("staging", "preview", "production"):
        if job_name.endswith(f"-{env}"):
            return env
    return "unknown"


def get_gcs_prefix() -> str:
    namespace = _derive_namespace(JOB_NAME)
    return f"{namespace}/{JOB_NAME}" if JOB_NAME else f"{namespace}/unknown-job"


def compress_logs(log_dirs: list[str]) -> Path | None:
    """Compress log directories into a single tar.gz archive.

    Files that vanish between directory listing and read are silently
    skipped — this is expected during shutdown when unify/unillm trace
    files with ``_PENDING_`` suffixes may be renamed or deleted.
    """
    existing = [d for d in log_dirs if os.path.isdir(d) and os.listdir(d)]
    if not existing:
        return None

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    tmp = Path(tempfile.mktemp(suffix=f"_{ts}.tar.gz"))
    skipped = 0

    with tarfile.open(tmp, "w:gz") as tar:
        for log_dir in existing:
            arcname = os.path.basename(log_dir)
            for root, dirs, files in os.walk(log_dir):
                for fname in files:
                    full = os.path.join(root, fname)
                    arc = os.path.join(arcname, os.path.relpath(full, log_dir))
                    try:
                        tar.add(full, arcname=arc, recursive=False)
                    except (FileNotFoundError, OSError):
                        skipped += 1

    if skipped:
        print(f"[upload_pod_logs] Skipped {skipped} vanished file(s)")

    return tmp


def upload_to_gcs(local_path: Path, gcs_prefix: str) -> str:
    """Upload a file to GCS. Returns the gs:// URI."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob_name = f"{gcs_prefix}/{local_path.name}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    return f"gs://{BUCKET_NAME}/{blob_name}"


def main():
    dry_run = "--dry-run" in sys.argv

    if not JOB_NAME:
        print("[upload_pod_logs] UNITY_CONVERSATION_JOB_NAME not set, skipping upload")
        return

    prefix = get_gcs_prefix()
    print(f"[upload_pod_logs] Compressing logs from {LOG_DIRS}...")

    archive = compress_logs(LOG_DIRS)
    if archive is None:
        print("[upload_pod_logs] No log files found, nothing to upload")
        return

    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"[upload_pod_logs] Archive: {archive.name} ({size_mb:.1f} MB)")

    if dry_run:
        print(
            f"[upload_pod_logs] DRY RUN: would upload to gs://{BUCKET_NAME}/{prefix}/",
        )
        archive.unlink()
        return

    try:
        uri = upload_to_gcs(archive, prefix)
        print(f"[upload_pod_logs] Uploaded to {uri}")
    except Exception as e:
        print(f"[upload_pod_logs] Upload failed: {e}", file=sys.stderr)
    finally:
        archive.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
