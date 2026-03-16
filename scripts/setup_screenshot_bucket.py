#!/usr/bin/env python3
"""Create the GCS bucket for ephemeral images with a lifecycle policy.

Usage:
    python3 scripts/setup_screenshot_bucket.py
    python3 scripts/setup_screenshot_bucket.py --dry-run
    python3 scripts/setup_screenshot_bucket.py --bucket my-custom-bucket --ttl 72

Environment:
    UNITY_IMAGE_BUCKET  Override bucket name (default: unity-screenshots).
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Set up GCS image bucket")
    parser.add_argument(
        "--bucket",
        default=os.environ.get("UNITY_IMAGE_BUCKET", "unity-screenshots"),
        help="Bucket name (default: unity-screenshots)",
    )
    parser.add_argument(
        "--location",
        default="us-central1",
        help="Bucket location (default: us-central1)",
    )
    parser.add_argument(
        "--ttl",
        type=int,
        default=24,
        help="Object TTL in hours (default: 24)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from google.cloud import storage

    client = storage.Client()
    bucket_name = args.bucket

    print(f"[setup] Bucket: {bucket_name}")
    print(f"[setup] Location: {args.location}")
    print(f"[setup] TTL: {args.ttl} hours ({args.ttl // 24} days)")

    if args.dry_run:
        print("[setup] Dry run — no changes made")
        return

    bucket = client.bucket(bucket_name)
    if bucket.exists():
        print(f"[setup] Bucket {bucket_name} already exists")
    else:
        bucket.storage_class = "STANDARD"
        bucket = client.create_bucket(bucket, location=args.location)
        print(f"[setup] Created bucket {bucket_name} in {args.location}")

    ttl_days = max(1, args.ttl // 24)
    bucket.add_lifecycle_delete_rule(age=ttl_days)
    bucket.patch()
    print(f"[setup] Lifecycle rule: delete objects after {ttl_days} day(s)")

    print("[setup] Done")


if __name__ == "__main__":
    main()
