#!/usr/bin/env python3
"""Sync the SHARED_UNIFY_KEY across Orchestra environments.

Ensures the production SHARED_UNIFY_KEY is also registered in the staging
Orchestra database so the same key works in both environments.

Prerequisites:
  - SHARED_UNIFY_KEY and ORCHESTRA_ADMIN_KEY must be set (in .env or env).
  - The Orchestra staging deployment must include the `custom_key` parameter
    on the admin POST /api_key endpoint.

Usage:
  .venv/bin/python scripts/dev/sync_shared_key.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

import requests

PROD_URL = "https://api.unify.ai/v0"
STAGING_URL = "https://api.staging.internal.saas.unify.ai/v0"


def _bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def main():
    shared_key = os.environ.get("SHARED_UNIFY_KEY", "")
    admin_key = os.environ.get("ORCHESTRA_ADMIN_KEY", "")

    if not shared_key:
        print("ERROR: SHARED_UNIFY_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    if not admin_key:
        print("ERROR: ORCHESTRA_ADMIN_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    # 1. Identify the key owner from production.
    print("[1/4] Identifying key owner from production...")
    resp = requests.get(f"{PROD_URL}/user/basic-info", headers=_bearer(shared_key))
    if resp.status_code != 200:
        print(
            f"ERROR: Could not authenticate SHARED_UNIFY_KEY against production ({resp.status_code}).",
            file=sys.stderr,
        )
        print(f"       {resp.text}", file=sys.stderr)
        sys.exit(1)

    user_info = resp.json()
    email = user_info["email"]
    print(f"       Owner: {user_info['first']} {user_info['last']} ({email})")

    # 2. Check if the key already works on staging.
    print("[2/4] Checking if key is already valid on staging...")
    resp = requests.get(f"{STAGING_URL}/user/basic-info", headers=_bearer(shared_key))
    if resp.status_code == 200:
        print("       Key is already valid on staging. Nothing to do.")
        return

    # 3. Look up the user on staging via admin API.
    print("[3/4] Looking up user on staging...")
    resp = requests.get(
        f"{STAGING_URL}/admin/user/by-email",
        params={"email": email},
        headers=_bearer(admin_key),
    )
    if resp.status_code != 200 or resp.json() is None:
        print(
            f"ERROR: Could not find user {email} on staging ({resp.status_code}).",
            file=sys.stderr,
        )
        print(f"       {resp.text}", file=sys.stderr)
        print("       The user may need to sign in to staging first.", file=sys.stderr)
        sys.exit(1)

    staging_user = resp.json()
    staging_user_id = staging_user["id"]
    staging_api_key = staging_user.get("api_key", "")
    print(f"       Found staging user_id: {staging_user_id}")

    if staging_api_key == shared_key:
        print("       Staging key already matches production. Nothing to do.")
        return

    # 4. Replace the staging key with the production key.
    #    The by-email lookup auto-creates a key, so the user always has one
    #    at this point. Use reset with custom_key to atomically swap it.
    print("[4/4] Replacing staging key with production key...")
    resp = requests.post(
        f"{STAGING_URL}/admin/api_key/reset",
        params={
            "user_id": staging_user_id,
            "custom_key": shared_key,
        },
        headers=_bearer(admin_key),
    )
    if resp.status_code == 400 and "already in use" in resp.text:
        print(
            "       Key value already registered on staging (under a different user).",
        )
        print(f"       Response: {resp.text}", file=sys.stderr)
        sys.exit(1)
    elif resp.status_code not in (200, 201):
        print(
            f"ERROR: Failed to reset staging key ({resp.status_code}).",
            file=sys.stderr,
        )
        print(f"       {resp.text}", file=sys.stderr)
        sys.exit(1)

    print("       Done! SHARED_UNIFY_KEY is now valid on both production and staging.")


if __name__ == "__main__":
    main()
