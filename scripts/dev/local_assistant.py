#!/usr/bin/env python3
"""
Create or fetch a local assistant and print the env vars needed to run
unity locally.

A "local assistant" is stored in production Orchestra (is_local=True) and
uses production adapters/communication, but runs unity on your machine
instead of on GKE.

Usage:
    # Create a new local assistant (or fetch existing by name):
    python scripts/dev/local_assistant.py --api-key YOUR_KEY --name "Dev Assistant"

    # Fetch an existing local assistant by ID:
    python scripts/dev/local_assistant.py --api-key YOUR_KEY --id 42

    # Source directly into your shell:
    source <(python scripts/dev/local_assistant.py --api-key YOUR_KEY --name "Dev")

    # Write to a .env file:
    python scripts/dev/local_assistant.py --api-key YOUR_KEY --name "Dev" > .env.local
"""

import argparse
import json
import sys

import requests

ORCHESTRA_URL = "https://api.unify.ai/v0"


def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"}


def _find_existing(api_key: str, first_name: str, surname: str):
    """Find an existing assistant by name."""
    resp = requests.get(f"{ORCHESTRA_URL}/assistant", headers=_headers(api_key))
    resp.raise_for_status()
    for a in resp.json().get("info", []):
        if a.get("first_name") == first_name and a.get("surname") == surname:
            return a
    return None


def _fetch_by_id(api_key: str, agent_id: int):
    """Fetch an assistant by ID."""
    resp = requests.get(
        f"{ORCHESTRA_URL}/assistant/{agent_id}",
        headers=_headers(api_key),
    )
    resp.raise_for_status()
    return resp.json().get("info")


def _create(api_key: str, first_name: str, surname: str):
    """Create a local assistant with no infra provisioned."""
    payload = {
        "first_name": first_name,
        "surname": surname,
        "is_local": True,
        "create_infra": False,
    }
    resp = requests.post(
        f"{ORCHESTRA_URL}/assistant",
        json=payload,
        headers=_headers(api_key),
    )
    resp.raise_for_status()
    return resp.json().get("info")


def _get_user_info(api_key: str):
    """Fetch the authenticated user's profile."""
    resp = requests.get(f"{ORCHESTRA_URL}/user", headers=_headers(api_key))
    resp.raise_for_status()
    return resp.json().get("info", resp.json())


def _print_env(assistant: dict, user: dict, api_key: str):
    """Print env vars matching SessionDetails.export_to_env() format."""
    lines = [
        f'export ASSISTANT_ID="{assistant.get("agent_id", "")}"',
        f'export ASSISTANT_FIRST_NAME="{assistant.get("first_name", "")}"',
        f'export ASSISTANT_SURNAME="{assistant.get("surname", "")}"',
        f'export ASSISTANT_NAME="{assistant.get("first_name", "")} {assistant.get("surname", "")}"',
        f'export ASSISTANT_AGE="{assistant.get("age", "") or ""}"',
        f'export ASSISTANT_NATIONALITY="{assistant.get("nationality", "") or ""}"',
        f'export ASSISTANT_TIMEZONE="{assistant.get("timezone", "") or ""}"',
        f'export ASSISTANT_ABOUT="{assistant.get("about", "") or ""}"',
        f'export ASSISTANT_NUMBER="{assistant.get("phone", "") or ""}"',
        f'export ASSISTANT_EMAIL="{assistant.get("email", "") or ""}"',
        f'export ASSISTANT_DESKTOP_MODE="{assistant.get("desktop_mode", "ubuntu") or "ubuntu"}"',
        f'export ASSISTANT_DESKTOP_URL="{assistant.get("desktop_url", "") or ""}"',
        f'export ASSISTANT_USER_DESKTOP_MODE="{assistant.get("user_desktop_mode", "") or ""}"',
        f'export ASSISTANT_USER_DESKTOP_FILESYS_SYNC="{assistant.get("user_desktop_filesys_sync", False)}"',
        f'export ASSISTANT_USER_DESKTOP_URL="{assistant.get("user_desktop_url", "") or ""}"',
        f'export USER_ID="{assistant.get("user_id", "")}"',
        f'export USER_FIRST_NAME="{user.get("first_name", "") or ""}"',
        f'export USER_SURNAME="{user.get("last_name", "") or user.get("surname", "") or ""}"',
        f'export USER_NAME="{user.get("first_name", "")} {user.get("last_name", "") or user.get("surname", "")}"',
        f'export USER_NUMBER="{assistant.get("user_phone", "") or ""}"',
        f'export USER_EMAIL="{user.get("email", "") or ""}"',
        f'export UNIFY_KEY="{api_key}"',
    ]
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Create or fetch a local assistant and print env vars.",
    )
    parser.add_argument("--api-key", required=True, help="Unify API key")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", help="Assistant name (format: 'FirstName' or 'FirstName Surname')")
    group.add_argument("--id", type=int, help="Existing assistant agent_id")
    args = parser.parse_args()

    api_key = args.api_key

    if args.id is not None:
        assistant = _fetch_by_id(api_key, args.id)
        if not assistant:
            print(f"Error: assistant {args.id} not found", file=sys.stderr)
            sys.exit(1)
    else:
        parts = args.name.strip().split(None, 1)
        first_name = parts[0]
        surname = parts[1] if len(parts) > 1 else ""

        assistant = _find_existing(api_key, first_name, surname)
        if assistant:
            print(
                f"# Found existing assistant: {first_name} {surname} "
                f"(id={assistant['agent_id']})",
                file=sys.stderr,
            )
        else:
            print(
                f"# Creating local assistant: {first_name} {surname}",
                file=sys.stderr,
            )
            assistant = _create(api_key, first_name, surname)
            print(
                f"# Created assistant id={assistant['agent_id']}",
                file=sys.stderr,
            )

    user = _get_user_info(api_key)
    _print_env(assistant, user, api_key)


if __name__ == "__main__":
    main()
