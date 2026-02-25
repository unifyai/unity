#!/usr/bin/env python3
"""
Create or fetch a local assistant and print the env vars needed to run
unity locally.

A "local assistant" is stored in production Orchestra (is_local=True) and
uses production adapters/communication, but runs unity on your machine
instead of on GKE.

Usage:
    # Create a new local assistant (or fetch existing by name):
    python scripts/dev/local_assistant.py --name "Dev Assistant"

    # Fetch an existing local assistant by ID:
    python scripts/dev/local_assistant.py --id 42

    # Target staging:
    python scripts/dev/local_assistant.py --name "Dev" --staging

    # Source directly into your shell:
    source <(python scripts/dev/local_assistant.py --name "Dev")

    # Write to a .env file:
    python scripts/dev/local_assistant.py --name "Dev" > .env.local

Requires UNIFY_KEY in the environment (loaded from .env).
"""

from dotenv import load_dotenv
import argparse
import os
import sys

import requests

load_dotenv()

ORCHESTRA_URLS = {
    "prod": "https://api.unify.ai/v0",
    "staging": "https://orchestra-staging-lz5fmz6i7q-ew.a.run.app/v0",
}


def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"}


def _find_existing(orchestra_url: str, api_key: str, first_name: str, surname: str):
    """Find an existing assistant by name."""
    resp = requests.get(f"{orchestra_url}/assistant", headers=_headers(api_key))
    resp.raise_for_status()
    for a in resp.json().get("info", []):
        if a.get("first_name") == first_name and a.get("surname") == surname:
            return a
    return None


def _fetch_by_id(orchestra_url: str, api_key: str, agent_id: int):
    """Fetch an assistant by ID."""
    resp = requests.get(
        f"{orchestra_url}/assistant/{agent_id}",
        headers=_headers(api_key),
    )
    resp.raise_for_status()
    return resp.json().get("info")


def _create(
    orchestra_url: str,
    api_key: str,
    first_name: str,
    surname: str,
    age: int = 25,
    nationality: str = "US",
    about: str = "Local Assistant",
    voice_id: str = "ThT5KcBeYPX3keUQqHPh",
    voice_provider: str = "elevenlabs",
    voice_mode: str = "tts",
):
    """Create a local assistant."""
    payload = {
        "first_name": first_name,
        "surname": surname,
        "age": age,
        "nationality": nationality,
        "about": about,
        "voice_id": voice_id,
        "voice_provider": voice_provider,
        "voice_mode": voice_mode,
        "is_local": True,
    }
    resp = requests.post(
        f"{orchestra_url}/assistant",
        json=payload,
        headers=_headers(api_key),
    )
    resp.raise_for_status()
    return resp.json().get("info")


def _get_user_info(orchestra_url: str, api_key: str):
    """Fetch the authenticated user's profile."""
    resp = requests.get(f"{orchestra_url}/user", headers=_headers(api_key))
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", help="Assistant name (format: 'FirstName' or 'FirstName Surname')")
    group.add_argument("--id", type=int, help="Existing assistant agent_id")
    parser.add_argument("--staging", action="store_true", help="Target the staging environment (default: prod)")
    parser.add_argument("--age", type=int, default=25, help="Age of the assistant (default: 25)")
    parser.add_argument("--nationality", default="US", help="Nationality (default: US)")
    parser.add_argument("--about", default="Local Assistant", help="Description (default: Local Assistant)")
    parser.add_argument("--voice-id", default="ThT5KcBeYPX3keUQqHPh", help="Voice ID (default: ThT5KcBeYPX3keUQqHPh)")
    parser.add_argument("--voice-provider", default="elevenlabs", help="Voice provider (default: elevenlabs)")
    parser.add_argument("--voice-mode", choices=["tts", "sts"], default="tts", help="Voice mode (default: tts)")
    args = parser.parse_args()

    env = "staging" if args.staging else "prod"
    orchestra_url = ORCHESTRA_URLS[env]
    api_key = os.getenv("UNIFY_KEY")
    if not api_key:
        print("Error: UNIFY_KEY env var required", file=sys.stderr)
        sys.exit(1)

    print(f"# Environment: {env} ({orchestra_url})", file=sys.stderr)

    if args.id is not None:
        assistant = _fetch_by_id(orchestra_url, api_key, args.id)
        if not assistant:
            print(f"Error: assistant {args.id} not found", file=sys.stderr)
            sys.exit(1)
    else:
        parts = args.name.strip().split(None, 1)
        first_name = parts[0]
        surname = parts[1] if len(parts) > 1 else ""

        assistant = _find_existing(orchestra_url, api_key, first_name, surname)
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
            assistant = _create(
                orchestra_url,
                api_key,
                first_name,
                surname,
                age=args.age,
                nationality=args.nationality,
                about=args.about,
                voice_id=args.voice_id,
                voice_provider=args.voice_provider,
                voice_mode=args.voice_mode,
            )
            print(
                f"# Created assistant id={assistant['agent_id']}",
                file=sys.stderr,
            )

    user = _get_user_info(orchestra_url, api_key)
    _print_env(assistant, user, api_key)


if __name__ == "__main__":
    main()
