#!/usr/bin/env python3
"""
Create or fetch a local assistant and print the env vars needed to run
unity locally.

A "local assistant" is stored in Orchestra with `is_local=True` and runs
unity on your machine instead of on GKE. Preview assistants are stored in
staging Orchestra but use preview communication/adapters at runtime.

Usage:
    # Create a new local assistant (or fetch existing by name):
    python scripts/dev/local_assistant.py --name "Dev Assistant"

    # Fetch an existing local assistant by ID:
    python scripts/dev/local_assistant.py --id 42

    # Target production:
    python scripts/dev/local_assistant.py --name "Dev" --env production

    # Target preview:
    python scripts/dev/local_assistant.py --name "Dev" --env preview

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
    "production": "https://api.unify.ai/v0",
    "staging": "https://api.staging.internal.saas.unify.ai/v0",
    "preview": "https://api.staging.internal.saas.unify.ai/v0",
}
COMMS_URLS = {
    "production": "https://unity-comms-app-262420637606.us-central1.run.app",
    "staging": "https://unity-comms-app-staging-262420637606.us-central1.run.app",
    "preview": "https://unity-comms-app-preview-262420637606.us-central1.run.app",
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
    """Fetch an assistant by ID from the list endpoint."""
    resp = requests.get(f"{orchestra_url}/assistant", headers=_headers(api_key))
    resp.raise_for_status()
    for a in resp.json().get("info", []):
        if str(a.get("agent_id")) == str(agent_id):
            return a
    return None


def _create(
    orchestra_url: str,
    api_key: str,
    first_name: str,
    surname: str,
    deploy_env: str,
    age: int = 25,
    nationality: str = "US",
    about: str = "Local Assistant",
    voice_id: str = "ThT5KcBeYPX3keUQqHPh",
    voice_provider: str = "elevenlabs",
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
        "is_local": True,
    }
    if deploy_env == "preview":
        payload["deploy_env"] = "preview"
    resp = requests.post(
        f"{orchestra_url}/assistant",
        json=payload,
        headers=_headers(api_key),
    )
    resp.raise_for_status()
    return resp.json().get("info")


def _get_user_info(orchestra_url: str, api_key: str):
    """Fetch the authenticated user's basic info."""
    resp = requests.get(f"{orchestra_url}/user/basic-info", headers=_headers(api_key))
    resp.raise_for_status()
    return resp.json()


def _v(val):
    if val is None:
        return ""
    return str(val)


def _print_env(assistant: dict, user: dict, api_key: str, deploy_env: str):
    """Print a .env file matching the unity .env structure.

    Assistant and user fields are populated from the API response.
    Secrets and service keys are left blank for the developer to fill in.
    """
    first = _v(assistant.get("first_name"))
    surname = _v(assistant.get("surname"))
    name = f"{first} {surname}".strip() if first else ""
    user_first = _v(user.get("first"))
    user_last = _v(user.get("last"))
    user_name = f"{user_first} {user_last}".strip() if user_first else ""

    lines = [
        f"UNIFY_BASE_URL={ORCHESTRA_URLS[deploy_env]}",
        "LIVEKIT_SIP_URI=",
        "LIVEKIT_URL=",
        "LIVEKIT_API_KEY=",
        "LIVEKIT_API_SECRET=",
        "OPENAI_API_KEY=",
        "ANTHROPIC_API_KEY=",
        f"USER_ID={_v(assistant.get('user_id'))}",
        f"UNIFY_KEY={api_key}",
        f"UNITY_COMMS_URL={COMMS_URLS[deploy_env]}",
        "DEEPGRAM_API_KEY=",
        "CARTESIA_API_KEY=",
        "ELEVEN_API_KEY=",
        "GOOGLE_APPLICATION_CREDENTIALS=",
        f"ASSISTANT_ID={_v(assistant.get('agent_id'))}",
        f'USER_NAME="{user_name}"' if user_name else "USER_NAME=",
        f"ASSISTANT_NAME={name}",
        f"ASSISTANT_AGE={_v(assistant.get('age'))}",
        f'ASSISTANT_NATIONALITY="{_v(assistant.get("nationality"))}"',
        f'ASSISTANT_ABOUT="{_v(assistant.get("about"))}"',
        f"ASSISTANT_NUMBER={_v(assistant.get('phone'))}",
        f"ASSISTANT_EMAIL={_v(assistant.get('email'))}",
        f"USER_NUMBER={_v(user.get('phone_number'))}",
        f"USER_EMAIL={_v(user.get('email'))}",
        "ORCHESTRA_ADMIN_KEY=",
        f"DEPLOY_ENV={deploy_env}",
        "SHARED_UNIFY_KEY=",
        f"VOICE_PROVIDER={_v(assistant.get('voice_provider'))}",
        f"VOICE_ID={_v(assistant.get('voice_id'))}",
        "TAVILY_API_KEY=",
        "PROJECT_ID=",
    ]
    print("\n".join(lines))
    print(
        "The env variables left empty are secrets that you can configure "
        "through the GCP secret manager...",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create or fetch a local assistant and print env vars.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--name",
        help="Assistant name (format: 'FirstName' or 'FirstName Surname')",
    )
    group.add_argument("--id", type=int, help="Existing assistant agent_id")
    parser.add_argument(
        "--env",
        choices=["production", "staging", "preview"],
        default="staging",
        help="Target deploy environment (default: staging)",
    )
    parser.add_argument(
        "--age",
        type=int,
        default=25,
        help="Age of the assistant (default: 25)",
    )
    parser.add_argument("--nationality", default="US", help="Nationality (default: US)")
    parser.add_argument(
        "--about",
        default="Local Assistant",
        help="Description (default: Local Assistant)",
    )
    parser.add_argument(
        "--voice-id",
        default="ThT5KcBeYPX3keUQqHPh",
        help="Voice ID (default: ThT5KcBeYPX3keUQqHPh)",
    )
    parser.add_argument(
        "--voice-provider",
        default="elevenlabs",
        help="Voice provider (default: elevenlabs)",
    )
    parser.add_argument(
        "--voice-mode",
        choices=["tts", "sts"],
        default="tts",
        help="Voice mode (default: tts)",
    )
    args = parser.parse_args()

    env = args.env
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
                deploy_env=env,
                age=args.age,
                nationality=args.nationality,
                about=args.about,
                voice_id=args.voice_id,
                voice_provider=args.voice_provider,
            )
            print(
                f"# Created assistant id={assistant['agent_id']}",
                file=sys.stderr,
            )

    user = _get_user_info(orchestra_url, api_key)
    _print_env(assistant, user, api_key, deploy_env=env)


if __name__ == "__main__":
    main()
