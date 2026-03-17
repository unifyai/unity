"""
Helpers for Vantage Microsoft 365 authentication.

These scripts read the client configuration from `__init__.py`, which is the
current source of truth for the Vantage demo credentials.
"""

from pathlib import Path
import os
import re

import requests

DEFAULT_SCOPES = (
    "Mail.ReadWrite Files.ReadWrite.All Notes.ReadWrite.All " "User.Read offline_access"
)

DIR = Path(__file__).parent
INIT_FILE = DIR / "__init__.py"


def _load_ms365_config():
    text = INIT_FILE.read_text(encoding="utf-8")
    values = dict(
        re.findall(r'"name": "(MS365_[A-Z_]+)",\s*"value": "([^"]+)"', text),
    )

    required = [
        "MS365_TENANT_ID",
        "MS365_CLIENT_ID",
        "MS365_REFRESH_TOKEN",
    ]
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise RuntimeError(
            "Missing Vantage MS365 config values in __init__.py: " + ", ".join(missing),
        )
    return values


def load_tenant_and_client():
    values = _load_ms365_config()
    return values["MS365_TENANT_ID"], values["MS365_CLIENT_ID"]


def refresh_access_token(scope=DEFAULT_SCOPES):
    values = _load_ms365_config()
    tenant_id = values["MS365_TENANT_ID"]
    client_id = values["MS365_CLIENT_ID"]
    refresh_token = values["MS365_REFRESH_TOKEN"]

    r = requests.post(
        f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
        data={
            "client_id": client_id,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "scope": scope,
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"Refresh token exchange failed ({r.status_code}): {r.text[:300]}",
        )
    return r.json()["access_token"]


def resolve_access_token(token_file=None):
    if token_file:
        path = Path(token_file)
        if not path.exists():
            raise FileNotFoundError(f"Token file not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    env_token = os.environ.get("MS365_ACCESS_TOKEN", "").strip()
    if env_token:
        return env_token

    return refresh_access_token()
