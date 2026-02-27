"""
Parser for ``.secrets.json`` — a gitignored file containing secret values
organized by org / team / user / assistant.

The file lives at the repo root and is **never committed to source control**.
Only ``name`` + ``description`` metadata appears in client code; the actual
``value`` fields live exclusively in this file.

File format::

    {
        "org": {
            "<org_id>": {
                "SECRET_NAME": {"value": "...", "description": "..."},
                ...
            }
        },
        "team": {
            "<team_id>": { ... }
        },
        "user": {
            "<user_id>": { ... }
        },
        "assistant": {
            "<assistant_id>": { ... }
        }
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / ".secrets.json"


def load_secrets(
    org_id: int | None = None,
    team_ids: list[int] | None = None,
    user_id: str | None = None,
    assistant_id: int | None = None,
    *,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load and merge secrets for the given identity from ``.secrets.json``.

    Cascade order: org -> team(s) -> user -> assistant (more specific wins
    by ``name``).  When multiple team_ids are provided, they are processed
    in ascending order so higher team_ids take precedence.
    Returns an empty list if the file does not exist.
    """
    secrets_path = path or _DEFAULT_PATH
    if not secrets_path.exists():
        return []

    try:
        data = json.loads(secrets_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to parse %s", secrets_path, exc_info=True)
        return []

    merged: dict[str, dict[str, Any]] = {}

    # org level
    if org_id is not None:
        level_data = data.get("org", {}).get(str(org_id), {})
        for secret_name, secret_info in level_data.items():
            merged[secret_name] = {
                "name": secret_name,
                "value": secret_info.get("value", ""),
                "description": secret_info.get("description", ""),
            }

    # team level (sorted ascending — later teams override earlier)
    if team_ids:
        team_data = data.get("team", {})
        for tid in sorted(team_ids):
            for secret_name, secret_info in team_data.get(str(tid), {}).items():
                merged[secret_name] = {
                    "name": secret_name,
                    "value": secret_info.get("value", ""),
                    "description": secret_info.get("description", ""),
                }

    # user level
    if user_id is not None:
        level_data = data.get("user", {}).get(str(user_id), {})
        for secret_name, secret_info in level_data.items():
            merged[secret_name] = {
                "name": secret_name,
                "value": secret_info.get("value", ""),
                "description": secret_info.get("description", ""),
            }

    # assistant level
    if assistant_id is not None:
        level_data = data.get("assistant", {}).get(str(assistant_id), {})
        for secret_name, secret_info in level_data.items():
            merged[secret_name] = {
                "name": secret_name,
                "value": secret_info.get("value", ""),
                "description": secret_info.get("description", ""),
            }

    return list(merged.values())
