"""
Parser for ``.secrets.json`` — a gitignored file containing secret values
organized by org / user / assistant.

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
    user_id: str | None = None,
    assistant_id: int | None = None,
    *,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load and merge secrets for the given identity from ``.secrets.json``.

    Cascade order: org -> user -> assistant (more specific wins by ``name``).
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

    for level_key, id_value in [
        ("org", str(org_id) if org_id is not None else None),
        ("user", str(user_id) if user_id is not None else None),
        ("assistant", str(assistant_id) if assistant_id is not None else None),
    ]:
        if id_value is None:
            continue
        level_data = data.get(level_key, {}).get(id_value, {})
        for secret_name, secret_info in level_data.items():
            merged[secret_name] = {
                "name": secret_name,
                "value": secret_info.get("value", ""),
                "description": secret_info.get("description", ""),
            }

    return list(merged.values())
