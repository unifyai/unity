"""Helpers for serializing chat context safely.

These utilities preserve message structure while redacting heavyweight image
payloads so parent/inspection context dumps remain token-bounded.
"""

from __future__ import annotations

import re
from typing import Any

_DATA_IMAGE_URL_RE = re.compile(
    r"data:image/(?P<mime>[a-zA-Z0-9.+-]+);base64,[A-Za-z0-9+/=\s]+",
)
_BASE64_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
_IMAGE_BLOB_KEYS = {
    "image",
    "image_base64",
    "screenshot",
    "screenshot_b64",
    "thumbnail",
    "b64",
    "data",
}


def _redact_data_image_urls(value: str) -> str:
    """Redact base64 payloads from data:image URLs while preserving MIME."""

    def _repl(match: re.Match[str]) -> str:
        mime = match.group("mime")
        return f"data:image/{mime};base64,<omitted>"

    return _DATA_IMAGE_URL_RE.sub(_repl, value)


def _looks_like_base64_blob(value: str) -> bool:
    """Heuristic for large base64 strings that should never be in context dumps."""
    compact = "".join(value.split())
    if len(compact) < 512:
        return False
    sample = compact[:4096]
    return all(ch in _BASE64_CHARS for ch in sample)


def _sanitize_node(node: Any, *, parent_key: str | None = None) -> Any:
    if isinstance(node, dict):
        out: dict[Any, Any] = {}
        for key, value in node.items():
            out[key] = _sanitize_node(value, parent_key=str(key))
        return out

    if isinstance(node, list):
        return [_sanitize_node(item, parent_key=parent_key) for item in node]

    if isinstance(node, str):
        redacted = _redact_data_image_urls(node)
        if parent_key and parent_key.lower() in _IMAGE_BLOB_KEYS:
            if redacted.startswith("data:image/") or _looks_like_base64_blob(redacted):
                return "<image omitted>"
        return redacted

    return node


def make_messages_safe_for_context_dump(
    messages: list[dict] | None,
) -> list[dict]:
    """Return a deep-copied context list with image blobs redacted.

    This function is intended specifically for prompt/context serialization
    paths (inspection asks, parent-chat headers, simulated prompt builders).
    """
    if not messages:
        return []
    sanitized = _sanitize_node(messages)
    if isinstance(sanitized, list):
        return sanitized
    return []
