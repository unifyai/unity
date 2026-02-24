from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any


def now_utc_iso() -> str:
    """Return current UTC time with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def monotonic_ms() -> int:
    """Return monotonic clock in milliseconds."""
    return time.monotonic_ns() // 1_000_000


def short_hash(value: str, *, length: int = 12) -> str:
    """Return a short stable hash for correlation IDs."""
    digest = hashlib.sha1((value or "").encode("utf-8")).hexdigest()
    return digest[:length]


def content_trace_id(prefix: str, content: str) -> str:
    """Build a stable ID from content."""
    return f"{prefix}-{short_hash(content or '')}"


def payload_trace_id(prefix: str, channel: str, event_json: str) -> str:
    """Build a stable ID from a transport payload."""
    return f"{prefix}-{short_hash(f'{channel}|{event_json}')}"


def trace_kv(namespace: str, **fields: Any) -> str:
    """Render a structured, grep-friendly trace line."""
    parts: list[str] = []
    for key, value in fields.items():
        if isinstance(value, (dict, list, tuple)):
            value_text = json.dumps(
                value,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        elif value is None:
            value_text = "null"
        else:
            value_text = str(value)
        parts.append(f"{key}={value_text.replace(chr(10), '\\n')}")
    suffix = " ".join(parts)
    if suffix:
        return f"[TRACE::{namespace}] {suffix}"
    return f"[TRACE::{namespace}]"
