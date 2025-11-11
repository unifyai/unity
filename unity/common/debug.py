from __future__ import annotations

import inspect
import sys
from datetime import datetime
from typing import Any

__all__ = ["CURSOR_DEBUG_LOG"]


def CURSOR_DEBUG_LOG(*message_parts: Any) -> None:
    """
    TEMPORARY debug logger used exclusively during failure investigation.
    Always logs unconditionally to stderr with a timestamp and caller location.
    """
    caller = inspect.stack()[1]
    location = f"{caller.filename}:{caller.lineno}:{caller.function}"
    timestamp = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
    text = " ".join(str(p) for p in message_parts)
    print(
        f"[CURSOR-DEBUG] {timestamp} {location} :: {text}",
        file=sys.stderr,
        flush=True,
    )
