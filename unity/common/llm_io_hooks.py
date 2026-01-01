"""
LLM I/O Debug Hooks
===================

Monkeypatches the ``unillm`` client to:
1. Track cache hit/miss statistics (always enabled via :pyfunc:`get_cache_stats`)
2. Optionally write log files (when ``LLM_IO_LOG`` is enabled)

Install via :pyfunc:`install_llm_io_hooks` early at startup (called
automatically from ``unity/__init__.py``).

When ``LLM_IO_LOG`` is enabled, request+response payloads are written to
``logs/llm/<session>/`` as combined files:

- During the call: ``{timestamp}_pending.txt`` (contains request only)
- After completion: ``{timestamp}_hit.txt`` or ``{timestamp}_miss.txt``
  (contains both request and response, with cache status in filename)

If an LLM call hangs or crashes, the ``_pending.txt`` file remains as evidence
of the incomplete request.
"""

from __future__ import annotations

import functools
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from unity.settings import SETTINGS

# Module-level state
_HOOKS_INSTALLED = False
_LLM_IO_DIR: str | None = None

# Global cache statistics (since unillm doesn't provide a global counter)
_CACHE_HITS = 0
_CACHE_MISSES = 0


def get_cache_stats() -> dict[str, int | float]:
    """Get cache hit/miss statistics for LLM calls.

    Returns a dict with:
        - hits: Number of cache hits
        - misses: Number of cache misses
        - hit_rate: Percentage of hits (0.0 if no calls)
    """
    total = _CACHE_HITS + _CACHE_MISSES
    hit_rate = (_CACHE_HITS / total * 100) if total > 0 else 0.0
    return {
        "hits": _CACHE_HITS,
        "misses": _CACHE_MISSES,
        "hit_rate": hit_rate,
    }


def reset_cache_stats() -> None:
    """Reset the cache statistics counters to zero."""
    global _CACHE_HITS, _CACHE_MISSES
    _CACHE_HITS = 0
    _CACHE_MISSES = 0


def _record_cache_status(cache_status: str) -> None:
    """Record a cache hit or miss in the global stats."""
    global _CACHE_HITS, _CACHE_MISSES
    if cache_status == "hit":
        _CACHE_HITS += 1
    elif cache_status == "miss":
        _CACHE_MISSES += 1
    # Ignore "unknown" or other statuses


def _derive_socket_name() -> str:
    """Derive a unique socket name from the terminal's TTY device.

    Mirrors the logic in tests/_shell_common.sh::_derive_socket_name() to ensure
    consistent naming whether tests are run via parallel_run.sh or directly via pytest.

    Returns:
        - 'unity_dev_ttysXXX' if running in a TTY (e.g., terminal session)
        - 'unity_pidXXX' if not running in a TTY (e.g., background process)
    """
    import sys

    try:
        # Try to get the TTY device path (e.g., '/dev/ttys042')
        tty_path = os.ttyname(sys.stdout.fileno())
        # Sanitize: /dev/ttys042 -> unity_dev_ttys042
        tty_id = tty_path.replace("/", "_")
        return f"unity{tty_id}"
    except (OSError, AttributeError):
        # Not a TTY (e.g., piped output, background process)
        return f"unity_pid{os.getpid()}"


def _get_socket_subdir() -> str:
    """Determine the log subdirectory for LLM I/O debug files.

    Returns a datetime-prefixed directory name for natural time-based ordering:
        - UNITY_LOG_SUBDIR if set (e.g., '2025-12-05T14-30-45_unity_dev_ttys042')
        - Falls back to UNITY_TEST_SOCKET for legacy compatibility
        - Derives terminal ID for direct pytest invocations (same as parallel_run.sh would)
    """
    # Prefer the datetime-prefixed log subdir if available
    if SETTINGS.UNITY_LOG_SUBDIR:
        return SETTINGS.UNITY_LOG_SUBDIR
    # Fallback to socket name for backward compatibility
    if SETTINGS.UNITY_TEST_SOCKET:
        return SETTINGS.UNITY_TEST_SOCKET
    # Derive terminal ID (same logic as _shell_common.sh) for direct pytest invocations
    socket_name = _derive_socket_name()
    return f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_{socket_name}"


def _get_repo_root() -> Path:
    """Determine the repository root directory.

    Prefers UNITY_LOG_ROOT if set in SETTINGS, allowing explicit worktree targeting.
    Otherwise falls back to detecting the repo root from this file's location,
    which correctly resolves to the worktree when running from one.

    This fixes the issue where Background Agents (which use git worktrees)
    would have logs written to the main repo instead of their worktree.
    """
    # Allow explicit override for flexibility
    if SETTINGS.UNITY_LOG_ROOT:
        return Path(SETTINGS.UNITY_LOG_ROOT)

    # Derive repo root from this file's location (works correctly in worktrees)
    # __file__ is unity/common/llm_io_hooks.py, so go up 2 levels to repo root
    try:
        return Path(__file__).resolve().parent.parent.parent
    except Exception:
        # Fallback to cwd if __file__ resolution fails
        return Path(os.getcwd())


def _ensure_io_dir() -> str | None:
    """Ensure the per-session LLM I/O debug directory exists.

    Returns None if LLM_IO_LOG is not enabled (file writing is skipped).

    Directory structure:
        logs/llm/{datetime}_{socket_name}/{session_id}/
    """
    global _LLM_IO_DIR
    if _LLM_IO_DIR is not None:
        return _LLM_IO_DIR

    # Only write files when LLM_IO_LOG is enabled
    if not SETTINGS.LLM_IO_LOG:
        return None

    try:
        from unity.constants import SESSION_ID
    except Exception:
        return None

    try:
        # Socket-scoped subdirectory for terminal isolation
        socket_subdir = _get_socket_subdir()
        root = _get_repo_root() / "logs" / "llm" / socket_subdir
        root.mkdir(parents=True, exist_ok=True)
        session_safe = re.sub(r"[^0-9A-Za-z._-]", "-", str(SESSION_ID))
        session_dir = root / session_safe
        session_dir.mkdir(parents=True, exist_ok=True)
        _LLM_IO_DIR = str(session_dir)
    except Exception:
        _LLM_IO_DIR = None

    return _LLM_IO_DIR


def _normalize_body(body: Any) -> str:
    """Normalize a body payload to a string for writing."""
    if isinstance(body, str):
        return body
    try:
        return json.dumps(body, indent=4, default=str)
    except Exception:
        return str(body)


def _write_request_pending(
    request_body: Any,
    *,
    label: str | None = None,
) -> Path | None:
    """Write the request payload immediately with a _pending suffix.

    Returns the file path so we can append the response and rename later.
    If the LLM call hangs/crashes, the _pending file remains as evidence.
    """
    io_dir = _ensure_io_dir()
    if io_dir is None:
        return None

    try:
        now = datetime.now(timezone.utc)
        hhmmss = now.strftime("%H%M%S")
        ns = time.time_ns() % 1_000_000_000
        base = f"{hhmmss}_{ns:09d}_pending"
        path = Path(io_dir) / f"{base}.txt"

        # Handle filename collision
        i = 1
        while path.exists():
            path = Path(io_dir) / f"{base}_{i}.txt"
            i += 1

        body_str = _normalize_body(request_body)
        label_prefix = f"[{label}] " if label else ""

        with path.open("w", encoding="utf-8") as f:
            f.write(f"🔄 {label_prefix}LLM request ➡️\n")
            f.write(body_str.rstrip())
            f.write("\n")

        # Log to console
        try:
            from unity.constants import LOGGER

            LOGGER.info(f"📝 LLM request (pending) written to {path}")
        except Exception:
            pass

        return path
    except Exception:
        return None


def _append_response_and_finalize(
    pending_path: Path | None,
    response_body: Any,
    cache_status: str,
    *,
    label: str | None = None,
) -> None:
    """Append the response to the pending file and rename to reflect cache status.

    The final filename will be: {timestamp}_hit.txt or {timestamp}_miss.txt
    """
    if pending_path is None or not pending_path.exists():
        return

    try:
        body_str = _normalize_body(response_body)
        label_prefix = f"[{label}] " if label else ""

        # Append response to the file
        with pending_path.open("a", encoding="utf-8") as f:
            f.write(f"\n🔄 {label_prefix}LLM response ⬅️ [cache: {cache_status}]\n")
            f.write(body_str.rstrip())
            f.write("\n")

        # Rename from _pending to _hit or _miss
        new_name = pending_path.name.replace("_pending", f"_{cache_status}")
        new_path = pending_path.parent / new_name
        pending_path.rename(new_path)

        # Log to console
        try:
            from unity.constants import LOGGER

            LOGGER.info(f"📝 LLM I/O finalized: {new_path}")
        except Exception:
            pass
    except Exception:
        # Silent best-effort
        pass


def _serialize_kw(kw: dict) -> dict:
    """Serialize the kw dict for logging, handling Pydantic models and special types."""
    from pydantic import BaseModel

    def _convert(obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        if hasattr(obj, "model_json_schema"):
            # Pydantic model class (not instance)
            return {"__pydantic_schema__": obj.model_json_schema()}
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

    return _convert(kw)


def _wrap_generate_non_stream(
    original_fn: Callable,
    *,
    is_async: bool,
) -> Callable:
    """Wrap _generate_non_stream to capture exact request/response and cache status."""

    if is_async:

        @functools.wraps(original_fn)
        async def async_wrapper(self, endpoint, prompt, **kwargs):
            from unillm import acapture_cache_events

            # Build the kw dict the same way the original does
            kw = self._handle_kw(
                prompt=prompt,
                endpoint=endpoint,
                stream=False,
                stream_options=None,
            )

            # Write request immediately (before LLM call) so we don't lose it if call hangs
            pending_path = None
            try:
                request_body = _serialize_kw(kw)
                pending_path = _write_request_pending(request_body, label=endpoint)
            except Exception:
                pass

            # Call original with cache event capture
            async with acapture_cache_events() as events:
                result = await original_fn(self, endpoint, prompt, **kwargs)

            # Extract cache status, record stats, and finalize the log file
            cache_status = events[0]["cache_status"] if events else "unknown"
            _record_cache_status(cache_status)
            try:
                if hasattr(result, "model_dump"):
                    resp_body = result.model_dump()
                else:
                    resp_body = result
                _append_response_and_finalize(
                    pending_path,
                    resp_body,
                    cache_status,
                    label=endpoint,
                )
            except Exception:
                pass

            return result

        return async_wrapper

    else:

        @functools.wraps(original_fn)
        def sync_wrapper(self, endpoint, prompt, **kwargs):
            from unillm import capture_cache_events

            # Build the kw dict the same way the original does
            kw = self._handle_kw(
                prompt=prompt,
                endpoint=endpoint,
                stream=False,
                stream_options=None,
            )

            # Write request immediately (before LLM call) so we don't lose it if call hangs
            pending_path = None
            try:
                request_body = _serialize_kw(kw)
                pending_path = _write_request_pending(request_body, label=endpoint)
            except Exception:
                pass

            # Call original with cache event capture
            with capture_cache_events() as events:
                result = original_fn(self, endpoint, prompt, **kwargs)

            # Extract cache status, record stats, and finalize the log file
            cache_status = events[0]["cache_status"] if events else "unknown"
            _record_cache_status(cache_status)
            try:
                if hasattr(result, "model_dump"):
                    resp_body = result.model_dump()
                else:
                    resp_body = result
                _append_response_and_finalize(
                    pending_path,
                    resp_body,
                    cache_status,
                    label=endpoint,
                )
            except Exception:
                pass

            return result

        return sync_wrapper


def install_llm_io_hooks() -> bool:
    """
    Install monkeypatches on the unify client to capture LLM I/O.

    This should be called early at startup when LLM_IO_LOG is enabled.
    Returns True if hooks were installed, False if already installed or failed.
    """
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return False

    try:
        from unillm import AsyncUnify, Unify

        # Wrap the non-stream methods (these are where cache + API calls happen)
        Unify._generate_non_stream = _wrap_generate_non_stream(
            Unify._generate_non_stream,
            is_async=False,
        )
        AsyncUnify._generate_non_stream = _wrap_generate_non_stream(
            AsyncUnify._generate_non_stream,
            is_async=True,
        )

        _HOOKS_INSTALLED = True

        try:
            from unity.constants import LOGGER

            LOGGER.info("📡 LLM I/O debug hooks installed")
        except Exception:
            pass

        return True

    except Exception as e:
        try:
            from unity.constants import LOGGER

            LOGGER.warning(f"Failed to install LLM I/O hooks: {e}")
        except Exception:
            pass
        return False
