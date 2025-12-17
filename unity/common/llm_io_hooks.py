"""
LLM I/O Debug Hooks
===================

Monkeypatches the ``unify`` client to capture the *exact* request payload
that is sent to the chat completions endpoint or read from cache.

Install via :pyfunc:`install_llm_io_hooks` early at startup (called
automatically from ``unity/__init__.py`` when ``LLM_IO_DEBUG`` is enabled).

The captured payloads are written to ``.llm_io_debug/<session>/`` as text files.
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

# Module-level state
_HOOKS_INSTALLED = False
_LLM_IO_DIR: str | None = None


def _get_socket_subdir() -> str:
    """Determine the log subdirectory for LLM I/O debug files.

    Returns a datetime-prefixed directory name for natural time-based ordering:
        - UNITY_LOG_SUBDIR if set (e.g., '2025-12-05T14-30-45_unity_dev_ttys042')
        - Falls back to UNITY_TEST_SOCKET for legacy compatibility
        - 'standalone' for direct invocations outside parallel_run.sh
    """
    # Prefer the datetime-prefixed log subdir if available
    log_subdir = os.environ.get("UNITY_LOG_SUBDIR", "").strip()
    if log_subdir:
        return log_subdir
    # Fallback to socket name for backward compatibility
    socket = os.environ.get("UNITY_TEST_SOCKET", "").strip()
    if socket:
        return socket
    return "standalone"


def _get_repo_root() -> Path:
    """Determine the repository root directory.

    Prefers UNITY_LOG_ROOT env var if set, allowing explicit worktree targeting.
    Otherwise falls back to detecting the repo root from this file's location,
    which correctly resolves to the worktree when running from one.

    This fixes the issue where Background Agents (which use git worktrees)
    would have logs written to the main repo instead of their worktree.
    """
    # Allow explicit override for flexibility
    log_root = os.environ.get("UNITY_LOG_ROOT", "").strip()
    if log_root:
        return Path(log_root)

    # Derive repo root from this file's location (works correctly in worktrees)
    # __file__ is unity/common/llm_io_hooks.py, so go up 2 levels to repo root
    try:
        return Path(__file__).resolve().parent.parent.parent
    except Exception:
        # Fallback to cwd if __file__ resolution fails
        return Path(os.getcwd())


def _ensure_io_dir() -> str | None:
    """Ensure the per-session LLM I/O debug directory exists.

    Directory structure:
        .llm_io_debug/{socket_or_standalone}/{session_id}/
    """
    global _LLM_IO_DIR
    if _LLM_IO_DIR is not None:
        return _LLM_IO_DIR

    try:
        from unity.constants import SESSION_ID
    except Exception:
        return None

    try:
        # Socket-scoped subdirectory for terminal isolation
        socket_subdir = _get_socket_subdir()
        root = _get_repo_root() / ".llm_io_debug" / socket_subdir
        root.mkdir(parents=True, exist_ok=True)
        session_safe = re.sub(r"[^0-9A-Za-z._-]", "-", str(SESSION_ID))
        session_dir = root / session_safe
        session_dir.mkdir(parents=True, exist_ok=True)
        _LLM_IO_DIR = str(session_dir)
    except Exception:
        _LLM_IO_DIR = None

    return _LLM_IO_DIR


def _write_llm_io(
    header: str,
    body: Any,
    *,
    label: str | None = None,
    kind: str = "io",
) -> None:
    """Write a single LLM I/O artifact to the debug folder.

    Args:
        header: Header line for the file content.
        body: The payload to write (dict, str, or other serializable).
        label: Optional label (e.g., endpoint name) to include in header.
        kind: Type of I/O - "request" or "response". Included in filename
              after timestamp to maintain alphabetical ordering.
    """
    io_dir = _ensure_io_dir()
    if io_dir is None:
        return

    try:
        now = datetime.now(timezone.utc)
        hhmmss = now.strftime("%H%M%S")
        ns = time.time_ns() % 1_000_000_000
        # Include kind in filename after timestamp for alphabetical ordering
        base = f"{hhmmss}_{ns:09d}_{kind}"
        path = Path(io_dir) / f"{base}.txt"

        # Handle filename collision
        i = 1
        while path.exists():
            path = Path(io_dir) / f"{base}_{i}.txt"
            i += 1

        # Normalize body to string
        if not isinstance(body, str):
            try:
                body = json.dumps(body, indent=4, default=str)
            except Exception:
                body = str(body)

        label_prefix = f"[{label}] " if label else ""
        with path.open("w", encoding="utf-8") as f:
            f.write(f"🔄 {label_prefix}{header}\n")
            f.write(body.rstrip())
            f.write("\n")

        # Log to console
        try:
            from unity.constants import LOGGER

            LOGGER.info(f"📝 LLM {kind} written to {path}")
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
    """Wrap _generate_non_stream to capture exact request/response."""

    if is_async:

        @functools.wraps(original_fn)
        async def async_wrapper(self, endpoint, prompt, **kwargs):
            # Build the kw dict the same way the original does
            kw = self._handle_kw(
                prompt=prompt,
                endpoint=endpoint,
                stream=False,
                stream_options=None,
                use_custom_keys=kwargs.get("use_custom_keys", False),
                tags=kwargs.get("tags"),
                drop_params=kwargs.get("drop_params"),
                region=kwargs.get("region"),
                log_query_body=kwargs.get("log_query_body"),
                log_response_body=kwargs.get("log_response_body"),
            )

            # Log the request
            try:
                _write_llm_io(
                    "LLM request ➡️:",
                    _serialize_kw(kw),
                    label=endpoint,
                    kind="request",
                )
            except Exception:
                pass

            # Call original
            result = await original_fn(self, endpoint, prompt, **kwargs)

            # Log the response
            try:
                if hasattr(result, "model_dump"):
                    resp_body = result.model_dump()
                else:
                    resp_body = result
                _write_llm_io(
                    "LLM response ⬅️:",
                    resp_body,
                    label=endpoint,
                    kind="response",
                )
            except Exception:
                pass

            return result

        return async_wrapper

    else:

        @functools.wraps(original_fn)
        def sync_wrapper(self, endpoint, prompt, **kwargs):
            # Build the kw dict the same way the original does
            kw = self._handle_kw(
                prompt=prompt,
                endpoint=endpoint,
                stream=False,
                stream_options=None,
                use_custom_keys=kwargs.get("use_custom_keys", False),
                tags=kwargs.get("tags"),
                drop_params=kwargs.get("drop_params"),
                region=kwargs.get("region"),
                log_query_body=kwargs.get("log_query_body"),
                log_response_body=kwargs.get("log_response_body"),
            )

            # Log the request
            try:
                _write_llm_io(
                    "LLM request ➡️:",
                    _serialize_kw(kw),
                    label=endpoint,
                    kind="request",
                )
            except Exception:
                pass

            # Call original
            result = original_fn(self, endpoint, prompt, **kwargs)

            # Log the response
            try:
                if hasattr(result, "model_dump"):
                    resp_body = result.model_dump()
                else:
                    resp_body = result
                _write_llm_io(
                    "LLM response ⬅️:",
                    resp_body,
                    label=endpoint,
                    kind="response",
                )
            except Exception:
                pass

            return result

        return sync_wrapper


def install_llm_io_hooks() -> bool:
    """
    Install monkeypatches on the unify client to capture LLM I/O.

    This should be called early at startup when LLM_IO_DEBUG is enabled.
    Returns True if hooks were installed, False if already installed or failed.
    """
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return False

    try:
        from unify.universal_api.clients.uni_llm import AsyncUnify, Unify

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
