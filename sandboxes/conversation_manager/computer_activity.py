"""
Computer activity tracking for the ConversationManager sandbox.

We intentionally do **not** store screenshots here.

In CodeAct + web mode, the Magnitude agent runs in a separate Chromium instance
(via agent-service). For sandbox UX we mainly need:
- a lightweight "is the computer interface being used?" indicator
- the latest known URL (best-effort, when available)
- a short, bounded list of recent computer actions

This module provides a tiny tracker plus a helper to wrap a `ComputerPrimitives`
instance in-place (by monkey-patching methods) so we can record activity without
changing Unity core classes.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from sandboxes.display.formatting import strip_ansi, truncate


@dataclass(frozen=True)
class ComputerAction:
    ts: float
    kind: str
    detail: str


class ComputerActivity:
    """In-memory tracker for computer activity (bounded history)."""

    def __init__(self, *, max_actions: int = 50) -> None:
        self._max_actions = int(max_actions)
        self._lock = asyncio.Lock()
        self._actions: list[ComputerAction] = []
        self._last_url: Optional[str] = None
        self._connected: Optional[bool] = None
        self._last_error: Optional[str] = None

    async def mark_connected(self, ok: bool, *, error: str | None = None) -> None:
        async with self._lock:
            self._connected = bool(ok)
            self._last_error = error

    def mark_connected_sync(self, ok: bool, *, error: str | None = None) -> None:
        """Synchronous variant for initialization paths (best-effort)."""
        self._connected = bool(ok)
        self._last_error = error

    async def set_url(self, url: str | None) -> None:
        if not url:
            return
        async with self._lock:
            self._last_url = str(url)

    async def record(self, kind: str, detail: str) -> None:
        now = time.time()
        clean = strip_ansi(detail or "")
        async with self._lock:
            self._actions.append(
                ComputerAction(
                    ts=now,
                    kind=str(kind),
                    detail=truncate(clean, 140),
                ),
            )
            if len(self._actions) > self._max_actions:
                self._actions = self._actions[-self._max_actions :]

    async def snapshot(self) -> dict[str, Any]:
        """Return a snapshot for UI rendering."""
        async with self._lock:
            return {
                "connected": self._connected,
                "last_error": self._last_error,
                "last_url": self._last_url,
                "actions": list(self._actions),
            }

    def snapshot_sync(self) -> dict[str, Any]:
        """Best-effort sync snapshot (no locking)."""
        return {
            "connected": self._connected,
            "last_error": self._last_error,
            "last_url": self._last_url,
            "actions": list(self._actions),
        }


def install_computer_activity_hooks(
    *,
    computer_primitives: Any,
    activity: ComputerActivity,
    emit_line: Callable[[str], None] | None = None,
) -> None:
    """
    Patch `ComputerPrimitives` methods in-place to record activity.

    Notes
    -----
    - This is sandbox-only wiring and intentionally best-effort.
    - `emit_line` can be used to push a short status line into the REPL/GUI log.
    """

    def _emit(s: str) -> None:
        if emit_line is None:
            return
        try:
            emit_line(s)
        except Exception:
            return

    def _wrap_async(
        method_name: str,
        *,
        formatter: Callable[[tuple[Any, ...], dict[str, Any]], tuple[str, str]],
        post: Callable[[Any], Awaitable[None]] | None = None,
    ) -> None:
        orig = getattr(computer_primitives, method_name, None)
        if orig is None or not callable(orig):
            return

        async def _wrapped(*args: Any, **kwargs: Any) -> Any:
            kind, detail = formatter(args, kwargs)
            await activity.record(kind, detail)
            _emit(f"[computer] {kind}: {detail}")
            try:
                out = await orig(*args, **kwargs)
            except Exception as exc:
                await activity.mark_connected(
                    False,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            if post is not None:
                try:
                    await post(out)
                except Exception:
                    pass
            return out

        setattr(computer_primitives, method_name, _wrapped)

    async def _post_navigate(_out: Any) -> None:
        # Best-effort: ask backend for current URL (works for magnitude + mock).
        try:
            url = await computer_primitives.computer.backend.get_current_url()
            await activity.set_url(url)
        except Exception:
            pass

    _wrap_async(
        "navigate",
        formatter=lambda a, k: (
            "navigate",
            str((k.get("url") if "url" in k else (a[0] if a else ""))),
        ),
        post=_post_navigate,
    )
    _wrap_async(
        "act",
        formatter=lambda a, k: (
            "act",
            str(
                (k.get("instruction") if "instruction" in k else (a[0] if a else "")),
            ),
        ),
    )
    _wrap_async(
        "observe",
        formatter=lambda a, k: (
            "observe",
            str((k.get("query") if "query" in k else (a[0] if a else ""))),
        ),
    )
    _wrap_async(
        "query",
        formatter=lambda a, k: (
            "query",
            str((k.get("query") if "query" in k else (a[0] if a else ""))),
        ),
    )
