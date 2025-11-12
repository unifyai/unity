from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Literal, Any


SessionKind = Literal["actor", "execute"]


@dataclass
class _SessionEntry:
    kind: SessionKind
    handle: Any | None


class ActiveSessionRegistry:
    """Process-local registry guarding a single interactive session.

    An "interactive session" is either Actor.act or TaskScheduler.execute.
    Exactly one may be in-flight at a time. Callers must:
      1) try_reserve(kind, owner_label) before launching
      2) adopt(handle, kind) once a handle is available
      3) release_if(handle) when done
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._entry: Optional[_SessionEntry] = None

    async def is_active(self) -> bool:
        async with self._lock:
            ent = self._entry
            if ent is None:
                return False
            h = ent.handle
            try:
                return h is not None and hasattr(h, "done") and not bool(h.done())
            except Exception:
                return h is not None

    async def current(self) -> Optional[_SessionEntry]:
        async with self._lock:
            return self._entry

    async def try_reserve(self, kind: SessionKind) -> bool:
        """Atomically reserve the singleton slot for kind if free.

        Returns True on success; False if an active (or reserved) session exists.
        """
        async with self._lock:
            if self._entry is not None:
                # If a previous handle is present but finished, free it eagerly
                ent = self._entry
                h = ent.handle
                try:
                    if h is not None and hasattr(h, "done") and bool(h.done()):
                        self._entry = None
                except Exception:
                    pass
            if self._entry is not None:
                return False
            self._entry = _SessionEntry(kind=kind, handle=None)
            return True

    async def adopt(self, handle: Any, kind: SessionKind) -> None:
        """Attach the live handle to the reserved slot (must match kind)."""
        async with self._lock:
            if self._entry is None:
                # Late adoption without reservation – treat as active now
                self._entry = _SessionEntry(kind=kind, handle=handle)
                return
            self._entry.handle = handle

    async def release_if(self, handle: Any) -> None:
        """Release the slot if it's held by this handle (id equality)."""
        async with self._lock:
            if self._entry is None:
                return
            try:
                if self._entry.handle is handle:
                    self._entry = None
            except Exception:
                # Best-effort release; leave entry untouched on unexpected errors
                pass
