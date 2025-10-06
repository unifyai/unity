from __future__ import annotations

import asyncio


class InterjectAdapter:
    """Adapter that would forward interjections as events.

    Not scheduled yet to avoid interfering with the legacy loop during
    delegation. When the evented path is fully enabled, this will consume
    `orchestrator.interject_queue` and post `interjected` events.
    """

    def __init__(self, orchestrator: "Orchestrator") -> None:
        self._orch = orchestrator

    def schedule(self) -> asyncio.Task | None:  # pragma: no cover - skeleton
        if self._orch._tg is None:
            raise RuntimeError("TaskGroup not initialized")

        async def _task():
            while True:
                payload = await self._orch.interject_queue.get()
                await self._orch.events.put({"type": "interjected", "content": payload})

        return self._orch._tg.create_task(_task())


class ControlAdapter:
    """Adapter that would convert control signals into events.

    Not scheduled yet; when enabled it should observe cancel/pause/resume and
    post `cancel_requested`, `pause_requested`, `resume_requested`.
    """

    def __init__(self, orchestrator: "Orchestrator") -> None:
        self._orch = orchestrator

    def schedule(self) -> asyncio.Task | None:  # pragma: no cover - skeleton
        if self._orch._tg is None:
            raise RuntimeError("TaskGroup not initialized")

        async def _watch_cancel():
            await self._orch.cancel_event.wait()
            await self._orch.events.put({"type": "cancel_requested", "reason": None})

        return self._orch._tg.create_task(_watch_cancel())
