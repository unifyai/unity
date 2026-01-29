"""
Sandbox-only SimulatedActor wrapper.

Goal:
- Keep production `unity.actor.simulated.SimulatedActor` unchanged.
- Make sandbox runs feel realistic by:
  - emitting throttled progress notifications (so the UI isn't spammy)
  - producing an actionable completion result (found / not found / ask for missing info)

This is intentionally generic and reusable across scenarios.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Type

from pydantic import BaseModel

from unity.actor.simulated import SimulatedActor
from unity.common.async_tool_loop import SteerableToolHandle


class _ThrottledHandle(SteerableToolHandle):
    """Proxy a handle but rate-limit next_notification()."""

    def __init__(self, inner: SteerableToolHandle, *, interval_s: float) -> None:
        self._inner = inner
        self._interval_s = float(max(0.0, interval_s))

    # ---- SteerableToolHandle surface (delegate) ----
    async def ask(self, question: str, **kwargs):  # type: ignore[override]
        return await self._inner.ask(question, **kwargs)

    def interject(self, message: str, **kwargs):  # type: ignore[override]
        return self._inner.interject(message, **kwargs)

    async def pause(self):  # type: ignore[override]
        return await self._inner.pause()

    async def resume(self):  # type: ignore[override]
        return await self._inner.resume()

    def stop(self, reason: str | None = None, **kwargs):  # type: ignore[override]
        return self._inner.stop(reason, **kwargs)

    async def result(self):  # type: ignore[override]
        return await self._inner.result()

    def done(self) -> bool:  # type: ignore[override]
        return bool(self._inner.done())

    async def next_clarification(self):  # type: ignore[override]
        return await self._inner.next_clarification()

    async def answer_clarification(self, call_id: str, answer: str) -> None:  # type: ignore[override]
        return await self._inner.answer_clarification(call_id, answer)

    async def next_notification(self):  # type: ignore[override]
        if self._interval_s > 0:
            await asyncio.sleep(self._interval_s)
        return await self._inner.next_notification()


class SandboxSimulatedActor(SimulatedActor):
    """
    SimulatedActor with sandbox-only behavior improvements.

    - Throttles notification cadence (prevents event spam).
    - Generates an actionable planned result using the actor's LLM and assigns it
      onto the underlying simulated handle (best-effort).
    """

    def __init__(
        self,
        *args: Any,
        notification_interval_s: float = 1.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._notification_interval_s = float(max(0.0, notification_interval_s))

    async def act(
        self,
        description: str,
        *,
        clarification_enabled: bool = True,
        response_format: Optional[Type[BaseModel]] = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        entrypoint: Optional[int] = None,
        session_suffix: Optional[str] = None,
        **kwargs: Any,
    ) -> SteerableToolHandle:
        handle = await super().act(
            description,
            clarification_enabled=clarification_enabled,
            response_format=response_format,
            _parent_chat_context=_parent_chat_context,
            _clarification_up_q=_clarification_up_q,
            _clarification_down_q=_clarification_down_q,
            entrypoint=entrypoint,
            session_suffix=session_suffix,
            **kwargs,
        )

        # Best-effort: generate an actionable completion message and assign it to
        # the underlying simulated handle (used when the duration/step limit completes).
        async def _prebake() -> None:
            try:
                # Use the actor's stateful LLM + system message (includes simulation_guidance).
                prompt = (
                    "You are simulating an Actor execution.\n"
                    "Return ONE concise, user-facing outcome message.\n"
                    "Rules:\n"
                    "- If an identifier is missing, ask for it.\n"
                    "- If lookup fails, say not found and what you need.\n"
                    "- Do not invent real side effects unless explicitly requested.\n"
                    "- Keep to 1-3 sentences.\n\n"
                    f"Action:\n{description}"
                )
                # `SimulatedActor` owns `_llm`; keep best-effort if surface changes.
                llm = getattr(self, "_llm", None)
                if llm is None:
                    return
                planned = await llm.generate(prompt)  # type: ignore[no-any-return]
                planned = str(planned or "").strip()
                if not planned:
                    return
                # Only assign if the underlying handle supports it.
                if hasattr(handle, "_planned_result"):
                    setattr(handle, "_planned_result", planned)
            except Exception:
                return

        asyncio.create_task(_prebake())

        # Throttle notifications to keep UX readable.
        return _ThrottledHandle(handle, interval_s=self._notification_interval_s)
