from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from unity.contact_manager.simulated import SimulatedContactManager
from unity.function_manager.primitives import Primitives
from unity.task_scheduler.simulated import SimulatedTaskScheduler
from unity.transcript_manager.simulated import SimulatedTranscriptManager

# ---------------------------------------------------------------------------
# Shared timeouts (standardize across steerability tests)
# ---------------------------------------------------------------------------

HANDLE_REGISTRATION_TIMEOUT = 60.0
STEERING_EVENT_TIMEOUT = 60.0
CLARIFICATION_TIMEOUT = 60.0
PLAN_COMPLETION_TIMEOUT = 180.0


class _ClarificationForcingMixin:
    """Mixin that forces simulated clarification flow for matching user queries.

    This is test-only infrastructure: it deterministically enables clarification
    queues for specific natural-language prompts, without requiring the Actor's
    generated plan to pass `_requests_clarification=True` explicitly.
    """

    def __init__(
        self,
        *args: Any,
        clarification_triggers: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        self._clarification_triggers = [
            t.lower() for t in (clarification_triggers or [])
        ]

    def _should_force_clarification(self, text: str) -> bool:
        if not self._clarification_triggers:
            return False
        hay = text.lower()
        return any(t in hay for t in self._clarification_triggers)

    def _maybe_inject_clarification_kwargs(
        self,
        text: str,
        kwargs: dict[str, Any],
    ) -> None:
        # Respect explicit caller intent (and avoid overriding provided queues).
        if kwargs.get("_requests_clarification"):
            return
        if not self._should_force_clarification(text):
            return
        kwargs["_requests_clarification"] = True
        kwargs["_clarification_up_q"] = asyncio.Queue()
        kwargs["_clarification_down_q"] = asyncio.Queue()


class ClarificationForcingContactManager(
    _ClarificationForcingMixin,
    SimulatedContactManager,
):
    async def ask(self, text: str, **kwargs: Any):  # noqa: ANN001
        kwargs = dict(kwargs)
        self._maybe_inject_clarification_kwargs(text, kwargs)
        return await super().ask(text, **kwargs)

    async def update(self, text: str, **kwargs: Any):  # noqa: ANN001
        kwargs = dict(kwargs)
        self._maybe_inject_clarification_kwargs(text, kwargs)
        return await super().update(text, **kwargs)


class ClarificationForcingTranscriptManager(
    _ClarificationForcingMixin,
    SimulatedTranscriptManager,
):
    async def ask(self, text: str, **kwargs: Any):  # noqa: ANN001
        kwargs = dict(kwargs)
        self._maybe_inject_clarification_kwargs(text, kwargs)
        return await super().ask(text, **kwargs)


class ClarificationForcingTaskScheduler(
    _ClarificationForcingMixin,
    SimulatedTaskScheduler,
):
    async def ask(self, text: str, **kwargs: Any):  # noqa: ANN001
        kwargs = dict(kwargs)
        self._maybe_inject_clarification_kwargs(text, kwargs)
        return await super().ask(text, **kwargs)

    async def update(self, text: str, **kwargs: Any):  # noqa: ANN001
        kwargs = dict(kwargs)
        self._maybe_inject_clarification_kwargs(text, kwargs)
        return await super().update(text, **kwargs)


@pytest.fixture
def create_primitives_with_clarification_forcing() -> Callable[..., Primitives]:
    """Factory to build `Primitives` with clarification-forcing simulated managers."""

    def _factory(
        *,
        contact_desc: str,
        contact_clarification_triggers: list[str] | None = None,
        transcript_desc: str | None = None,
        transcript_clarification_triggers: list[str] | None = None,
        task_desc: str | None = None,
        task_clarification_triggers: list[str] | None = None,
    ) -> Primitives:
        primitives = Primitives()

        primitives._contacts = ClarificationForcingContactManager(  # type: ignore[attr-defined]
            description=contact_desc,
            clarification_triggers=contact_clarification_triggers or [],
        )
        if transcript_desc is not None:
            primitives._transcripts = ClarificationForcingTranscriptManager(  # type: ignore[attr-defined]
                description=transcript_desc,
                clarification_triggers=transcript_clarification_triggers or [],
            )
        if task_desc is not None:
            primitives._tasks = ClarificationForcingTaskScheduler(  # type: ignore[attr-defined]
                description=task_desc,
                clarification_triggers=task_clarification_triggers or [],
            )

        return primitives

    return _factory


# ---------------------------------------------------------------------------
# Actor factory (optional helper for future tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def create_actor_with_primitives():
    """Async factory to create a `HierarchicalActor` wired to a given `Primitives`.

    This keeps test setup consistent and ensures the actor is always closed.
    """

    import contextlib
    from contextlib import asynccontextmanager

    from unity.actor.environments import StateManagerEnvironment
    from unity.actor.hierarchical_actor import HierarchicalActor

    @asynccontextmanager
    async def _factory(primitives: Primitives):
        actor = HierarchicalActor(
            headless=True,
            browser_mode="legacy",
            connect_now=False,
            environments=[StateManagerEnvironment(primitives)],
        )
        try:
            yield actor
        finally:
            with contextlib.suppress(Exception):
                await actor.close()

    return _factory
