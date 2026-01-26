"""
Test driver for ConversationManager stepping.

Provides deterministic event-by-event testing without relying on background
tasks or the async event loop. This keeps test-only code out of production
modules while providing a clean stepping API.

Usage:
    driver = CMStepDriver(conversation_manager)
    result = await driver.step(some_event)
    assert driver.cm.mode == "call"  # access underlying CM state
"""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass
from typing import TYPE_CHECKING

from unity.conversation_manager.domains.event_handlers import EventHandler
from unity.conversation_manager.events import (
    Event,
    SMSSent,
    EmailSent,
    UnifyMessageSent,
    PhoneCallSent,
    ActorHandleStarted,
)

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager


@dataclass(frozen=True)
class StepResult:
    """Result of processing a single event step."""

    input_event: Event
    llm_requested: bool
    llm_ran: bool
    output_events: list[Event]
    llm_step_count: int = 0  # Number of LLM thinking steps taken


# Context variable to track LLM run requests during stepping
_step_llm_requests: contextvars.ContextVar[list[tuple[float, bool]] | None] = (
    contextvars.ContextVar("_step_llm_requests", default=None)
)


class CMStepDriver:
    """
    Test driver for stepping through ConversationManager events.

    Provides deterministic event-by-event testing by:
    - Recording LLM run requests during event handling
    - Running the LLM immediately (instead of via debouncer)
    - Capturing and applying published output events to local state

    Attributes are proxied to the underlying ConversationManager, so you can
    access cm.mode, cm.contact_index, etc. directly through the driver.
    """

    def __init__(self, cm: "ConversationManager"):
        self._cm = cm
        # Track all tool calls made across all steps for test assertions
        self.all_tool_calls: list[str] = []

    @property
    def cm(self) -> "ConversationManager":
        """Access the underlying ConversationManager."""
        return self._cm

    def __getattr__(self, name: str):
        """Proxy attribute access to the underlying ConversationManager."""
        return getattr(self._cm, name)

    async def step(self, event: Event, *, publish: bool = False) -> StepResult:
        """Process one event deterministically and return produced output events.

        This bypasses the normal async event-driven flow by:
        - Recording any requested LLM runs during event handling
        - Running the LLM immediately (if requested)
        - Capturing and applying any published output events to local state

        Args:
            event: Input event to process.
            publish: Whether to forward published events to the broker.

        Returns:
            StepResult with output events produced during this step.
        """
        published_events: list[Event] = []
        output_events: list[Event] = []
        llm_requested = False
        llm_ran = False

        original_publish = self._cm.event_broker.publish

        async def publish_wrapper(channel: str, message: str) -> int:
            try:
                evt = Event.from_json(message)
            except Exception:
                evt = None
            if evt is not None:
                published_events.append(evt)
            if publish:
                return await original_publish(channel, message)
            return 0

        step_requests: list[tuple[float, bool]] = []
        token = _step_llm_requests.set(step_requests)

        # Patch request_llm_run to use our context var
        original_request = self._cm.request_llm_run

        async def patched_request(delay=0, cancel_running=False) -> None:
            requests = _step_llm_requests.get()
            if requests is not None:
                requests.append((delay, cancel_running))
                return
            # Fall back to normal behavior if not in step context
            self._cm._pending_llm_requests.append((delay, cancel_running))

        try:
            self._cm.event_broker.publish = publish_wrapper
            self._cm.request_llm_run = patched_request

            await EventHandler.handle_event(
                event,
                self._cm,
                is_voice_call=self._cm.call_manager.uses_realtime_api,
            )

            llm_requested = bool(step_requests)
            step_requests.clear()

            if llm_requested:
                llm_ran = True
                tool_name = await self._cm._run_llm()
                # Track tool call for test assertions
                if tool_name:
                    self.all_tool_calls.append(tool_name)

            # Apply any published events to local state so callers can inspect state
            # without depending on background broker subscribers.
            for evt in published_events:
                if isinstance(
                    evt,
                    (
                        SMSSent,
                        EmailSent,
                        UnifyMessageSent,
                        PhoneCallSent,
                        ActorHandleStarted,
                    ),
                ):
                    output_events.append(evt)
                await EventHandler.handle_event(
                    evt,
                    self._cm,
                    is_voice_call=self._cm.call_manager.uses_realtime_api,
                )
        finally:
            self._cm.event_broker.publish = original_publish
            self._cm.request_llm_run = original_request
            _step_llm_requests.reset(token)

        return StepResult(
            input_event=event,
            llm_requested=llm_requested,
            llm_ran=llm_ran,
            output_events=output_events,
        )

    async def step_until_wait(
        self,
        event: Event,
        *,
        max_steps: int = 5,
        publish: bool = False,
    ) -> StepResult:
        """Process an event and keep running LLM until it calls 'wait'.

        This gives the LLM continuous control until it explicitly decides to
        stop by calling the 'wait' tool.

        Args:
            event: Input event to process.
            max_steps: Maximum LLM steps to prevent infinite loops (default 5).
            publish: Whether to forward published events to the broker.

        Returns:
            StepResult with all output events produced across all steps.
        """
        all_output_events: list[Event] = []
        llm_ran = False

        original_publish = self._cm.event_broker.publish

        async def publish_wrapper(channel: str, message: str) -> int:
            try:
                evt = Event.from_json(message)
            except Exception:
                evt = None
            if evt is not None:
                if isinstance(
                    evt,
                    (
                        SMSSent,
                        EmailSent,
                        UnifyMessageSent,
                        PhoneCallSent,
                        ActorHandleStarted,
                    ),
                ):
                    all_output_events.append(evt)
                # Handle the event locally
                await EventHandler.handle_event(
                    evt,
                    self._cm,
                    is_voice_call=self._cm.call_manager.uses_realtime_api,
                )
            if publish:
                return await original_publish(channel, message)
            return 0

        step_requests: list[tuple[float, bool]] = []
        token = _step_llm_requests.set(step_requests)

        # Patch request_llm_run to use our context var
        original_request = self._cm.request_llm_run

        async def patched_request(delay=0, cancel_running=False) -> None:
            requests = _step_llm_requests.get()
            if requests is not None:
                requests.append((delay, cancel_running))
                return
            self._cm._pending_llm_requests.append((delay, cancel_running))

        try:
            self._cm.event_broker.publish = publish_wrapper
            self._cm.request_llm_run = patched_request

            # First, handle the incoming event
            await EventHandler.handle_event(
                event,
                self._cm,
                is_voice_call=self._cm.call_manager.uses_realtime_api,
            )

            llm_requested = bool(step_requests)
            step_requests.clear()

            # Run LLM in a loop until 'wait' is called or max_steps reached
            step_count = 0
            while llm_requested and step_count < max_steps:
                llm_ran = True
                tool_name = await self._cm._run_llm()
                step_count += 1

                # Track tool call for test assertions
                if tool_name:
                    self.all_tool_calls.append(tool_name)

                # Yield control to allow any background tasks (e.g., async ask
                # completions) to run and potentially emit events
                await asyncio.sleep(0)

                # Check if another LLM run was requested (e.g., by event handlers
                # processing events from background tasks)
                llm_requested = bool(step_requests)
                step_requests.clear()

                # Stop if 'wait' was called AND no new requests came in
                if tool_name == "wait" and not llm_requested:
                    break

                # If no explicit request but we didn't call 'wait', continue
                if not llm_requested and tool_name != "wait":
                    llm_requested = True

        finally:
            self._cm.event_broker.publish = original_publish
            self._cm.request_llm_run = original_request
            _step_llm_requests.reset(token)

        return StepResult(
            input_event=event,
            llm_requested=True,
            llm_ran=llm_ran,
            output_events=all_output_events,
            llm_step_count=step_count,
        )
