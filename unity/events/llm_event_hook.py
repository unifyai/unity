"""Wire unillm LLM events to the Unity EventBus.

This module provides the hook function that converts unillm's LLMEvent
dataclass into Unity EventBus events, and the setup function to install
the hook during Unity initialization.

The hook is installed once during unity.init() and remains active for the
lifetime of the process.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unillm import LLMEvent

# Module-level flag to prevent double-registration
_HOOK_INSTALLED = False


def _llm_event_to_eventbus(event: "LLMEvent") -> None:
    """Convert a unillm LLMEvent to an EventBus event and publish it.

    This hook is called synchronously by unillm after each LLM call completes.
    We convert the event to our LLMPayload format and publish it asynchronously
    to avoid blocking the LLM call.

    The hook is designed to be resilient - any errors are silently ignored
    to ensure LLM calls are never disrupted by logging failures.
    """
    try:
        from .event_bus import EVENT_BUS, Event
        from .types.llm import LLMPayload

        # Pass through the simplified event data directly
        payload = LLMPayload(
            request=event.request,
            response=event.response,
            provider_cost=event.provider_cost,
            billed_cost=event.billed_cost,
        )
        llm_event = Event(type="LLM", payload=payload)

        # Publish asynchronously to avoid blocking the LLM call
        try:
            loop = asyncio.get_running_loop()
            # Fire-and-forget: schedule the publish but don't wait for it
            loop.create_task(EVENT_BUS.publish(llm_event))
        except RuntimeError:
            # No event loop running - skip publishing
            # This can happen during synchronous test teardown
            pass

    except Exception:
        # Never let hook failures break LLM calls
        pass


def install_llm_event_hook() -> None:
    """Install the LLM event hook to wire unillm events to EventBus.

    This function is idempotent - calling it multiple times has no effect
    after the first successful installation.

    Should be called during unity.init() after the EventBus is initialized.

    Uses set_global_llm_event_hook() to ensure the hook is process-wide and
    works across all threads. This is critical because unity.init() may be
    called from a worker thread (via asyncio.to_thread in managers_utils.py)
    while LLM calls happen from the main async context.
    """
    global _HOOK_INSTALLED

    if _HOOK_INSTALLED:
        return

    try:
        import unillm

        # Use global hook to ensure it works across all threads/contexts.
        # This is essential because unity.init() may run in a thread pool
        # worker while LLM calls happen from the main async context.
        unillm.set_global_llm_event_hook(_llm_event_to_eventbus)
        _HOOK_INSTALLED = True
    except ImportError:
        # unillm not available - skip hook installation
        pass
    except Exception:
        # Any other error - skip silently to not break initialization
        pass
