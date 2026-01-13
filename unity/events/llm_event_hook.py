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

    This hook is called synchronously by unillm for each LLM request and
    response. We convert the event to our LLMPayload format and publish
    it asynchronously to avoid blocking the LLM call.

    The hook is designed to be resilient - any errors are silently ignored
    to ensure LLM calls are never disrupted by logging failures.
    """
    try:
        from .event_bus import EVENT_BUS, Event
        from .types.llm import LLMPayload

        # Extract message and tool counts from request kwargs
        request_kw = event.request_kw or {}
        messages = request_kw.get("messages", [])
        tools = request_kw.get("tools", [])

        # Build the payload with request info
        payload_data = {
            "phase": event.phase,
            "endpoint": event.endpoint,
            "model": event.model,
            "provider": event.provider,
            "stream": event.stream,
            "messages_count": len(messages) if isinstance(messages, list) else 0,
            "tools_count": len(tools) if isinstance(tools, list) else 0,
        }

        # Add response-specific fields
        if event.phase == "response":
            payload_data["cache_status"] = event.cache_status

            # Extract error message if present
            if event.error is not None:
                payload_data["error"] = str(event.error)

            # Extract response metadata if available
            if event.response is not None:
                # Response model
                if hasattr(event.response, "model"):
                    payload_data["response_model"] = event.response.model

                # Token usage
                if hasattr(event.response, "usage") and event.response.usage:
                    usage = event.response.usage
                    if hasattr(usage, "prompt_tokens"):
                        payload_data["prompt_tokens"] = usage.prompt_tokens
                    if hasattr(usage, "completion_tokens"):
                        payload_data["completion_tokens"] = usage.completion_tokens
                    if hasattr(usage, "total_tokens"):
                        payload_data["total_tokens"] = usage.total_tokens

                # Content preview (truncated for storage efficiency)
                try:
                    choices = getattr(event.response, "choices", None)
                    if choices and len(choices) > 0:
                        message = getattr(choices[0], "message", None)
                        if message:
                            content = getattr(message, "content", None)
                            if content:
                                preview = (
                                    content[:200] + "..."
                                    if len(content) > 200
                                    else content
                                )
                                payload_data["content_preview"] = preview
                except Exception:
                    pass

        payload = LLMPayload(**payload_data)
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
    """
    global _HOOK_INSTALLED

    if _HOOK_INSTALLED:
        return

    try:
        import unillm

        unillm.set_llm_event_hook(_llm_event_to_eventbus)
        _HOOK_INSTALLED = True
    except ImportError:
        # unillm not available - skip hook installation
        pass
    except Exception:
        # Any other error - skip silently to not break initialization
        pass
