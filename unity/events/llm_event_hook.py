"""Wire unillm LLM events to the Unity EventBus.

This module provides the hook function that converts unillm's LLMEvent
dataclass into Unity EventBus events, and the setup function to install
the hook during Unity initialization.

The hook is installed once during unity.init() and remains active for the
lifetime of the process.

Additionally, this module logs cumulative spending to the Assistants project
for monthly spending limit tracking. After each LLM call, the billed_cost is
atomically added to the cumulative_spend for the current month.
"""

from __future__ import annotations

import asyncio
import logging
import zoneinfo
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unillm import LLMEvent

logger = logging.getLogger(__name__)

# Module-level flag to prevent double-registration
_HOOK_INSTALLED = False


async def _update_cumulative_spend(billed_cost: float) -> None:
    """Update cumulative monthly spend after each LLM call.

    Costs are attributed to the user(s) specified by the COST_ATTRIBUTION
    ContextVar (set by ConversationManager / act tool).  When unset, falls
    back to the assistant's supervisor (SESSION_DETAILS.user.id).

    Each attributed user gets their own spending row keyed by
    (_user_id, _assistant_id, month).  For multi-user attribution the cost
    is split evenly.

    Parameters
    ----------
    billed_cost : float
        The cost of the LLM call to add to cumulative spend
    """
    from datetime import datetime

    from ..common.log_utils import atomic_upsert
    from ..session_details import SESSION_DETAILS
    from .cost_attribution import COST_ATTRIBUTION

    # Skip if no billed cost
    if not billed_cost or billed_cost <= 0:
        return

    # Get billing timezone from user's settings (fallback to UTC)
    user_tz_str = SESSION_DETAILS.assistant.timezone or "UTC"
    try:
        tz = zoneinfo.ZoneInfo(user_tz_str)
    except Exception:
        tz = zoneinfo.ZoneInfo("UTC")

    # Calculate current month in user's timezone
    month = datetime.now(tz).strftime("%Y-%m")

    user_ctx = SESSION_DETAILS.user_context
    assistant_ctx = SESSION_DETAILS.assistant_context
    assistant_id = SESSION_DETAILS.assistant.agent_id

    if not user_ctx or not assistant_ctx or not assistant_id:
        return

    context = f"{user_ctx}/{assistant_ctx}/Spending/Monthly"

    # Resolve attribution: per-user user_ids or fall back to supervisor
    user_ids = COST_ATTRIBUTION.get() or [SESSION_DETAILS.user.id]
    per_user_cost = billed_cost / len(user_ids)

    for uid in user_ids:
        try:
            cost_str = f"{per_user_cost:.10f}".rstrip("0").rstrip(".")
            await atomic_upsert(
                context=context,
                unique_keys={
                    "_user_id": "str",
                    "_assistant_id": "str",
                    "month": "str",
                },
                field="cumulative_spend",
                operation=f"+{cost_str}",
                initial_data={
                    "_assistant_id": str(assistant_id),
                    "month": month,
                },
                data_overrides={"_user_id": uid},
                add_to_all_context=True,
                project="Assistants",
            )
        except Exception as e:
            logger.debug(f"Failed to update cumulative spend for {uid}: {e}")


def _llm_event_to_eventbus(event: "LLMEvent") -> None:
    """Convert a unillm LLMEvent to an EventBus event and publish it.

    This hook is called synchronously by unillm after each LLM call completes.
    We convert the event to our LLMPayload format and publish it asynchronously
    to avoid blocking the LLM call.

    Additionally, this hook updates cumulative monthly spending for the assistant
    to support spending limit tracking.

    The hook is designed to be resilient - any errors are silently ignored
    to ensure LLM calls are never disrupted by logging failures.
    """
    try:
        from datetime import datetime, timezone

        from .event_bus import EVENT_BUS, Event
        from .types.llm import LLMPayload

        # Generate timestamp once for consistency between Event and derived columns
        ts = datetime.now(timezone.utc)

        # Pass through the simplified event data with derived time columns
        # for aggregation/grouping in usage analytics
        payload = LLMPayload(
            request=event.request,
            response=event.response,
            provider_cost=event.provider_cost,
            billed_cost=event.billed_cost,
            # Derived time columns for time-based aggregation
            # All columns use formats that Orchestra infers as date/datetime types
            time_minute=ts.replace(second=0, microsecond=0).isoformat(),
            time_hour=ts.replace(minute=0, second=0, microsecond=0).isoformat(),
            time_day=ts.strftime("%Y-%m-%d"),
            time_month=ts.replace(day=1).strftime("%Y-%m-%d"),  # First day of month
            time_year=ts.replace(month=1, day=1).strftime(
                "%Y-%m-%d",
            ),  # First day of year
        )
        llm_event = Event(type="LLM", payload=payload, timestamp=ts)

        # Publish asynchronously to avoid blocking the LLM call
        try:
            loop = asyncio.get_running_loop()
            # Fire-and-forget: schedule the publish but don't wait for it
            loop.create_task(EVENT_BUS.publish(llm_event))

            # Update cumulative spending for spending limit tracking
            if event.billed_cost and event.billed_cost > 0:
                loop.create_task(_update_cumulative_spend(event.billed_cost))
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
