import datetime as dt
import pytest
import random
from collections import deque
from unittest.mock import patch, MagicMock

from unity.events.event_bus import EventBus, Event
from unity.transcript_manager.types.message import Message
from unity.conversation_manager.cm_types import Medium
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_cache_only_skips_backend():
    """When cache has enough events for a type, backend should not be called for that type."""
    window = 5
    bus = EventBus()
    bus.set_window("Message", window)

    # Start from a known clean state for this type
    bus._deques.setdefault(
        "Message",
        bus._deques.get("Message", deque(maxlen=window)),
    ).clear()

    # Publish events to fill the cache
    base_ts = dt.datetime.now(dt.UTC)
    for i in range(window):
        evt = Event(
            type="Message",
            timestamp=base_ts + dt.timedelta(seconds=i),
            payload=Message(
                medium=random.choice(list(Medium)),
                sender_id=random.randint(0, 10),
                receiver_ids=[random.randint(0, 10)],
                timestamp=dt.datetime.now(dt.UTC).isoformat(),
                content=f"{i}",
                exchange_id=0,
            ),
        )
        await bus.publish(evt, blocking=True)

    # Use a spy to track get_logs calls
    original_get_logs = __import__("unify").get_logs
    spy = MagicMock(side_effect=original_get_logs)

    with patch("unify.get_logs", spy):
        # Search for fewer events than cache holds
        results = await bus.search(filter="type == 'Message'", limit=3)

    # Verify we got the expected results from cache
    assert len(results) == 3
    contents = [r.payload.get("content") for r in results]
    assert contents == ["4", "3", "2"]

    # Verify no backend call was made for Message type specifically
    # The filter arg starts with 'type == "Message"' when fetching Message events
    # (Other types like ToolLoop/Comms may be fetched but won't match our filter)
    message_fetch_calls = [
        call
        for call in spy.call_args_list
        if call.kwargs.get("filter", "").startswith('type == "Message"')
    ]
    assert (
        len(message_fetch_calls) == 0
    ), f"Backend was called for Message type when cache should have been sufficient: {message_fetch_calls}"
