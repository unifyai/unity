import pytest
from datetime import datetime, UTC

from unity.communication.transcript_manager.transcript_manager import TranscriptManager
from unity.communication.types.message import Message, VALID_MEDIA
from tests.helpers import _handle_project
from unity.events.event_bus import EventBus, Event
import random


@pytest.mark.unit
@pytest.mark.requires_real_unify
@pytest.mark.asyncio
@_handle_project
async def test_transcript_embedding_semantic_search():
    """
    Test the transcript manager's ability to perform semantic search via nearest message retrieval.
    """
    # Create the TranscriptManager instance
    tm = TranscriptManager(EventBus())

    # Create a few test messages
    msgs = [
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=1,
            receiver_id=2,
            timestamp="2025-05-19 12:00:00",
            content="Can you help me with my banking questions? I'm looking to set up a new account.",
            exchange_id=1,
        ),
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=2,
            receiver_id=1,
            timestamp="2025-05-19 12:00:01",
            content="I'd be happy to help with your banking needs! What type of account would you like to set up? Checking, savings, or investment?",
            exchange_id=1,
        ),
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=1,
            receiver_id=2,
            timestamp="2025-05-19 12:00:02",
            content="I'm interested in learning about Python programming, especially data science applications.",
            exchange_id=1,
        ),
    ]

    event_bus = EventBus()
    [
        await event_bus.publish(
            Event(
                type="Messages",
                timestamp=datetime.now(UTC).isoformat(),
                payload=msg,
            ),
        )
        for i, msg in enumerate(msgs)
    ]
    event_bus.join_published()

    # Ensure that a lexical search for the word 'budgeting' returns no results
    lexical_results = tm._search_messages(filter="'budgeting' in content")
    assert lexical_results == []

    # Use semantic search to find the nearest messages to the query
    nearest = tm._nearest_messages(text="banking and budgeting", k=2)

    # Verify the result length and type
    assert len(nearest) == 2
    assert all(isinstance(msg, Message) for msg in nearest)

    # Verify that the messages are returned in ascending order of distance
    assert nearest[0].content == msgs[-1].content
    assert nearest[1].content == msgs[-2].content

    # Test k-limit behavior
    all_nearest = tm._nearest_messages(text="banking and budgeting", k=10)
    assert len(all_nearest) == 3  # Should return all 3 messages we inserted
