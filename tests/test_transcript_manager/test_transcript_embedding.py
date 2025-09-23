import pytest

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message, VALID_MEDIA
from tests.helpers import _handle_project
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
    tm = TranscriptManager()

    # Create a few test messages
    msgs = [
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=1,
            receiver_ids=[2],
            timestamp="2025-05-19 12:00:00",
            content="Can you help me with my banking questions? I'm looking to set up a new account.",
            exchange_id=1,
        ),
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=2,
            receiver_ids=[1],
            timestamp="2025-05-19 12:00:01",
            content="I'd be happy to help with your banking needs! What type of account would you like to set up? Checking, savings, or investment?",
            exchange_id=1,
        ),
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=1,
            receiver_ids=[2],
            timestamp="2025-05-19 12:00:02",
            content="I'm interested in learning about Python programming, especially data science applications.",
            exchange_id=1,
        ),
    ]
    [tm.log_messages(m) for m in msgs]
    tm.join_published()

    # Ensure that a lexical search for the word 'budgeting' returns no results
    lexical_results = tm._filter_messages(filter="'budgeting' in content")
    assert lexical_results == []

    # Use semantic search to find the nearest messages to the query
    nearest = tm._search_messages(references={"content": "banking and budgeting"}, k=2)

    # Verify the result length and type
    assert len(nearest) == 2
    assert all(isinstance(msg, Message) for msg in nearest)

    # the last message is totally unrelated to the query so should not be in the results
    assert msgs[-1].content not in set([n.content for n in nearest])

    # Test k-limit behavior
    all_nearest = tm._search_messages(
        references={"content": "banking and budgeting"},
        k=10,
    )
    assert len(all_nearest) == 3  # Should return all 3 messages we inserted
