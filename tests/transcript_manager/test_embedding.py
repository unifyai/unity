import pytest

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message
from unity.conversation_manager.cm_types import VALID_MEDIA
from tests.helpers import _handle_project
import random
from unity.contact_manager.contact_manager import ContactManager


@pytest.mark.requires_real_unify
@pytest.mark.asyncio
@_handle_project
async def test_embedding_semantic_search():
    """
    Test the transcript manager's ability to perform semantic search via nearest message retrieval.
    """
    # Create the TranscriptManager instance
    tm = TranscriptManager()
    cm = ContactManager()

    # Create two real contacts and use their assigned ids
    alice_id = cm._create_contact(first_name="Alice")["details"]["contact_id"]
    bob_id = cm._create_contact(first_name="Bob")["details"]["contact_id"]

    # Create a few test messages
    msgs = [
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=alice_id,
            receiver_ids=[bob_id],
            timestamp="2025-05-19 12:00:00",
            content="Can you help me with my banking questions? I'm looking to set up a new account.",
            exchange_id=1,
        ),
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=bob_id,
            receiver_ids=[alice_id],
            timestamp="2025-05-19 12:00:01",
            content="I'd be happy to help with your banking needs! What type of account would you like to set up? Checking, savings, or investment?",
            exchange_id=1,
        ),
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=alice_id,
            receiver_ids=[bob_id],
            timestamp="2025-05-19 12:00:02",
            content="I'm interested in learning about Python programming, especially data science applications.",
            exchange_id=1,
        ),
    ]
    [tm.log_messages(m) for m in msgs]
    tm.join_published()

    # Ensure that a lexical search for the word 'budgeting' returns no results
    # New behavior: empty results return {"messages": []}
    lexical_result = tm._filter_messages(filter="'budgeting' in content")
    assert lexical_result == {"messages": []}

    # Use semantic search to find the nearest messages to the query
    nearest = tm._search_messages(references={"content": "banking and budgeting"}, k=2)[
        "messages"
    ]

    # Verify the result length and type
    assert len(nearest) == 2
    assert all(isinstance(msg, Message) for msg in nearest)

    # the last message is totally unrelated to the query so should not be in the results
    assert msgs[-1].content not in set([n.content for n in nearest])

    # Test k-limit behavior
    all_nearest = tm._search_messages(
        references={"content": "banking and budgeting"},
        k=10,
    )["messages"]
    assert len(all_nearest) == 3  # Should return all 3 messages we inserted
