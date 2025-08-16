import random
import pytest

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message, VALID_MEDIA
from unity.contact_manager.contact_manager import ContactManager
from unity.contact_manager.types.contact import Contact
from tests.helpers import _handle_project


@pytest.mark.unit
@pytest.mark.requires_real_unify
@pytest.mark.asyncio
@_handle_project
async def test_search_messages_simple_similarity():
    tm = TranscriptManager()

    msgs = [
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=1,
            receiver_ids=[2],
            timestamp="2025-05-19 12:00:00",
            content="I have some banking questions and budgeting needs",
            exchange_id=1,
        ),
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=2,
            receiver_ids=[1],
            timestamp="2025-05-19 12:00:01",
            content="Let's discuss banking plans tomorrow",
            exchange_id=1,
        ),
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=1,
            receiver_ids=[2],
            timestamp="2025-05-19 12:00:02",
            content="Totally unrelated: machine learning and Python",
            exchange_id=1,
        ),
    ]

    [tm.log_messages(m) for m in msgs]
    tm.join_published()

    nearest = tm._search_messages(references={"content": "banking and budgeting"}, k=2)

    assert len(nearest) == 2
    assert all(isinstance(m, Message) for m in nearest)
    contents = {m.content for m in nearest}
    assert "Totally unrelated: machine learning and Python" not in contents

    # When references is None/empty, skip semantic search and return most recent messages
    recent_only = tm._search_messages(references=None, k=2)
    assert [m.content for m in recent_only] == [
        "Totally unrelated: machine learning and Python",
        "Let's discuss banking plans tomorrow",
    ]
    recent_only_empty = tm._search_messages(references={}, k=2)
    assert [m.content for m in recent_only_empty] == [
        "Totally unrelated: machine learning and Python",
        "Let's discuss banking plans tomorrow",
    ]


@pytest.mark.unit
@pytest.mark.requires_real_unify
@pytest.mark.asyncio
@_handle_project
async def test_search_messages_cross_contact_and_message_disambiguation():
    tm = TranscriptManager()
    cm = ContactManager()

    # Create Contacts (auto-created via TranscriptManager when using Contact objects)
    alice = Contact(
        first_name="Alice",
        surname="A",
        bio="Senior accountant at Acme",
        contact_id=-1,
    )
    bob = Contact(
        first_name="Bob",
        surname="B",
        bio="Software engineer and manager",
        contact_id=-1,
    )
    carol = Contact(
        first_name="Carol",
        surname="C",
        bio="Junior accountant in training",
        contact_id=-1,
    )

    # Log messages – multiple people say the same phrase; only Alice is an accountant who said it
    msgs = [
        {
            "sender_id": alice,
            "receiver_ids": [bob],
            "content": "let's meet next week",
            "timestamp": "2025-06-01 09:00:00",
            "medium": random.choice(VALID_MEDIA),
        },
        {
            "sender_id": bob,
            "receiver_ids": [alice],
            "content": "let's meet next week",
            "timestamp": "2025-06-01 09:05:00",
            "medium": random.choice(VALID_MEDIA),
        },
        {
            "sender_id": carol,
            "receiver_ids": [alice],
            "content": "let's meet next month",
            "timestamp": "2025-06-02 10:00:00",
            "medium": random.choice(VALID_MEDIA),
        },
        {
            "sender_id": carol,
            "receiver_ids": [bob],
            "content": "availability later this week",
            "timestamp": "2025-06-03 11:00:00",
            "medium": random.choice(VALID_MEDIA),
        },
    ]

    [tm.log_messages(m) for m in msgs]
    tm.join_published()

    # Query mixes message content and sender bio; should pick Alice's message as best match
    refs = {"content": "let's meet next week", "bio": "account"}
    nearest = tm._search_messages(references=refs, k=3)

    assert len(nearest) >= 1
    top = nearest[0]

    # Resolve sender contact to verify identity
    sender = cm._filter_contacts(filter=f"contact_id == {top.sender_id}", limit=1)[0]
    assert sender.first_name == "Alice"
    assert top.content == "let's meet next week"
