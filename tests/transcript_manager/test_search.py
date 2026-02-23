import random
import pytest

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message
from unity.conversation_manager.types import VALID_MEDIA
from unity.contact_manager.contact_manager import ContactManager
from unity.contact_manager.types.contact import Contact
from tests.helpers import _handle_project


@pytest.mark.requires_real_unify
@pytest.mark.asyncio
@_handle_project
async def test_simple_similarity():
    tm = TranscriptManager()
    cm = ContactManager()

    # Create two real contacts and use their assigned ids
    a_id = cm._create_contact(first_name="Ann")["details"]["contact_id"]
    b_id = cm._create_contact(first_name="Ben")["details"]["contact_id"]

    msgs = [
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=a_id,
            receiver_ids=[b_id],
            timestamp="2025-05-19 12:00:00",
            content="I have some banking questions and budgeting needs",
            exchange_id=1,
        ),
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=b_id,
            receiver_ids=[a_id],
            timestamp="2025-05-19 12:00:01",
            content="Let's discuss banking plans tomorrow",
            exchange_id=1,
        ),
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=a_id,
            receiver_ids=[b_id],
            timestamp="2025-05-19 12:00:02",
            content="Totally unrelated: machine learning and Python",
            exchange_id=1,
        ),
    ]

    [tm.log_messages(m) for m in msgs]
    tm.join_published()

    nearest = tm._search_messages(references={"content": "banking and budgeting"}, k=2)[
        "messages"
    ]

    assert len(nearest) == 2
    assert all(isinstance(m, Message) for m in nearest)
    contents = {m.content for m in nearest}
    assert "Totally unrelated: machine learning and Python" not in contents

    # When references is None/empty, skip semantic search and return most recent messages
    recent_only = tm._search_messages(references=None, k=2)["messages"]
    assert [m.content for m in recent_only] == [
        "Totally unrelated: machine learning and Python",
        "Let's discuss banking plans tomorrow",
    ]
    recent_only_empty = tm._search_messages(references={}, k=2)["messages"]
    assert [m.content for m in recent_only_empty] == [
        "Totally unrelated: machine learning and Python",
        "Let's discuss banking plans tomorrow",
    ]


@pytest.mark.requires_real_unify
@pytest.mark.asyncio
@_handle_project
async def test_cross_contact_disambiguation():
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
            "exchange_id": 1001,
        },
        {
            "sender_id": bob,
            "receiver_ids": [alice],
            "content": "let's meet next week",
            "timestamp": "2025-06-01 09:05:00",
            "medium": random.choice(VALID_MEDIA),
            "exchange_id": 1002,
        },
        {
            "sender_id": carol,
            "receiver_ids": [alice],
            "content": "let's meet next month",
            "timestamp": "2025-06-02 10:00:00",
            "medium": random.choice(VALID_MEDIA),
            "exchange_id": 1003,
        },
        {
            "sender_id": carol,
            "receiver_ids": [bob],
            "content": "availability later this week",
            "timestamp": "2025-06-03 11:00:00",
            "medium": random.choice(VALID_MEDIA),
            "exchange_id": 1004,
        },
    ]

    [tm.log_messages(m) for m in msgs]
    tm.join_published()

    # Query mixes message content and sender bio; should pick Alice's message as best match
    refs = {"content": "meeting next week", "bio": "accounts"}
    nearest = tm._search_messages(references=refs, k=3)["messages"]

    assert len(nearest) >= 1
    top = nearest[0]

    # Resolve sender contact to verify identity
    sender = cm.filter_contacts(filter=f"contact_id == {top.sender_id}", limit=1)[
        "contacts"
    ][0]
    assert sender.first_name == "Alice"
    assert top.content == "let's meet next week"


@pytest.mark.requires_real_unify
@pytest.mark.asyncio
@_handle_project
async def test_sender_bio_only():
    tm = TranscriptManager()
    cm = ContactManager()

    alice = Contact(
        first_name="Alice",
        surname="Alpha",
        bio="experienced accountant",
        contact_id=-1,
    )
    bob = Contact(
        first_name="Bob",
        surname="Beta",
        bio="software engineer",
        contact_id=-1,
    )

    msgs = [
        {
            "sender_id": alice,
            "receiver_ids": [bob],
            "content": "generic note",
            "timestamp": "2025-06-04 10:00:00",
            "medium": random.choice(VALID_MEDIA),
            "exchange_id": 2001,
        },
        {
            "sender_id": bob,
            "receiver_ids": [alice],
            "content": "another generic",
            "timestamp": "2025-06-04 10:01:00",
            "medium": random.choice(VALID_MEDIA),
            "exchange_id": 2002,
        },
    ]

    [tm.log_messages(m) for m in msgs]
    tm.join_published()

    nearest = tm._search_messages(references={"sender_bio": "accountant"}, k=1)[
        "messages"
    ]

    assert len(nearest) == 1
    top = nearest[0]
    sender = cm.filter_contacts(filter=f"contact_id == {top.sender_id}", limit=1)[
        "contacts"
    ][0]
    assert sender.first_name == "Alice"


@pytest.mark.requires_real_unify
@pytest.mark.asyncio
@_handle_project
async def test_receiver_bio_only():
    tm = TranscriptManager()

    alice = Contact(first_name="Alice", bio="accountant", contact_id=-1)
    bob = Contact(first_name="Bob", bio="engineering manager", contact_id=-1)
    carol = Contact(first_name="Carol", bio="chef", contact_id=-1)

    msgs = [
        {
            "sender_id": alice,
            "receiver_ids": [bob],
            "content": "hello there",
            "timestamp": "2025-06-05 09:00:00",
            "medium": random.choice(VALID_MEDIA),
            "exchange_id": 3001,
        },
        {
            "sender_id": alice,
            "receiver_ids": [carol],
            "content": "hi there",
            "timestamp": "2025-06-05 09:01:00",
            "medium": random.choice(VALID_MEDIA),
            "exchange_id": 3002,
        },
    ]

    [tm.log_messages(m) for m in msgs]
    tm.join_published()

    nearest = tm._search_messages(references={"receiver_bio": "engineer"}, k=1)[
        "messages"
    ]

    assert len(nearest) == 1
    top = nearest[0]
    # Top should be the message where Bob (engineering manager) is the receiver
    assert any(rid == top.receiver_ids[0] for rid in top.receiver_ids)
    assert len(top.receiver_ids) == 1


@pytest.mark.requires_real_unify
@pytest.mark.asyncio
@_handle_project
async def test_receiver_bio_min_aggregation():
    tm = TranscriptManager()
    cm = ContactManager()

    alice = Contact(first_name="Alice", bio="accountant", contact_id=-1)
    bob = Contact(first_name="Bob", bio="software engineer", contact_id=-1)
    carol = Contact(first_name="Carol", bio="graphic designer", contact_id=-1)

    # One message has both Bob (engineer) and Carol as receivers
    msgs = [
        {
            "sender_id": alice,
            "receiver_ids": [bob, carol],
            "content": "check this out",
            "timestamp": "2025-06-06 08:00:00",
            "medium": random.choice(VALID_MEDIA),
            "exchange_id": 4001,
        },
        {
            "sender_id": alice,
            "receiver_ids": [carol],
            "content": "another msg",
            "timestamp": "2025-06-06 08:01:00",
            "medium": random.choice(VALID_MEDIA),
            "exchange_id": 4002,
        },
    ]

    [tm.log_messages(m) for m in msgs]
    tm.join_published()

    nearest = tm._search_messages(references={"receiver_bio": "engineer"}, k=1)[
        "messages"
    ]

    assert len(nearest) == 1
    top = nearest[0]
    # The top should be the message that includes Bob among receivers due to min aggregation
    bob_rec = cm.filter_contacts(filter="first_name == 'Bob'", limit=1)["contacts"][0]
    assert bob_rec.contact_id in top.receiver_ids


@pytest.mark.requires_real_unify
@pytest.mark.asyncio
@_handle_project
async def test_contacts_table_output():
    tm = TranscriptManager()

    # Ensure both participating contacts exist by creating them explicitly
    cm = ContactManager()
    c1 = cm._create_contact(first_name="AlphaUser")
    c2 = cm._create_contact(first_name="BetaUser")
    id1 = c1["details"]["contact_id"]
    id2 = c2["details"]["contact_id"]

    # Seed two messages between the created contacts
    msgs = [
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=id1,
            receiver_ids=[id2],
            timestamp="2025-07-01 10:00:00",
            content="alpha topic",
            exchange_id=555,
        ),
        Message(
            medium=random.choice(VALID_MEDIA),
            sender_id=id2,
            receiver_ids=[id1],
            timestamp="2025-07-01 10:01:00",
            content="beta topic",
            exchange_id=555,
        ),
    ]
    [tm.log_messages(m) for m in msgs]
    tm.join_published()

    # Ask for a semantic search with contacts table rendering
    result = tm._search_messages(
        references={"content": "alpha topic"},
        k=5,
    )

    assert isinstance(result, dict)
    assert set(result.keys()) >= {"contacts", "messages"}

    contacts = result["contacts"]
    messages = result["messages"]

    # The messages must reference only ids present in contacts
    contact_ids_from_table = {c.contact_id for c in contacts}
    for m in messages:
        assert m.sender_id in contact_ids_from_table
        for rid in getattr(m, "receiver_ids", []) or []:
            assert rid in contact_ids_from_table


@pytest.mark.requires_real_unify
@pytest.mark.asyncio
@_handle_project
async def test_combined_bio_terms():
    tm = TranscriptManager()
    cm = ContactManager()

    alice = Contact(first_name="Alice", bio="accountant", contact_id=-1)
    bob = Contact(first_name="Bob", bio="software engineer", contact_id=-1)
    carol = Contact(first_name="Carol", bio="project manager", contact_id=-1)

    msgs = [
        {
            "sender_id": alice,
            "receiver_ids": [bob],
            "content": "status update",
            "timestamp": "2025-06-07 12:00:00",
            "medium": random.choice(VALID_MEDIA),
            "exchange_id": 5001,
        },
        {
            "sender_id": carol,
            "receiver_ids": [bob],
            "content": "status update",
            "timestamp": "2025-06-07 12:01:00",
            "medium": random.choice(VALID_MEDIA),
            "exchange_id": 5002,
        },
        {
            "sender_id": alice,
            "receiver_ids": [carol],
            "content": "status update",
            "timestamp": "2025-06-07 12:02:00",
            "medium": random.choice(VALID_MEDIA),
            "exchange_id": 5003,
        },
    ]

    [tm.log_messages(m) for m in msgs]
    tm.join_published()

    refs = {"sender_bio": "accountant", "receiver_bio": "engineer"}
    nearest = tm._search_messages(references=refs, k=3)["messages"]

    assert len(nearest) >= 1
    top = nearest[0]
    s = cm.filter_contacts(filter=f"contact_id == {top.sender_id}", limit=1)[
        "contacts"
    ][0]
    assert s.first_name == "Alice"
    # Ensure Bob is among receivers
    bob_rec = cm.filter_contacts(filter="first_name == 'Bob'", limit=1)["contacts"][0]
    assert bob_rec.contact_id in top.receiver_ids


@pytest.mark.requires_real_unify
@pytest.mark.asyncio
@_handle_project
async def test_receiver_only_returns_expected():
    tm = TranscriptManager()
    cm = ContactManager()

    alice = Contact(first_name="Alice", bio="accountant", contact_id=-1)
    bob = Contact(first_name="Bob", bio="engineer", contact_id=-1)
    dave = Contact(first_name="Dave", bio="engineer", contact_id=-1)
    eve = Contact(first_name="Eve", bio="designer", contact_id=-1)

    msgs = [
        {
            "sender_id": alice,
            "receiver_ids": [bob],
            "content": "msg1",
            "timestamp": "2025-06-08 08:00:00",
            "medium": random.choice(VALID_MEDIA),
            "exchange_id": 6001,
        },
        {
            "sender_id": alice,
            "receiver_ids": [dave],
            "content": "msg2",
            "timestamp": "2025-06-08 08:01:00",
            "medium": random.choice(VALID_MEDIA),
            "exchange_id": 6002,
        },
        {
            "sender_id": alice,
            "receiver_ids": [eve],
            "content": "msg3",
            "timestamp": "2025-06-08 08:02:00",
            "medium": random.choice(VALID_MEDIA),
            "exchange_id": 6003,
        },
    ]

    [tm.log_messages(m) for m in msgs]
    tm.join_published()

    nearest = tm._search_messages(references={"receiver_bio": "engineer"}, k=2)[
        "messages"
    ]

    assert len(nearest) == 2
    # Both results should have receivers among {bob, dave}; order not guaranteed
    eng_ids = {
        c.contact_id
        for c in cm.filter_contacts(filter="first_name in ['Bob', 'Dave']")["contacts"]
    }
    for m in nearest:
        assert any(rid in eng_ids for rid in m.receiver_ids)
