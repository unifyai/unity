import time
import random
import pytest
from datetime import datetime, UTC
import unify

from unity.transcript_manager.types.message import Message
from unity.conversation_manager.types import VALID_MEDIA
from unity.transcript_manager.transcript_manager import TranscriptManager
from tests.helpers import _handle_project
from unity.contact_manager.contact_manager import ContactManager

CONTACTS = [
    {
        "contact_id": 0,
        "first_name": "John",
        "surname": "Smith",
        "email_address": "johnsmith11@gmail.com",
        "phone_number": "+1234567890",
    },
    {
        "contact_id": 1,
        "first_name": "Nancy",
        "surname": "Gray",
        "email_address": "nancy_gray@outlook.com",
        "phone_number": "+1987654320",
    },
]

MESSAGES = [
    "Hello, how are you?",
    "Sorry I couldn't hear you",
    "Hell no, I won't do that",
    "Wow, did you see that?",
    "Goodbye",
]


@pytest.mark.asyncio
@_handle_project
async def test_log_messages():
    tm = TranscriptManager()
    [
        tm.log_messages(
            Message(
                medium=random.choice(VALID_MEDIA),
                sender_id=random.randint(0, 2),
                receiver_ids=[random.randint(0, 2)],
                timestamp=datetime.now(UTC),
                content=random.choice(MESSAGES),
                exchange_id=i,
            ),
        )
        for i in range(10)
    ]
    tm.join_published()


@pytest.mark.asyncio
@_handle_project
async def test_get_messages():
    start_time = datetime.now(UTC).isoformat()
    time.sleep(0.1)
    tm = TranscriptManager()

    # Hard-coded messages for deterministic testing
    test_messages = [
        Message(
            medium="email",
            sender_id=0,
            receiver_ids=[1],
            timestamp=datetime.now(UTC),
            content="Hello, how are you?",
            exchange_id=0,
        ),
        Message(
            medium="sms_message",
            sender_id=1,
            receiver_ids=[0],
            timestamp=datetime.now(UTC),
            content="Sorry I couldn't hear you",
            exchange_id=1,
        ),
        Message(
            medium="sms_message",
            sender_id=0,
            receiver_ids=[1],
            timestamp=datetime.now(UTC),
            content="Hell no, I won't do that",
            exchange_id=2,
        ),
        Message(
            medium="email",
            sender_id=1,
            receiver_ids=[0],
            timestamp=datetime.now(UTC),
            content="Wow, did you see that?",
            exchange_id=3,
        ),
        Message(
            medium="sms_message",
            sender_id=0,
            receiver_ids=[1],
            timestamp=datetime.now(UTC),
            content="Goodbye",
            exchange_id=4,
        ),
        Message(
            medium="sms_message",
            sender_id=1,
            receiver_ids=[0],
            timestamp=datetime.now(UTC),
            content="Hell no, I won't do that",
            exchange_id=5,
        ),
        Message(
            medium="email",
            sender_id=0,
            receiver_ids=[1],
            timestamp=datetime.now(UTC),
            content="Sorry I couldn't hear you",
            exchange_id=6,
        ),
        Message(
            medium="sms_message",
            sender_id=1,
            receiver_ids=[0],
            timestamp=datetime.now(UTC),
            content="Wow, did you see that?",
            exchange_id=7,
        ),
        Message(
            medium="sms_message",
            sender_id=0,
            receiver_ids=[1],
            timestamp=datetime.now(UTC),
            content="Hell no, I won't do that",
            exchange_id=8,
        ),
        Message(
            medium="email",
            sender_id=1,
            receiver_ids=[0],
            timestamp=datetime.now(UTC),
            content="Hello, how are you?",
            exchange_id=9,
        ),
    ]

    for msg in test_messages:
        tm.log_messages(msg)
    tm.join_published()

    ## get all

    result = tm._filter_messages()
    messages = result["messages"]
    assert len(messages) == 10
    assert all(isinstance(msg, Message) for msg in messages)

    ## search

    # sender
    messages = tm._filter_messages(filter="sender_id == 0")["messages"]
    assert len(messages) == 5
    assert all(isinstance(msg, Message) for msg in messages)

    # contains
    messages = tm._filter_messages(filter="'Hell' in content")["messages"]
    assert len(messages) == 5
    assert all(isinstance(msg, Message) for msg in messages)

    # does not contain
    messages = tm._filter_messages(filter="',' not in content")["messages"]
    assert len(messages) == 3
    assert all(isinstance(msg, Message) for msg in messages)

    # medium
    messages = tm._filter_messages(
        filter="medium in ('email', 'sms_message')",
    )["messages"]
    assert len(messages) == 10  # All 10 messages are either 'email' or 'sms_message'
    assert all(isinstance(msg, Message) for msg in messages)

    # timestamp
    messages = tm._filter_messages(filter=f"timestamp < '{start_time}'").get(
        "messages",
        [],
    )
    assert len(messages) == 0
    messages = tm._filter_messages(filter=f"timestamp > '{start_time}'")["messages"]
    assert len(messages) == 10


# ────────────────────────────────────────────────────────────────────────────
# 6.  Multiple receiver handling                                             #
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_multiple_receivers():
    """Ensure a single message can target multiple distinct receiver IDs."""
    tm = TranscriptManager()

    # Ensure contact id 2 exists so participant contacts are resolvable
    cm = ContactManager()
    cm._create_contact(first_name="TempUser")

    msg = Message(
        medium="email",
        sender_id=0,
        receiver_ids=[1, 2],
        timestamp=datetime.now(UTC),
        content="Quarterly results and next-steps.",
        exchange_id=4242,
    )

    # Log and flush to storage
    tm.log_messages(msg)
    tm.join_published()

    # Retrieve the message back – simplest: list everything for this exchange
    found = tm._filter_messages(filter="exchange_id == 4242")["messages"]
    assert (
        len(found) == 1
    ), "Exactly one message should have been logged for exchange 4242"

    m = found[0]
    # Primary assertions ----------------------------------------------------
    assert m.receiver_ids == [1, 2], "receiver_ids should preserve the full list"


@pytest.mark.asyncio
@_handle_project
async def test_filter_messages_contacts_table_output():
    tm = TranscriptManager()

    # Seed a few deterministic messages
    msgs = [
        Message(
            medium="email",
            sender_id=0,
            receiver_ids=[1],
            timestamp=datetime.now(UTC),
            content="hello A",
            exchange_id=111,
        ),
        Message(
            medium="email",
            sender_id=1,
            receiver_ids=[0],
            timestamp=datetime.now(UTC),
            content="hello B",
            exchange_id=111,
        ),
    ]
    [tm.log_messages(m) for m in msgs]
    tm.join_published()

    # Default now includes contacts and messages
    result = tm._filter_messages(
        filter="exchange_id == 111",
    )

    assert isinstance(result, dict)
    assert set(result.keys()) >= {"contacts", "messages"}

    contacts = result["contacts"]
    messages = result["messages"]

    # Validate uniqueness and coverage of contacts
    contact_ids_from_table = {
        (c.get("contact_id") if isinstance(c, dict) else getattr(c, "contact_id", None))
        for c in contacts
    }
    assert len(contact_ids_from_table) == len(contacts)

    referenced_ids = set()
    for m in messages:
        if getattr(m, "sender_id", None) is not None:
            referenced_ids.add(m.sender_id)
        for rid in getattr(m, "receiver_ids", []) or []:
            referenced_ids.add(rid)

    assert referenced_ids.issubset(
        contact_ids_from_table,
    ), "All participant ids must be included in contacts table"


@_handle_project
def test_clear():
    tm = TranscriptManager()

    # Seed a couple of messages (distinct exchange_ids)
    tm.log_messages(
        Message(
            medium="email",
            sender_id=0,
            receiver_ids=[1],
            timestamp=datetime.now(UTC),
            content="Alpha",
            exchange_id=1001,
        ),
    )
    tm.log_messages(
        Message(
            medium="sms_message",
            sender_id=1,
            receiver_ids=[0],
            timestamp=datetime.now(UTC),
            content="Beta",
            exchange_id=1002,
        ),
    )
    tm.join_published()

    # Sanity: messages exist before clear
    pre = tm._filter_messages()["messages"]
    assert len(pre) >= 2

    # Execute clear
    tm.clear()

    # After clear: context should exist again and exchange_id present
    fields_exchanges = unify.get_fields(context=tm._exchanges_ctx)
    assert "exchange_id" in fields_exchanges

    # Prior messages should be gone
    post = tm._filter_messages().get("messages", [])
    assert len(post) == 0
