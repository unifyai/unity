import time
import json
import random
import pytest
from datetime import datetime, UTC
import unify

from unity.transcript_manager.types.message import Message, VALID_MEDIA
from unity.transcript_manager.transcript_manager import TranscriptManager
from tests.helpers import _handle_project

CONTACTS = [
    {
        "contact_id": 0,
        "first_name": "John",
        "surname": "Smith",
        "email_address": "johnsmith11@gmail.com",
        "phone_number": "+1234567890",
        "whatsapp_number": "+1234567890",
    },
    {
        "contact_id": 1,
        "first_name": "Nancy",
        "surname": "Gray",
        "email_address": "nancy_gray@outlook.com",
        "phone_number": "+1987654320",
        "whatsapp_number": "+1987654320",
    },
]

MESSAGES = [
    "Hello, how are you?",
    "Sorry I couldn't hear you",
    "Hell no, I won't do that",
    "Wow, did you see that?",
    "Goodbye",
]


@pytest.mark.unit
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


@pytest.mark.unit
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
            medium="whatsapp_message",
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
            medium="whatsapp_message",
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
            medium="whatsapp_message",
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

    messages = tm._filter_messages()
    assert len(messages) == 10
    assert all(isinstance(msg, Message) for msg in messages)

    ## search

    # sender
    messages = tm._filter_messages(filter="sender_id == 0")
    assert len(messages) == 5
    assert all(isinstance(msg, Message) for msg in messages)

    # contains
    messages = tm._filter_messages(filter="'Hell' in content")
    assert len(messages) == 5
    assert all(isinstance(msg, Message) for msg in messages)

    # does not contain
    messages = tm._filter_messages(filter="',' not in content")
    assert len(messages) == 3
    assert all(isinstance(msg, Message) for msg in messages)

    # medium
    messages = tm._filter_messages(
        filter="medium in ('email', 'whatsapp_message')",
    )
    assert len(messages) == 7
    assert all(isinstance(msg, Message) for msg in messages)

    # timestamp
    messages = tm._filter_messages(filter=f"timestamp < '{start_time}'")
    assert len(messages) == 0
    messages = tm._filter_messages(filter=f"timestamp > '{start_time}'")
    assert len(messages) == 10


# ────────────────────────────────────────────────────────────────────────────
# 6.  Multiple receiver handling                                             #
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_multiple_receivers():
    """Ensure a single message can target multiple distinct receiver IDs."""
    tm = TranscriptManager()

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
    found = tm._filter_messages(filter="exchange_id == 4242")
    assert (
        len(found) == 1
    ), "Exactly one message should have been logged for exchange 4242"

    m = found[0]
    # Primary assertions ----------------------------------------------------
    assert m.receiver_ids == [1, 2], "receiver_ids should preserve the full list"


@pytest.mark.unit
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

    # Call with contacts table rendering enabled
    result = tm._filter_messages(
        filter="exchange_id == 111",
        return_with_contacts_table=True,
    )

    assert (
        isinstance(result, str)
        and "Contacts (once):" in result
        and "\n\nMessages:\n" in result
    )

    # Parse the two JSON blobs
    head, tail = result.split("\n\nMessages:\n", 1)
    contacts_json = head.split("Contacts (once):\n", 1)[1]
    contacts = json.loads(contacts_json)
    messages = json.loads(tail)

    # Validate uniqueness and coverage of contacts
    contact_ids_from_table = {c.get("contact_id") for c in contacts}
    assert len(contact_ids_from_table) == len(contacts)

    referenced_ids = set()
    for m in messages:
        if m.get("sender_id") is not None:
            referenced_ids.add(m["sender_id"])
        for rid in m.get("receiver_ids", []) or []:
            referenced_ids.add(rid)

    assert referenced_ids.issubset(
        contact_ids_from_table,
    ), "All participant ids must be included in contacts table"


@pytest.mark.unit
@_handle_project
def test_metadata_private_column_roundtrip():
    tm = TranscriptManager()

    unique_exchange = 8642001
    meta = {"foo": "bar", "n": 1}

    tm.log_messages(
        {
            "medium": "email",
            "sender_id": 0,
            "receiver_ids": [1],
            "timestamp": datetime.now(UTC),
            "content": "Metadata test message",
            "exchange_id": unique_exchange,
            "_metadata": meta,
        },
    )
    tm.join_published()

    # 1) Column exists in the Transcripts context (private column with leading underscore)
    fields = unify.get_fields(context=tm._transcripts_ctx)
    assert "_metadata" in fields, "_metadata column should exist in Transcripts"

    # 2) Raw log entry contains the metadata payload
    rows = unify.get_logs(
        context=tm._transcripts_ctx,
        filter=f"exchange_id == {unique_exchange}",
        limit=1,
    )
    assert rows and isinstance(rows, list)
    raw = rows[0].entries
    assert raw.get("_metadata") == meta

    # 3) Manager retrieval excludes private fields → _metadata should not appear
    msgs = tm._filter_messages(filter=f"exchange_id == {unique_exchange}")
    assert len(msgs) == 1
    assert getattr(msgs[0], "_metadata", None) is None


@pytest.mark.unit
@_handle_project
def test_transcript_manager_clear():
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
    pre = tm._filter_messages()
    assert len(pre) >= 2

    # Execute clear
    tm.clear()

    # After clear: contexts should exist again and private column present
    fields_transcripts = unify.get_fields(context=tm._transcripts_ctx)
    assert "_metadata" in fields_transcripts

    fields_exchanges = unify.get_fields(context=tm._exchanges_ctx)
    assert "exchange_id" in fields_exchanges

    # Prior messages should be gone
    post = tm._filter_messages()
    assert len(post) == 0
