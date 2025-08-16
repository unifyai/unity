import time
import random
import pytest
from datetime import datetime, UTC

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
