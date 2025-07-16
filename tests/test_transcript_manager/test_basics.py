import time
import unify
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


def _create_contacts():
    unify.create_logs(
        context="Contacts",
        entries=CONTACTS,
    )


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
    random.seed(0)
    tm = TranscriptManager()

    # log messages
    for i in range(10):
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
    tm.join_published()

    ## get all

    messages = tm._search_messages()
    assert len(messages) == 10
    assert all(isinstance(msg, Message) for msg in messages)

    ## search

    # sender

    messages = tm._search_messages(filter="sender_id == 0")
    assert len(messages) == 3
    assert all(isinstance(msg, Message) for msg in messages)

    # contains

    messages = tm._search_messages(filter="'Hell' in content")
    assert len(messages) == 3
    assert all(isinstance(msg, Message) for msg in messages)

    # does not contain

    messages = tm._search_messages(filter="',' not in content")
    assert len(messages) == 5
    assert all(isinstance(msg, Message) for msg in messages)

    # medium

    messages = tm._search_messages(
        filter="medium in ('email', 'whatsapp_message')",
    )
    assert len(messages) == 1
    assert all(isinstance(msg, Message) for msg in messages)

    # timestamp

    messages = tm._search_messages(filter=f"timestamp < '{start_time}'")
    assert len(messages) == 0
    messages = tm._search_messages(filter=f"timestamp > '{start_time}'")
    assert len(messages) == 10
