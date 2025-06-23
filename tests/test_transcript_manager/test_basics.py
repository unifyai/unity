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
    tm._log_messages(
        [
            Message(
                medium=random.choice(VALID_MEDIA),
                sender_id=random.randint(0, 2),
                receiver_id=random.randint(0, 2),
                timestamp=datetime.now(UTC),
                content=random.choice(MESSAGES),
                exchange_id=i,
            ).to_post_json()
            for i in range(10)
        ],
    )


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_get_messages():
    start_time = datetime.now(UTC).isoformat()
    time.sleep(0.1)
    random.seed(0)
    tm = TranscriptManager()

    # log messages
    tm._log_messages(
        [
            Message(
                medium=random.choice(VALID_MEDIA),
                sender_id=random.randint(0, 2),
                receiver_id=random.randint(0, 2),
                timestamp=datetime.now(UTC),
                content=random.choice(MESSAGES),
                exchange_id=i,
            ).to_post_json()
            for i in range(10)
        ],
    )

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


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_summarize_exchanges():
    tm = TranscriptManager()

    # create contacts
    _create_contacts()

    # phone call
    tm._log_messages(
        [
            Message(
                medium="phone_call",
                sender_id=i % 2,
                receiver_id=(i + 1) % 2,
                timestamp=datetime.now(UTC),
                content=msg,
                exchange_id=0,
            ).to_post_json()
            for i, msg in enumerate(
                [
                    "Hey, how's it going?",
                    "Yeah good thanks, how can I help you?",
                    "How are your office staplers doing? Are they underperforming?",
                    "Actually yeah, they're a bit rusty, but I can't make any buying decisions. My manager can.",
                    "Okay, no worries. Let's catch up again soon.",
                ],
            )
        ],
    )

    # email exchange
    tm._log_messages(
        [
            Message(
                medium="email",
                sender_id=i % 2,
                receiver_id=(i + 1) % 2,
                timestamp=datetime.now(UTC),
                content=msg,
                exchange_id=1,
            ).to_post_json()
            for i, msg in enumerate(
                [
                    "Great catching up the other day, did you manage to talk to your manager?",
                    "Hey, yeah I did actually. I'll reach out soon.",
                    "Okay great, thanks!",
                ],
            )
        ],
    )

    # whatsapp exchange
    tm._log_messages(
        [
            Message(
                medium="whatsapp_message",
                sender_id=(i + 1) % 2,
                receiver_id=i % 2,
                timestamp=datetime.now(UTC),
                content=msg,
                exchange_id=2,
            ).to_post_json()
            for i, msg in enumerate(
                [
                    "Hey, yeah we'd love to buy your staplers!",
                    "Great! Excited to hear :)",
                ],
            )
        ],
    )

    # summarize
    handle = await tm.summarize(from_exchanges=[0, 1, 2])
    summary = await handle.result()

    # retrieve summary
    summaries = tm._search_summaries()
    assert len(summaries) == 1
    summ = summaries[0].model_dump()
    # a) new column exists and is non-empty list
    assert isinstance(summ["message_ids"], list) and summ["message_ids"]
    # b) remaining keys still as expected
    for k, v in {
        "summary_id": 0,
        "exchange_ids": [0, 1, 2],
        "summary": summary,
    }.items():
        assert summ[k] == v
