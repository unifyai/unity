import time
import unify
import random
import pytest
from datetime import datetime, UTC

from unity.transcript_manager.types.message import Message, VALID_MEDIA
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.events.event_bus import EventBus, Event
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
    event_bus = EventBus()
    [
        await event_bus.publish(
            Event(
                type="message",
                timestamp=datetime.now(UTC).isoformat(),
                payload=Message(
                    medium=random.choice(VALID_MEDIA),
                    sender_id=random.randint(0, 2),
                    receiver_id=random.randint(0, 2),
                    timestamp=datetime.now(UTC).isoformat(),
                    content=random.choice(MESSAGES),
                    exchange_id=i,
                ),
            ),
        )
        for i in range(10)
    ]
    event_bus.join_published()


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_get_messages():
    start_time = datetime.now(UTC).isoformat()
    time.sleep(0.1)
    random.seed(0)
    event_bus = EventBus()
    transcript_manager = TranscriptManager(event_bus)

    # log messages
    [
        await event_bus.publish(
            Event(
                type="Messages",
                timestamp=datetime.now(UTC).isoformat(),
                payload=Message(
                    medium=random.choice(VALID_MEDIA),
                    sender_id=random.randint(0, 2),
                    receiver_id=random.randint(0, 2),
                    timestamp=datetime.now(UTC).isoformat(),
                    content=random.choice(MESSAGES),
                    exchange_id=i,
                ),
            ),
        )
        for i in range(10)
    ]
    event_bus.join_published()

    ## get all

    messages = transcript_manager._search_messages()
    assert len(messages) == 10
    assert all(isinstance(msg, Message) for msg in messages)

    ## search

    # sender

    messages = transcript_manager._search_messages(filter="sender_id == 0")
    assert len(messages) == 3
    assert all(isinstance(msg, Message) for msg in messages)

    # contains

    messages = transcript_manager._search_messages(filter="'Hell' in content")
    assert len(messages) == 3
    assert all(isinstance(msg, Message) for msg in messages)

    # does not contain

    messages = transcript_manager._search_messages(filter="',' not in content")
    assert len(messages) == 5
    assert all(isinstance(msg, Message) for msg in messages)

    # medium

    messages = transcript_manager._search_messages(
        filter="medium in ('email', 'whatsapp_message')",
    )
    assert len(messages) == 1
    assert all(isinstance(msg, Message) for msg in messages)

    # timestamp

    messages = transcript_manager._search_messages(filter=f"timestamp < '{start_time}'")
    assert len(messages) == 0
    messages = transcript_manager._search_messages(filter=f"timestamp > '{start_time}'")
    assert len(messages) == 10


@pytest.mark.unit
@pytest.mark.asyncio
@_handle_project
async def test_summarize_exchanges():
    event_bus = EventBus()
    transcript_manager = TranscriptManager(event_bus)

    # create contacts
    _create_contacts()

    # phone call
    [
        await event_bus.publish(
            Event(
                type="Messages",
                timestamp=datetime.now(UTC).isoformat(),
                payload=Message(
                    medium="phone_call",
                    sender_id=i % 2,
                    receiver_id=(i + 1) % 2,
                    timestamp=datetime.now(UTC).isoformat(),
                    content=msg,
                    exchange_id=0,
                ),
            ),
        )
        for i, msg in enumerate(
            [
                "Hey, how's it going?",
                "Yeah good thanks, how can I help you?",
                "How are your office staplers doing? Are they underperforming?",
                "Actually yeah, they're a bit rusty, but I can't make any buying decisions. My manager can.",
                "Okay, no worries. Let's catch up again soon.",
            ],
        )
    ]

    # email exchange
    [
        await event_bus.publish(
            Event(
                type="Messages",
                timestamp=datetime.now(UTC).isoformat(),
                payload=Message(
                    medium="email",
                    sender_id=i % 2,
                    receiver_id=(i + 1) % 2,
                    timestamp=datetime.now(UTC).isoformat(),
                    content=msg,
                    exchange_id=1,
                ),
            ),
        )
        for i, msg in enumerate(
            [
                "Great catching up the other day, did you manage to talk to your manager?",
                "Hey, yeah I did actually. I'll reach out soon.",
                "Okay great, thanks!",
            ],
        )
    ]

    # whatsapp exchange
    [
        await event_bus.publish(
            Event(
                type="Messages",
                timestamp=datetime.now(UTC).isoformat(),
                payload=Message(
                    medium="whatsapp_message",
                    sender_id=(i + 1) % 2,
                    receiver_id=i % 2,
                    timestamp=datetime.now(UTC).isoformat(),
                    content=msg,
                    exchange_id=2,
                ),
            ),
        )
        for i, msg in enumerate(
            [
                "Hey, yeah we'd love to buy your staplers!",
                "Great! Excited to hear :)",
            ],
        )
    ]
    event_bus.join_published()

    # summarize
    summary = await transcript_manager.summarize(exchange_ids=[0, 1, 2])
    event_bus.join_published()

    # retrieve summary
    summaries = transcript_manager._search_summaries()
    assert len(summaries) == 1
    assert summaries[0].model_dump() == {
        "exchange_ids": [0, 1, 2],
        "summary": summary,
    }
