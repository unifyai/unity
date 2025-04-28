import time
import unify
import random
from datetime import datetime

from communication.types.message import Message, VALID_MEDIA
from communication.transcript_manager.transcript_manager import TranscriptManager
from tests.helpers import _handle_project

CONTACTS = [
    {
        "user_id": 0,
        "first_name": "John",
        "surname": "Smith",
        "email": "johnsmith11@gmail.com",
        "phone": "+1234567890",
        "whatsapp": "+1234567890",
    },
    {
        "user_id": 1,
        "first_name": "Nancy",
        "surname": "Gray",
        "email": "nancy_gray@outlook.com",
        "phone": "+1987654320",
        "whatsapp": "+1987654320",
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


@_handle_project
def test_create_contact():
    transcript_manager = TranscriptManager()
    transcript_manager.start()
    transcript_manager.create_contact(
        first_name="Dan",
    )
    contacts = transcript_manager._get_contacts()
    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.model_dump() == {
        "contact_id": 0,
        "first_name": "Dan",
        "surname": None,
        "email_address": None,
        "phone_number": None,
        "whatsapp_number": None,
    }


@_handle_project
def test_update_contact():
    transcript_manager = TranscriptManager()
    transcript_manager.start()

    # create
    transcript_manager.create_contact(
        first_name="Dan",
    )

    # check
    contacts = transcript_manager._get_contacts()
    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.model_dump() == {
        "contact_id": 0,
        "first_name": "Dan",
        "surname": None,
        "email_address": None,
        "phone_number": None,
        "whatsapp_number": None,
    }

    # update
    transcript_manager.update_contact(
        contact_id=0,
        first_name="Daniel",
    )

    # check
    contacts = transcript_manager._get_contacts()
    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.model_dump() == {
        "contact_id": 0,
        "first_name": "Daniel",
        "surname": None,
        "email_address": None,
        "phone_number": None,
        "whatsapp_number": None,
    }


@_handle_project
def test_create_contacts():
    transcript_manager = TranscriptManager()
    transcript_manager.start()

    # first
    transcript_manager.create_contact(
        first_name="Dan",
    )
    contacts = transcript_manager._get_contacts()
    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.model_dump() == {
        "contact_id": 0,
        "first_name": "Dan",
        "surname": None,
        "email_address": None,
        "phone_number": None,
        "whatsapp_number": None,
    }

    # second
    transcript_manager.create_contact(
        first_name="Tom",
    )
    contacts = transcript_manager._get_contacts()
    assert len(contacts) == 2
    contact = contacts[0]
    assert contact.model_dump() == {
        "contact_id": 1,
        "first_name": "Tom",
        "surname": None,
        "email_address": None,
        "phone_number": None,
        "whatsapp_number": None,
    }


@_handle_project
def test_search_contacts():
    transcript_manager = TranscriptManager()
    transcript_manager.start()
    transcript_manager.create_contact(
        first_name="Dan",
    )


@_handle_project
def test_log_messages():
    transcript_manager = TranscriptManager()
    transcript_manager.start()
    transcript_manager.log_messages(
        [
            Message(
                medium=random.choice(VALID_MEDIA),
                sender_id=random.randint(0, 2),
                receiver_id=random.randint(0, 2),
                timestamp=datetime.now().isoformat(),
                content=random.choice(MESSAGES),
                exchange_id=i,
            )
            for i in range(10)
        ],
    )


@_handle_project
def test_get_messages():
    start_time = datetime.now().isoformat()
    time.sleep(0.1)
    random.seed(0)
    transcript_manager = TranscriptManager()
    transcript_manager.start()
    # log messages
    transcript_manager.log_messages(
        [
            Message(
                medium=random.choice(VALID_MEDIA),
                sender_id=random.randint(0, 2),
                receiver_id=random.randint(0, 2),
                timestamp=datetime.now().isoformat(),
                content=random.choice(MESSAGES),
                exchange_id=i,
            )
            for i in range(10)
        ],
    )

    ## get all

    messages = transcript_manager._get_messages()
    assert len(messages) == 10
    assert all(isinstance(msg, Message) for msg in messages)

    ## search

    # sender

    messages = transcript_manager._get_messages(filter="sender_id == 0")
    assert len(messages) == 3
    assert all(isinstance(msg, Message) for msg in messages)

    # contains

    messages = transcript_manager._get_messages(filter="'Hell' in content")
    assert len(messages) == 3
    assert all(isinstance(msg, Message) for msg in messages)

    # does not contain

    messages = transcript_manager._get_messages(filter="',' not in content")
    assert len(messages) == 5
    assert all(isinstance(msg, Message) for msg in messages)

    # medium

    messages = transcript_manager._get_messages(
        filter="medium in ('email', 'whatsapp_message')",
    )
    assert len(messages) == 1
    assert all(isinstance(msg, Message) for msg in messages)

    # timestamp

    messages = transcript_manager._get_messages(filter=f"timestamp < '{start_time}'")
    assert len(messages) == 0
    messages = transcript_manager._get_messages(filter=f"timestamp > '{start_time}'")
    assert len(messages) == 10


@_handle_project
def test_summarize_exchanges():
    transcript_manager = TranscriptManager()
    transcript_manager.start()

    # create contacts
    _create_contacts()

    # phone call
    transcript_manager.log_messages(
        [
            Message(
                medium="phone_call",
                sender_id=i % 2,
                receiver_id=(i + 1) % 2,
                timestamp=datetime.now().isoformat(),
                content=msg,
                exchange_id=0,
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
        ],
    )

    # email exchange
    transcript_manager.log_messages(
        [
            Message(
                medium="email",
                sender_id=i % 2,
                receiver_id=(i + 1) % 2,
                timestamp=datetime.now().isoformat(),
                content=msg,
                exchange_id=1,
            )
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
    transcript_manager.log_messages(
        [
            Message(
                medium="whatsapp_message",
                sender_id=(i + 1) % 2,
                receiver_id=i % 2,
                timestamp=datetime.now().isoformat(),
                content=msg,
                exchange_id=2,
            )
            for i, msg in enumerate(
                [
                    "Hey, yeah we'd love to buy your staplers!",
                    "Great! Excited to hear :)",
                ],
            )
        ],
    )

    # summarize
    summary = transcript_manager.summarize([0, 1, 2])

    # retrieve summary
    summaries = transcript_manager._get_summaries()
    assert len(summaries) == 1
    assert summaries[0].model_dump() == {
        "exchange_ids": [0, 1, 2],
        "summary": summary,
    }
