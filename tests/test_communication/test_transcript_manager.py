import random
from datetime import datetime

from communication.message import Message, VALID_MEDIA
from communication.transcript_manager import TranscriptManager
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
    "Wow, did you see that?",
    "Goodbye",
]


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
