from communication.message import Message
from datetime import datetime


def test_message():
    timestamp = datetime.now()
    msg = Message(
        medium="email",
        sender="Daniel Lenton",
        receiver="My Assistant",
        timestamp=timestamp,
        content="Hello, how are you?",
        exchange_id=0,
    )
    assert msg.medium == "email"
    assert msg.sender == "Daniel Lenton"
    assert msg.receiver == "My Assistant"
    assert msg.timestamp == timestamp
    assert msg.content == "Hello, how are you?"
    assert msg.exchange_id == 0
