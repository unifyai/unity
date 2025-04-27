from communication.message import Message
from datetime import datetime


def test_message():
    timestamp = datetime.now().isoformat()
    msg = Message(
        medium="email",
        sender_id=0,
        receiver_id=1,
        timestamp=timestamp,
        content="Hello, how are you?",
        exchange_id=0,
    )
    assert msg.medium == "email"
    assert msg.sender_id == 0
    assert msg.receiver_id == 1
    assert msg.timestamp == timestamp
    assert msg.content == "Hello, how are you?"
    assert msg.exchange_id == 0
