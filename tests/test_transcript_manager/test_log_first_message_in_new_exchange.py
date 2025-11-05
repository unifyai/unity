from __future__ import annotations

import pytest
from datetime import datetime, UTC

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message
from tests.helpers import _handle_project


@pytest.mark.unit
@_handle_project
def test_log_first_message_rejects_explicit_exchange_id_dict_payload():
    tm = TranscriptManager()

    payload = {
        "medium": "email",
        "sender_id": 0,
        "receiver_ids": [1],
        "timestamp": datetime.now(UTC),
        "content": "First message",
        "exchange_id": 123,  # invalid for this method
    }

    with pytest.raises(ValueError) as exc:
        tm.log_first_message_in_new_exchange(payload)

    msg = str(exc.value)
    assert "exchange_id" in msg
    assert "log_messages" in msg


@pytest.mark.unit
@_handle_project
def test_log_first_message_rejects_explicit_exchange_id_message_instance():
    tm = TranscriptManager()

    m = Message(
        medium="email",
        sender_id=0,
        receiver_ids=[1],
        timestamp=datetime.now(UTC),
        content="First message",
        exchange_id=42,  # invalid for this method
    )

    with pytest.raises(ValueError) as exc:
        tm.log_first_message_in_new_exchange(m)

    msg = str(exc.value)
    assert "exchange_id" in msg
    assert "log_messages" in msg
