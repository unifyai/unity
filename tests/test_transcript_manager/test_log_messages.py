from __future__ import annotations

import pytest
from datetime import datetime, UTC

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import VALID_MEDIA
from tests.helpers import _handle_project
from unity.contact_manager.types.contact import Contact


def _base_message(seed: int) -> dict:
    # Use Contact objects to auto-create contacts instead of hard-coded ids
    names = [
        ("Alice", "Bob"),
        ("Carlos", "Diana"),
        ("Eve", "Frank"),
    ]
    snd, rcv = names[seed % len(names)]
    return {
        "medium": VALID_MEDIA[seed % len(VALID_MEDIA)],
        "sender_id": Contact(first_name=snd),
        "receiver_ids": [Contact(first_name=rcv)],
        "timestamp": datetime.now(UTC),
        "content": f"msg-{seed}",
        # Note: exchange_id is intentionally omitted so it is auto-created
    }


@pytest.mark.unit
@_handle_project
def test_log_messages_sync_returns_ids_and_auto_increment():
    tm = TranscriptManager()

    # First message (no explicit ids provided)
    created1 = tm.log_messages(_base_message(0), synchronous=True)
    assert isinstance(created1, list) and len(created1) == 1
    m1 = created1[0]

    # IDs should be non-negative and increment upwards
    assert isinstance(m1.message_id, int) and m1.message_id >= 0
    assert isinstance(m1.exchange_id, int) and m1.exchange_id >= 0

    # Second message → ids should auto-increment independently
    created2 = tm.log_messages(_base_message(1), synchronous=True)
    assert isinstance(created2, list) and len(created2) == 1
    m2 = created2[0]

    assert m2.message_id == m1.message_id + 1
    assert m2.exchange_id == m1.exchange_id + 1


@pytest.mark.unit
@_handle_project
def test_log_messages_async_auto_increment_visible_via_filter_messages():
    tm = TranscriptManager()

    # Create multiple messages in async mode (we'll assert via _filter_messages)
    for i in range(3):
        tm.log_messages(_base_message(i), synchronous=False)

    tm.join_published()

    # Fetch messages and verify id sequences start at 0 and increment by 1
    msgs = tm._filter_messages(limit=None)["messages"]
    assert len(msgs) == 3

    # Sort by timestamp to preserve insertion order
    msgs_sorted = sorted(msgs, key=lambda m: m.timestamp)
    message_ids = [m.message_id for m in msgs_sorted]
    exchange_ids = [m.exchange_id for m in msgs_sorted]

    expected = list(range(0, len(msgs_sorted)))
    assert message_ids == expected
    assert exchange_ids == expected

    # Additionally, ensure both counters advance independently (not tied together)
    assert (message_ids[-1] - message_ids[0]) == 2
    assert (exchange_ids[-1] - exchange_ids[0]) == 2
    assert message_ids == exchange_ids


@pytest.mark.unit
@_handle_project
def test_log_messages_missing_exchange_id_raises_helpful_error():
    tm = TranscriptManager()

    # Intentionally omit exchange_id → should raise with guidance to use the new API
    with pytest.raises(ValueError) as exc:
        tm.log_messages(_base_message(0), synchronous=True)

    msg = str(exc.value)
    assert "exchange_id" in msg
    assert "log_first_message_in_new_exchange" in msg
