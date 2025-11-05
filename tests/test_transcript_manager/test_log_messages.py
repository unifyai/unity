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
def test_log_messages_basic_logging_with_explicit_exchange_id():
    tm = TranscriptManager()

    ex_id = 135790
    created = tm.log_messages(
        {
            "medium": VALID_MEDIA[0],
            "sender_id": 0,
            "receiver_ids": [1],
            "timestamp": datetime.now(UTC),
            "content": "basic logging",
            "exchange_id": ex_id,
        },
        synchronous=True,
    )

    assert isinstance(created, list) and len(created) == 1
    m = created[0]
    assert isinstance(m.message_id, int) and m.message_id >= 0
    assert m.exchange_id == ex_id

    # Ensure the message is persisted and retrievable by exchange id
    msgs = tm._filter_messages(filter=f"exchange_id == {ex_id}", limit=1)["messages"]
    assert msgs and msgs[0].exchange_id == ex_id

    # Also verify events are published and flushable without error
    tm.join_published()


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
