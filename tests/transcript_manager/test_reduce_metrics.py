from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.conversation_manager.cm_types import VALID_MEDIA
from unity.transcript_manager.types.message import Message
from unity.contact_manager.contact_manager import ContactManager


@pytest.mark.requires_real_unify
@_handle_project
def test_transcript_reduce_param_shapes():
    tm = TranscriptManager()
    cm = ContactManager()

    # Seed contacts and messages so metrics have real data to aggregate
    a_id = cm._create_contact(first_name="Alice")["details"]["contact_id"]
    b_id = cm._create_contact(first_name="Bob")["details"]["contact_id"]

    msgs = [
        Message(
            medium=VALID_MEDIA[0],
            sender_id=a_id,
            receiver_ids=[b_id],
            timestamp="2025-01-01 10:00:00",
            content="hello one",
            exchange_id=1,
        ),
        Message(
            medium=VALID_MEDIA[1 if len(VALID_MEDIA) > 1 else 0],
            sender_id=b_id,
            receiver_ids=[a_id],
            timestamp="2025-01-01 10:00:01",
            content="hello two",
            exchange_id=1,
        ),
        Message(
            medium=VALID_MEDIA[0],
            sender_id=a_id,
            receiver_ids=[b_id],
            timestamp="2025-01-01 10:00:02",
            content="hello three",
            exchange_id=2,
        ),
    ]
    tm.log_messages(msgs, synchronous=True)
    tm.join_published()

    # Single key, no grouping
    scalar = tm._reduce(metric="sum", keys="message_id")
    assert isinstance(scalar, (int, float))

    # Multiple keys, no grouping
    multi = tm._reduce(metric="max", keys=["message_id"])
    assert isinstance(multi, dict)
    assert set(multi.keys()) == {"message_id"}

    # Single key, group_by string
    grouped_str = tm._reduce(
        metric="sum",
        keys="message_id",
        group_by="medium",
    )
    assert isinstance(grouped_str, dict)

    # Multiple keys, group_by string
    grouped_str_multi = tm._reduce(
        metric="min",
        keys=["message_id"],
        group_by="medium",
    )
    assert isinstance(grouped_str_multi, dict)

    # Single key, group_by list
    grouped_list = tm._reduce(
        metric="sum",
        keys="message_id",
        group_by=["medium", "sender_id"],
    )
    assert isinstance(grouped_list, dict)

    # Multiple keys, group_by list
    grouped_list_multi = tm._reduce(
        metric="mean",
        keys=["message_id"],
        group_by=["medium", "sender_id"],
    )
    assert isinstance(grouped_list_multi, dict)

    # Filter as string
    filtered_scalar = tm._reduce(
        metric="sum",
        keys="message_id",
        filter="message_id >= 0",
    )
    assert isinstance(filtered_scalar, (int, float))

    # Filter as per-key dict
    filtered_multi = tm._reduce(
        metric="sum",
        keys=["message_id"],
        filter={"message_id": "message_id >= 0"},
    )
    assert isinstance(filtered_multi, dict)
