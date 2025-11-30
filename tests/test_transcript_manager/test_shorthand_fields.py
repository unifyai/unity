from __future__ import annotations

from datetime import datetime, UTC

from unity.transcript_manager.types.message import Message
from tests.helpers import _handle_project


@_handle_project
def test_json_shorthand_no_prune():
    msg = Message(
        medium="email",
        sender_id=7,
        receiver_ids=[3, 4],
        timestamp=datetime.now(UTC),
        content="hello",
        exchange_id=123,
    )

    dumped = msg.model_dump(mode="json", context={"shorthand": True})

    # Aliased keys should exist
    for k in ("mid", "med", "sid", "rids", "ts", "c", "xid"):
        assert k in dumped, f"expected shorthand key {k} in dump"

    # Original keys should not be present
    for k in (
        "message_id",
        "medium",
        "sender_id",
        "receiver_ids",
        "timestamp",
        "content",
        "exchange_id",
    ):
        assert k not in dumped, f"did not expect original key {k} in dump"

    # With shorthand only (no prune), default-empty images should remain under 'imgs'
    assert "imgs" in dumped and isinstance(dumped["imgs"], list)


@_handle_project
def test_json_shorthand_prune_images():
    msg = Message(
        medium="email",
        sender_id=1,
        receiver_ids=[2],
        timestamp=datetime.now(UTC),
        content="no images",
        exchange_id=4242,
    )

    dumped = msg.model_dump(
        mode="json",
        context={"shorthand": True, "prune_empty": True},
    )

    # Aliased keys present
    for k in ("mid", "med", "sid", "rids", "ts", "c", "xid"):
        assert k in dumped, f"expected shorthand key {k} in dump"

    # Empty images should be pruned when prune_empty=True, so 'imgs' absent
    assert "imgs" not in dumped
