from __future__ import annotations

from datetime import datetime, timezone, timedelta, UTC

import unify

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message
from tests.helpers import _handle_project


@_handle_project  # ensures isolated Unify context per run
def test_rewrites_sender_and_receivers():
    """Verify TranscriptManager.update_contact_id swaps ids across *all* messages."""

    tm = TranscriptManager()

    OLD_ID = 11  # arbitrary id unique to this test context
    NEW_ID = 42
    EX_ID = 999999  # unique exchange so we can filter easily

    base_ts = datetime.now(timezone.utc)

    # 1) Old id appears as *sender_id*
    tm.log_messages(
        Message(
            medium="email",
            sender_id=OLD_ID,
            receiver_ids=[OLD_ID + 1],
            timestamp=base_ts,
            content="Sender uses OLD_ID",
            exchange_id=EX_ID,
        ),
    )

    # 2) Old id appears inside *receiver_ids*
    tm.log_messages(
        Message(
            medium="email",
            sender_id=OLD_ID + 2,
            receiver_ids=[OLD_ID],
            timestamp=base_ts + timedelta(seconds=1),
            content="Receiver uses OLD_ID",
            exchange_id=EX_ID,
        ),
    )

    # 3) Old id appears multiple times in receiver_ids → should deduplicate
    tm.log_messages(
        Message(
            medium="email",
            sender_id=OLD_ID + 3,
            receiver_ids=[OLD_ID, OLD_ID, OLD_ID + 2],
            timestamp=base_ts + timedelta(seconds=2),
            content="Duplicate OLD_ID receivers",
            exchange_id=EX_ID,
        ),
    )

    # Flush logs to storage
    tm.join_published()

    # Sanity pre-check
    initial = tm._filter_messages(filter=f"exchange_id == {EX_ID}")["messages"]
    assert len(initial) == 3, "Exactly three messages should have been logged initially"

    # --- Run the update ---------------------------------------------------
    outcome = tm.update_contact_id(original_contact_id=OLD_ID, new_contact_id=NEW_ID)

    # Outcome shape & counts
    assert outcome["outcome"] == "contact ids updated"
    assert outcome["details"]["old_contact_id"] == OLD_ID
    assert outcome["details"]["new_contact_id"] == NEW_ID
    assert (
        outcome["details"]["updated_messages"] == 3
    ), "All three messages should be counted as updated"

    # --- Post-conditions --------------------------------------------------
    updated = tm._filter_messages(filter=f"exchange_id == {EX_ID}")["messages"]
    assert len(updated) == 3, "Message count should remain unchanged after update"

    for msg in updated:
        # Old id should be gone completely
        assert msg.sender_id != OLD_ID, "sender_id was not updated"
        assert OLD_ID not in msg.receiver_ids, "receiver_ids still contains old id"
        # New id should replace where appropriate
        if msg.content.startswith("Sender"):
            assert msg.sender_id == NEW_ID, "sender_id not replaced with new id"
        if msg.content.startswith("Receiver") or msg.content.startswith("Duplicate"):
            assert NEW_ID in msg.receiver_ids, "new id missing from receiver_ids"
        # No duplicate ids in receiver list
        assert len(msg.receiver_ids) == len(
            set(msg.receiver_ids),
        ), "receiver_ids not deduplicated"


@_handle_project
def test_rewrites_exchange_counterparty_contact_id():
    """``update_contact_id`` rewrites ``Exchanges.counterparty_contact_id``.

    A merge retires ``original_contact_id``. Without the Exchanges
    rewrite, the next body-to-body dedup query (which filters on
    ``counterparty_contact_id``) would still route to the retired row
    and the ``ON DELETE SET NULL`` cascade would later strip the
    counterparty pointer off the surviving exchange.
    """
    tm = TranscriptManager()

    OLD_CID = 7777
    NEW_CID = 8888

    exid, _ = tm.log_first_message_in_new_exchange(
        {
            "medium": "email",
            "sender_id": OLD_CID,
            "receiver_ids": [OLD_CID + 1],
            "timestamp": datetime.now(UTC),
            "content": "seed",
        },
    )
    tm.join_published()

    rows = unify.get_logs(
        context=tm._exchanges_ctx,
        filter=f"exchange_id == {exid}",
        limit=1,
    )
    assert rows and rows[0].entries.get("counterparty_contact_id") == OLD_CID

    outcome = tm.update_contact_id(
        original_contact_id=OLD_CID,
        new_contact_id=NEW_CID,
    )

    assert outcome["details"]["updated_exchanges"] >= 1

    rows = unify.get_logs(
        context=tm._exchanges_ctx,
        filter=f"exchange_id == {exid}",
        limit=1,
    )
    assert rows and rows[0].entries.get("counterparty_contact_id") == NEW_CID
