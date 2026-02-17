from __future__ import annotations

from datetime import datetime, UTC
import unify

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.exchange import Exchange
from unity.transcript_manager.types.message import Message
from tests.helpers import _handle_project


@_handle_project
def test_row_created_explicit_id():
    tm = TranscriptManager()

    ex_id = 424242
    tm.log_messages(
        Message(
            medium="email",
            sender_id=0,
            receiver_ids=[1],
            timestamp=datetime.now(UTC),
            content="exchanges explicit id",
            exchange_id=ex_id,
        ),
    )
    tm.join_published()

    rows = unify.get_logs(
        context=tm._exchanges_ctx,
        filter=f"exchange_id == {ex_id}",
        limit=1,
    )
    assert rows and rows[0].entries.get("exchange_id") == ex_id
    assert isinstance(rows[0].entries.get("metadata"), dict)
    assert rows[0].entries.get("medium") == "email"


@_handle_project
def test_get_metadata_returns_model():
    tm = TranscriptManager()

    ex_id = 555001
    tm.log_messages(
        Message(
            medium="email",
            sender_id=0,
            receiver_ids=[1],
            timestamp=datetime.now(UTC),
            content="exchanges metadata fetch",
            exchange_id=ex_id,
        ),
    )
    tm.join_published()

    ex = tm.get_exchange_metadata(ex_id)
    assert isinstance(ex, Exchange)
    assert ex.exchange_id == ex_id
    assert ex.medium == "email"
    assert isinstance(ex.metadata, dict)


@_handle_project
def test_update_metadata_updates_row():
    tm = TranscriptManager()

    # Create a fresh exchange via the new helper
    exid, _ = tm.log_first_message_in_new_exchange(
        {
            "medium": "sms_message",
            "sender_id": 1,
            "receiver_ids": [2],
            "timestamp": datetime.now(UTC),
            "content": "seed",
        },
    )
    tm.join_published()

    # Update metadata
    new_meta = {"foo": "bar", "n": 1}
    updated = tm.update_exchange_metadata(exid, new_meta)
    assert isinstance(updated, Exchange)
    assert updated.exchange_id == exid
    assert updated.metadata == new_meta
    # Medium should remain as originally set
    assert updated.medium == "sms_message"

    # Round-trip fetch
    fetched = tm.get_exchange_metadata(exid)
    assert fetched.metadata == new_meta


@_handle_project
def test_update_metadata_upserts():
    tm = TranscriptManager()

    exid = 987654321
    meta = {"seed": "manual"}
    upserted = tm.update_exchange_metadata(exid, meta)

    assert isinstance(upserted, Exchange)
    assert upserted.exchange_id == exid
    assert upserted.metadata == meta
    # Upsert path sets blank medium
    assert upserted.medium == ""


@_handle_project
def test_filter_by_metadata():
    tm = TranscriptManager()

    # Seed two new exchanges with distinct metadata
    ex_billing, _ = tm.log_first_message_in_new_exchange(
        {
            "medium": "sms_message",
            "sender_id": 1,
            "receiver_ids": [2],
            "timestamp": datetime.now(UTC),
            "content": "seed billing",
        },
        exchange_initial_metadata={"topic": "billing", "ref": "A"},
    )
    ex_support, _ = tm.log_first_message_in_new_exchange(
        {
            "medium": "sms_message",
            "sender_id": 1,
            "receiver_ids": [2],
            "timestamp": datetime.now(UTC),
            "content": "seed support",
        },
        exchange_initial_metadata={"topic": "support", "ref": "B"},
    )

    # Restrict the search space to our seeded ids for determinism
    ids_expr = f"exchange_id in [{ex_billing}, {ex_support}]"
    ret = tm.filter_exchanges(
        filter=f"{ids_expr} and metadata.get('topic') == 'billing'",
    )
    exchanges = ret.get("exchanges", [])
    assert isinstance(exchanges, list)
    assert any(isinstance(e, Exchange) for e in exchanges)
    # Exactly one should match billing within the restricted ids
    assert len(exchanges) == 1
    assert exchanges[0].exchange_id == ex_billing
    assert exchanges[0].metadata.get("topic") == "billing"


@_handle_project
def test_filter_by_nested_metadata():
    tm = TranscriptManager()

    # Seed nested metadata with tags
    ex_nested, _ = tm.log_first_message_in_new_exchange(
        {
            "medium": "email",
            "sender_id": 3,
            "receiver_ids": [4],
            "timestamp": datetime.now(UTC),
            "content": "nested meta seed",
        },
        exchange_initial_metadata={
            "thread": {"id": "T-123", "tags": ["billing", "vip"]},
        },
    )
    ex_other, _ = tm.log_first_message_in_new_exchange(
        {
            "medium": "email",
            "sender_id": 3,
            "receiver_ids": [4],
            "timestamp": datetime.now(UTC),
            "content": "other seed",
        },
        exchange_initial_metadata={
            "thread": {"id": "T-999", "tags": ["ops"]},
        },
    )

    ids_expr = f"exchange_id in [{ex_nested}, {ex_other}]"
    # Safe guards against missing keys; check id AND membership in tags
    flt = (
        f"{ids_expr} and "
        "metadata.get('thread') and "
        "metadata['thread'].get('id') == 'T-123' and "
        "'vip' in (metadata['thread'].get('tags') or [])"
    )
    ret = tm.filter_exchanges(filter=flt)
    exchanges = ret.get("exchanges", [])
    assert len(exchanges) == 1
    ex = exchanges[0]
    assert isinstance(ex, Exchange)
    assert ex.exchange_id == ex_nested
    assert ex.metadata.get("thread", {}).get("id") == "T-123"
    assert "vip" in (ex.metadata.get("thread", {}).get("tags") or [])
