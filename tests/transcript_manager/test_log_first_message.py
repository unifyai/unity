from __future__ import annotations

import pytest
from datetime import datetime, UTC

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message
from tests.helpers import _handle_project
import unify


@_handle_project
def test_rejects_explicit_id_dict():
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


@_handle_project
def test_rejects_explicit_id_model():
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


@_handle_project
def test_creates_exchange_returns_id():
    tm = TranscriptManager()

    payload = {
        "medium": "sms_message",
        "sender_id": 1,
        "receiver_ids": [2],
        "timestamp": datetime.now(UTC),
        "content": "first sms",
    }

    exid = tm.log_first_message_in_new_exchange(payload)

    assert isinstance(exid, int) and exid >= 0

    # Exchanges row should exist (metadata default to dict) and medium set
    rows_e = unify.get_logs(
        context=tm._exchanges_ctx,
        filter=f"exchange_id == {exid}",
        limit=1,
    )
    assert rows_e and rows_e[0].entries.get("exchange_id") == exid
    assert isinstance(rows_e[0].entries.get("metadata"), dict)
    assert rows_e[0].entries.get("medium") == "sms_message"

    # Transcript message should exist for this exchange
    rows_m = unify.get_logs(
        context=tm._transcripts_ctx,
        filter=f"exchange_id == {exid}",
        limit=1,
    )
    assert rows_m and rows_m[0].entries.get("exchange_id") == exid


@_handle_project
def test_sets_initial_metadata():
    tm = TranscriptManager()

    payload = {
        "medium": "email",
        "sender_id": 0,
        "receiver_ids": [1],
        "timestamp": datetime.now(UTC),
        "content": "first email",
    }
    meta = {"thread_id": "abc123", "origin": "inbound"}

    exid = tm.log_first_message_in_new_exchange(payload, exchange_initial_metadata=meta)
    assert isinstance(exid, int) and exid >= 0

    rows_e = unify.get_logs(
        context=tm._exchanges_ctx,
        filter=f"exchange_id == {exid}",
        limit=1,
    )
    assert rows_e and rows_e[0].entries.get("exchange_id") == exid
    assert rows_e[0].entries.get("medium") == "email"
    assert rows_e[0].entries.get("metadata") == meta
