from __future__ import annotations

from datetime import datetime, UTC
import unify
import pytest

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message
from tests.helpers import _handle_project


@pytest.mark.unit
@_handle_project
def test_exchanges_row_created_for_explicit_exchange_id():
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


@pytest.mark.unit
@_handle_project
def test_exchanges_row_created_for_auto_assigned_exchange_id():
    tm = TranscriptManager()

    created = tm.log_messages(
        Message(
            medium="sms_message",
            sender_id=1,
            receiver_ids=[2],
            timestamp=datetime.now(UTC),
            content="auto exchange id",
            # exchange_id intentionally omitted to use auto-counting
        ),
        synchronous=True,
    )
    tm.join_published()

    assert created and created[0].exchange_id is not None
    ex_id = int(created[0].exchange_id)
    assert ex_id >= 0

    rows = unify.get_logs(
        context=tm._exchanges_ctx,
        filter=f"exchange_id == {ex_id}",
        limit=1,
    )
    assert rows and int(rows[0].entries.get("exchange_id")) == ex_id
    assert isinstance(rows[0].entries.get("metadata"), dict)
