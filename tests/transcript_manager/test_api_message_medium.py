"""Verify the api_message medium works correctly in the TranscriptManager."""

from __future__ import annotations

from datetime import UTC, datetime

import unify

from tests.helpers import _handle_project
from unity.conversation_manager.types import Medium
from unity.transcript_manager.transcript_manager import TranscriptManager


@_handle_project
def test_api_message_in_valid_media():
    """api_message is a valid medium value."""
    from unity.conversation_manager.types import VALID_MEDIA

    assert "api_message" in VALID_MEDIA
    assert Medium.API_MESSAGE == "api_message"
    assert Medium.API_MESSAGE.mode.value == "text"


@_handle_project
def test_log_and_filter_api_message():
    """Messages with medium=api_message can be logged and filtered."""
    tm = TranscriptManager()

    exchange_id, msg_id = tm.log_first_message_in_new_exchange(
        {
            "medium": "api_message",
            "sender_id": 1,
            "receiver_ids": [0],
            "timestamp": datetime.now(UTC),
            "content": "Hello from the API",
        },
    )
    assert isinstance(exchange_id, int)

    msgs = tm._filter_messages(
        filter=f"exchange_id == {exchange_id}",
        limit=1,
    )["messages"]
    assert len(msgs) == 1
    assert msgs[0].medium == "api_message"
    assert msgs[0].content == "Hello from the API"


@_handle_project
def test_api_message_exchange_medium():
    """Exchange created from an api_message has the correct medium."""
    tm = TranscriptManager()

    exchange_id, _ = tm.log_first_message_in_new_exchange(
        {
            "medium": "api_message",
            "sender_id": 1,
            "receiver_ids": [0],
            "timestamp": datetime.now(UTC),
            "content": "API exchange test",
        },
    )

    rows = unify.get_logs(
        context=tm._exchanges_ctx,
        filter=f"exchange_id == {exchange_id}",
        limit=1,
    )
    assert rows
    assert rows[0].entries.get("medium") == "api_message"
