from __future__ import annotations

import pytest
from datetime import datetime, UTC

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.conversation_manager.cm_types import VALID_MEDIA
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


@_handle_project
def test_basic_logging_explicit_id():
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


@_handle_project
def test_missing_id_raises_error():
    tm = TranscriptManager()

    # Intentionally omit exchange_id → should raise with guidance to use the new API
    with pytest.raises(ValueError) as exc:
        tm.log_messages(_base_message(0), synchronous=True)

    msg = str(exc.value)
    assert "exchange_id" in msg
    assert "log_first_message_in_new_exchange" in msg


@_handle_project
def test_async_mode_uses_async_logger(monkeypatch):
    """
    With synchronous=False, log_messages should use the async logger
    (AsyncLoggerManager.log_create) rather than blocking unify.log.
    """
    import unify

    tm = TranscriptManager()

    # Track which logging path is used for Transcripts (not Exchanges or other tables)
    sync_transcript_calls = []
    async_log_create_calls = []

    original_unify_log = unify.log
    original_log_create = TranscriptManager._LOGGER.log_create

    def mock_unify_log(*args, **kwargs):
        # Only track calls to Transcripts context, ignore Exchanges etc.
        ctx = kwargs.get("context", "")
        if "Transcripts" in ctx:
            sync_transcript_calls.append((args, kwargs))
        return original_unify_log(*args, **kwargs)

    def mock_log_create(*args, **kwargs):
        async_log_create_calls.append((args, kwargs))
        return original_log_create(*args, **kwargs)

    monkeypatch.setattr(unify, "log", mock_unify_log)
    monkeypatch.setattr(TranscriptManager._LOGGER, "log_create", mock_log_create)

    message = {
        "medium": VALID_MEDIA[0],
        "sender_id": 0,
        "receiver_ids": [1],
        "timestamp": datetime.now(UTC),
        "content": "async test message",
        "exchange_id": 99999,
    }

    # Clear any setup-related calls
    sync_transcript_calls.clear()
    async_log_create_calls.clear()

    tm.log_messages(message, synchronous=False)

    # With synchronous=False, the async logger (log_create) should be used,
    # NOT the blocking unify.log for Transcripts
    assert (
        len(async_log_create_calls) > 0
    ), "synchronous=False should use AsyncLoggerManager.log_create, but it wasn't called"
    assert len(sync_transcript_calls) == 0, (
        f"synchronous=False should NOT call unify.log for Transcripts persistence, "
        f"but it was called {len(sync_transcript_calls)} time(s)"
    )

    tm.join_published()
