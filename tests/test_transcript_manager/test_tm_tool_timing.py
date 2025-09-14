from __future__ import annotations

import os
import time
import pytest

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message
from tests.helpers import _handle_project


def _enable_timing():
    os.environ["CONTACT_MANAGER_TOOL_TIMING"] = "1"
    # Keep prints off by default; uncomment to see console timing
    # os.environ["CONTACT_MANAGER_TOOL_TIMING_PRINT"] = "1"


@pytest.mark.unit
@_handle_project
def test_tool_list_columns_timing():
    _enable_timing()
    tm = TranscriptManager()
    t0 = time.perf_counter()
    cols = tm._list_columns()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(cols, dict) and cols
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_filter_messages_timing():
    _enable_timing()
    tm = TranscriptManager()

    # seed minimal data
    tm.log_messages(
        Message(
            medium="email",
            sender_id=0,
            receiver_ids=[1],
            timestamp="2025-05-20 12:00:00",
            content="perf msg",
            exchange_id=4242,
        ),
    )
    tm.join_published()

    t0 = time.perf_counter()
    rows = tm._filter_messages(filter="exchange_id == 4242")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert rows and rows[0].exchange_id == 4242
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_search_messages_timing():
    _enable_timing()
    tm = TranscriptManager()

    # seed
    tm.log_messages(
        [
            Message(
                medium="email",
                sender_id=0,
                receiver_ids=[1],
                timestamp="2025-05-20 12:00:00",
                content="banking and budgeting",
                exchange_id=11,
            ),
            Message(
                medium="email",
                sender_id=1,
                receiver_ids=[0],
                timestamp="2025-05-20 12:00:01",
                content="random unrelated",
                exchange_id=12,
            ),
        ],
    )
    tm.join_published()

    t0 = time.perf_counter()
    nearest = tm._search_messages(references={"content": "banking"}, k=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert nearest and nearest[0].content
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")
