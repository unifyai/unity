from __future__ import annotations

import pytest
from datetime import datetime, UTC

from unity.transcript_manager.types.message import Message
from unity.image_manager.types import AnnotatedImageRefs, RawImageRef, AnnotatedImageRef
from tests.helpers import _handle_project


@_handle_project
def test_json_omits_empty_fields():
    """
    Before the Message serializer change, model_dump(mode="json") included
    empty containers such as images: [] and screen_share: {}. After the change,
    empty fields are pruned during JSON serialization.
    """

    # Message with default-empty images and screen_share → should omit both
    msg_empty = Message(
        medium="email",
        sender_id=0,
        receiver_ids=[1],
        timestamp=datetime.now(UTC),
        content="no images",
        exchange_id=4242,
    )
    dumped_empty = msg_empty.model_dump(mode="json", context={"prune_empty": True})
    assert "images" not in dumped_empty, "empty images field should be omitted"
    assert "screen_share" not in dumped_empty, "empty screen_share should be omitted"

    # Message with a non-empty images list → images must be preserved
    msg_with_image = Message(
        medium="email",
        sender_id=1,
        receiver_ids=[2],
        timestamp=datetime.now(UTC),
        content="with image",
        exchange_id=4243,
        images=AnnotatedImageRefs.model_validate(
            [
                AnnotatedImageRef(
                    raw_image_ref=RawImageRef(image_id=101),
                    annotation="ref",
                ),
            ],
        ),
    )
    dumped_with_image = msg_with_image.model_dump(
        mode="json",
        context={"prune_empty": True},
    )
    assert "images" in dumped_with_image, "non-empty images should be present"
    # screen_share remains empty here and should still be omitted
    assert "screen_share" not in dumped_with_image


@pytest.mark.asyncio
@_handle_project
async def test_ask_search_tool_omits_empty(static_now, monkeypatch):
    """
    Real tool-loop: TM.ask triggers search_messages on first step. Verify the
    tool result inserted into the transcript does not expose empty fields like
    images: [] or screen_share: {} to the LLM.
    """
    import json
    from unity.settings import SETTINGS
    from unity.transcript_manager.transcript_manager import TranscriptManager

    # Force search_messages on the first step so the assertion below is deterministic
    monkeypatch.setattr(SETTINGS, "FIRST_ASK_TOOL_IS_SEARCH", True)
    from unity.transcript_manager.types.message import Message

    tm = TranscriptManager()

    # Seed messages with no images/screen_share
    tm.log_messages(
        [
            Message(
                medium="email",
                sender_id=0,
                receiver_ids=[1],
                timestamp=static_now,
                content="banking and budgeting discussion",
                exchange_id=100,
            ),
            Message(
                medium="email",
                sender_id=1,
                receiver_ids=[0],
                timestamp=static_now,
                content="random unrelated",
                exchange_id=101,
            ),
        ],
    )
    tm.join_published()

    # Ask a question that keeps default policy (search_messages first)
    handle = await tm.ask(
        "Find anything about banking",
        _return_reasoning_steps=True,
    )
    answer, steps = await handle.result()

    # Locate the tool message for search_messages
    tool_msgs = [m for m in steps if isinstance(m, dict) and m.get("role") == "tool"]
    assert tool_msgs, "Expected at least one tool message in steps"
    search_msgs = [m for m in tool_msgs if m.get("name") == "search_messages"]
    assert search_msgs, "Expected a search_messages tool call in steps"

    # Parse its content (pretty-printed JSON) and check messages payload
    payload = json.loads(search_msgs[0].get("content", "{}"))
    assert isinstance(payload, dict) and "messages" in payload
    for msg in payload.get("messages", []):
        assert "images" not in msg, "empty images must be omitted in tool output"
        assert (
            "screen_share" not in msg
        ), "empty screen_share must be omitted in tool output"
