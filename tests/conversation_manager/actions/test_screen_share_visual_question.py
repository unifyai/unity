"""
tests/conversation_manager/actions/test_screen_share_visual_question.py
========================================================================

Tests that the CM slow brain answers simple visual questions about a shared
screen directly — without deferring to ``act``.

When screen sharing is active and screenshots are in the buffer, the CM LLM
receives the image as part of its context. Basic visual questions ("what can
you see on my screen?") should be answered inline, not delegated to the Actor.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    assert_efficient,
    filter_events_by_type,
)
from tests.conversation_manager.conftest import BOSS
from unity.conversation_manager.events import (
    UnifyMessageReceived,
    UnifyMessageSent,
    ActorHandleStarted,
)
from unity.conversation_manager.cm_types import ScreenshotEntry

pytestmark = pytest.mark.eval

GRUB_IMAGE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "images" / "grub_screen.jpg"
)


@pytest.mark.asyncio
@_handle_project
async def test_visual_question_answered_without_act(initialized_cm):
    """User asks what's on their shared screen — CM should answer directly.

    The CM LLM receives the screenshot as an image in its context. A simple
    visual question like "what can you see on my screen?" should be answered
    inline, not delegated via act().
    """
    cm = initialized_cm
    cm.cm.user_screen_share_active = True

    assert GRUB_IMAGE_PATH.exists(), f"Test image not found: {GRUB_IMAGE_PATH}"
    grub_b64 = base64.b64encode(GRUB_IMAGE_PATH.read_bytes()).decode()

    cm.cm._screenshot_buffer.append(
        ScreenshotEntry(
            b64=grub_b64,
            utterance="What can you see on my screen?",
            timestamp=datetime.now(timezone.utc),
            source="user",
        ),
    )

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content="What can you see on my screen?",
        ),
    )

    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert len(actor_events) == 0, (
        f"Simple visual question should NOT trigger act — the CM can see the "
        f"image directly. Got {len(actor_events)} ActorHandleStarted event(s) "
        f"with queries: {[e.query for e in actor_events]}"
    )

    reply_events = filter_events_by_type(result.output_events, UnifyMessageSent)
    assert (
        len(reply_events) >= 1
    ), "CM should have sent a reply describing what it sees on screen"

    assert_efficient(result, 3)
