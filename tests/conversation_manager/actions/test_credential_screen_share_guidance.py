"""
tests/conversation_manager/actions/test_credential_screen_share_guidance.py
===========================================================================

Tests for the credential setup guidance flow during screen sharing.

Full scenario: The user wants the assistant to manage their Google Drive.
The assistant has already explained the credential + SDK approach. The user
is now on a screen share, starting from the Google Cloud homepage and asking
for step-by-step guidance on creating the necessary credentials.

The screenshot shows the Google Cloud homepage (cloud.google.com) — the very
beginning of the credential creation journey.  From here the user still needs
to: sign in, create/select a project, enable the Drive API, navigate to IAM,
create a service account, generate a JSON key, and finally share it via the
console's Secrets page.  That's many steps ahead, which makes dispatching
``act`` for full documentation clearly justified.

This is an eval test — it exercises real LLM behavior with a realistic
multi-turn conversation context.
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
    UserScreenShareStarted,
)
from unity.conversation_manager.types import ScreenshotEntry

pytestmark = pytest.mark.eval

GCP_HOMEPAGE_IMAGE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "images" / "gcp_homepage.jpg"
)


async def _build_conversation_context(cm):
    """Minimal context: user wants Google Drive managed and jumps straight
    to screen sharing on the GCP homepage. The assistant has NOT laid out
    any steps — the user is at absolute square one.
    """
    await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "I want you to manage my Google Drive. I've gone to "
                "cloud.google.com to try and get the credentials you'll need "
                "but I have no idea what to do. Let me share my screen."
            ),
        ),
    )

    # Activate screen share
    await cm.step(UserScreenShareStarted(reason=""), run_llm=False)

    # Clear tool call history from setup turn
    cm.all_tool_calls.clear()


@pytest.mark.asyncio
@_handle_project
async def test_credential_guidance_with_full_conversation_context(initialized_cm):
    """After a full conversation about Google Drive setup, user shares screen
    on the Google Cloud homepage and asks for step-by-step guidance.

    The screenshot shows cloud.google.com — the very start of the journey.
    From here the user needs to sign in, create a project, enable Drive API,
    create a service account, and download credentials.  That's many steps
    the CM brain can't walk through from memory alone.

    Expected behavior:
    - Reply with a best-guess first step based on what it sees
      (e.g., "you'll want to sign in first" or "click Get started")
    - Dispatch act to pull the full step-by-step instructions
    """
    cm = initialized_cm

    await _build_conversation_context(cm)

    assert (
        GCP_HOMEPAGE_IMAGE_PATH.exists()
    ), f"Test image not found: {GCP_HOMEPAGE_IMAGE_PATH}"
    gcp_b64 = base64.b64encode(GCP_HOMEPAGE_IMAGE_PATH.read_bytes()).decode()

    cm.cm._screenshot_buffer.append(
        ScreenshotEntry(
            b64=gcp_b64,
            utterance="This is the Google Cloud homepage. Where do I start?",
            timestamp=datetime.now(timezone.utc),
            source="user",
            filepath="Screenshots/User/gcp-homepage.jpg",
        ),
    )

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "OK I'm on the Google Cloud homepage now. I can see a "
                "'Get started for free' button and a 'Sign in' link. "
                "What's the first thing I need to do? Can you walk me "
                "through the whole process step by step?"
            ),
        ),
    )

    # --- Assert: CM replied (not silent) ---
    reply_events = filter_events_by_type(result.output_events, UnifyMessageSent)
    assert len(reply_events) >= 1, (
        "CM should reply to the user — not leave them waiting in silence "
        "while on a screen share"
    )

    reply_content = " ".join(e.content for e in reply_events).lower()

    # --- Assert: reply contains a best-guess first step ---
    has_first_step_guidance = any(
        term in reply_content
        for term in [
            "sign in",
            "get started",
            "log in",
            "account",
            "click",
            "first",
            "start",
        ]
    )
    assert has_first_step_guidance, (
        f"CM reply should suggest a concrete first step based on what it "
        f"sees (sign in, get started, etc.), not just defer entirely.\n"
        f"Full reply: {' '.join(e.content for e in reply_events)}"
    )

    # --- Assert: act was dispatched for full walkthrough ---
    # Check both output events and all_tool_calls — the act dispatch is
    # sometimes captured only in the tool call log due to event timing
    # in the test driver.
    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    act_dispatched = len(actor_events) >= 1 or "act" in cm.all_tool_calls
    assert act_dispatched, (
        "CM should dispatch act to pull full step-by-step instructions "
        "for creating Google Cloud credentials for Drive access. The user "
        "explicitly asked to be walked through the whole process, and "
        "there are many steps ahead (sign in → create project → enable "
        "API → create service account → download key). The CM can't "
        "reliably guide all of that from memory alone.\n"
        f"Reply was: {' '.join(e.content for e in reply_events)}\n"
        f"Tool calls: {cm.all_tool_calls}"
    )

    assert_efficient(result, 5)
