"""
tests/conversation_manager/voice/test_fast_brain_meet_controls.py
==================================================================

Eval tests verifying the fast brain correctly distinguishes between
Meet window controls and console UI elements.

Production issue: when the user reported difficulty controlling the
assistant's screen during a Unify Meet call, the fast brain incorrectly
told them to undock the Meet window to access the "Show assistant screen"
and "Enable mouse and keyboard control" buttons — claiming these lived
on the console behind the Meet overlay. In reality, these buttons are
inside the Meet window's own bottom bar and are always visible.

The fast brain should only recommend undocking the Meet window when
directing the user to an actual console page (Profile, Resources, Chat,
Billing, etc.), never for controls that are part of the Meet UI itself.
"""

from __future__ import annotations

import re

import pytest

from tests.helpers import _handle_project
from unity.conversation_manager.prompt_builders import build_voice_agent_prompt
from unity.settings import SETTINGS

from tests.conversation_manager.voice.test_fast_brain_deferral import (
    get_fast_brain_response,
)

pytestmark = pytest.mark.eval

MODEL = SETTINGS.conversation.FAST_BRAIN_MODEL

UNDOCK_PATTERNS = [
    r"undock",
    r"glove icon",
    r"drag.{0,20}(meet|window|aside|side)",
    r"move.{0,20}(meet|window|aside|side)",
    r"(console|profile|resources|chat).{0,30}(hidden|behind|covered|blocked)",
    r"(behind|under).{0,20}(meet|overlay|window)",
]

INIT_SYSTEM_MESSAGE = (
    "[system] You have just started up and your systems are still "
    "syncing — loading files, pulling up previous conversations, "
    "and connecting to your tools. This takes a few moments. "
    "If the user asks you to do something that requires looking "
    "things up or taking action, let them know naturally that "
    "you are still getting set up (e.g. 'I'm just pulling up our "
    "previous sessions — give me a moment and I'll get right on "
    "that'). Do NOT say 'I can't do that' — frame it as a brief "
    "delay, not a limitation. You will receive a notification "
    "when everything is ready."
)


def _mentions_undocking(response: str) -> bool:
    low = response.lower()
    return any(re.search(p, low) for p in UNDOCK_PATTERNS)


def _build_meet_prompt() -> str:
    return build_voice_agent_prompt(
        bio="A helpful and efficient assistant.",
        assistant_name="David Miller",
        boss_first_name="Dan",
        boss_surname="Lenton",
        boss_email_address="user@example.com",
        is_boss_user=True,
        contact_rolling_summary=None,
        channel="unify_meet",
    ).flatten()


@pytest.mark.asyncio
@_handle_project
async def test_does_not_suggest_undocking_for_screen_control_issue():
    """When the user reports they can't control the assistant's screen,
    the fast brain should NOT tell them to undock the Meet window.

    The "Show assistant screen" and "Enable mouse and keyboard control"
    buttons are inside the Meet window's bottom bar, not on the console.
    """
    prompt = _build_meet_prompt()
    conversation = [
        {"role": "system", "content": INIT_SYSTEM_MESSAGE},
        {"role": "assistant", "content": "Hey Dan — how can I help?"},
        {
            "role": "user",
            "content": (
                "Hey, your computer's still not set up. "
                "I'm unable to control your screen."
            ),
        },
    ]

    response = await get_fast_brain_response(prompt, conversation, model=MODEL)

    assert not _mentions_undocking(response), (
        f"Fast brain told the user to undock the Meet window for a screen "
        f"control issue. The 'Show assistant screen' and 'Enable mouse and "
        f"keyboard control' buttons are inside the Meet window — undocking "
        f"is not needed.\nResponse: {response}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_does_not_suggest_undocking_for_assistant_screen_button():
    """When the user asks about the 'Show assistant screen' button,
    the fast brain should NOT direct them to undock the Meet window.
    """
    prompt = _build_meet_prompt()
    conversation = [
        {"role": "assistant", "content": "Hey Dan — how can I help?"},
        {
            "role": "user",
            "content": "Where's the button to show your screen?",
        },
    ]

    response = await get_fast_brain_response(prompt, conversation, model=MODEL)

    assert not _mentions_undocking(response), (
        f"Fast brain told the user to undock the Meet window to find the "
        f"'Show assistant screen' button. This button is in the Meet's "
        f"own bottom bar — always visible during a call.\n"
        f"Response: {response}"
    )

    low = response.lower()
    assert "bottom" in low or "bar" in low or "meet" in low, (
        f"Fast brain should reference the Meet's bottom bar when directing "
        f"the user to the 'Show assistant screen' button.\n"
        f"Response: {response}"
    )
