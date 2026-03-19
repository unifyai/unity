"""
tests/conversation_manager/voice/test_fast_brain_init_awareness.py
==================================================================

Eval tests verifying the fast brain (Voice Agent) handles the
initialization window gracefully.

When a call starts during a cold start, the conversation manager may
still be initializing.  The fast brain receives a system message about
this and should:

1. Acknowledge action requests naturally and defer, rather than
   hallucinating results or ignoring the request.
2. After receiving the initialization-complete notification, treat
   subsequent requests normally (no unnecessary hedging).
"""

from __future__ import annotations

import re

import pytest

from tests.helpers import _handle_project
from unity.conversation_manager.prompt_builders import build_voice_agent_prompt
from unity.conversation_manager.livekit_unify_adapter import UnifyLLM
from unity.settings import SETTINGS
from livekit.agents import llm

pytestmark = pytest.mark.eval

MODEL = SETTINGS.conversation.FAST_BRAIN_MODEL

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

INIT_COMPLETE_NOTIFICATION = (
    "[notification] Initialization complete — all actions are now "
    "available. Full conversation history has been loaded."
)

DEFERRAL_PATTERNS = [
    r"let me check",
    r"let me look",
    r"i'll check",
    r"i'll look",
    r"checking",
    r"looking into",
    r"one moment",
    r"just a moment",
    r"give me a (moment|second|sec)",
    r"hold on",
    r"let me find",
    r"i'll find",
    r"let me see",
    r"let me.{0,10}(verify|confirm|get)",
    r"i'll get (on|to|right on) (it|that)",
    r"(getting|setting|starting).{0,15}(up|ready|going)",
    r"(a |few |couple( of)? )?minutes",
    r"i'll (let you know|update you|get back to you)",
    r"(i'll|let me) (work|get) on that",
    r"(soon|shortly|in a moment|momentarily)",
    r"bear with me",
    r"right on it",
    r"on it",
    # Startup-specific: explains *why* there's a delay
    r"(pulling|loading|syncing).{0,20}(up|files|sessions|conversations|data)",
    r"still (setting|getting|booting|starting|spinning|warming) up",
    r"still (syncing|loading|connecting)",
    r"systems? (are |is )?(still )?(coming|loading|syncing|starting)",
]


def _has_deferral(response: str) -> bool:
    low = response.lower().replace("\u2018", "'").replace("\u2019", "'")
    return any(re.search(p, low) for p in DEFERRAL_PATTERNS)


def _build_prompt() -> str:
    return build_voice_agent_prompt(
        bio="A helpful and efficient assistant.",
        boss_first_name="Alex",
        boss_surname="Thompson",
        boss_phone_number="+1-555-0100",
        boss_email_address="alex.thompson@example.com",
        is_boss_user=True,
        contact_rolling_summary=None,
    ).flatten()


async def _get_response(
    system_prompt: str,
    messages: list[dict[str, str]],
) -> str:
    llm_instance = UnifyLLM(model=MODEL, reasoning_effort="low")
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content=system_prompt)
    for msg in messages:
        chat_ctx.add_message(role=msg["role"], content=msg["content"])
    chat_ctx.add_message(
        role="user",
        content="Respond as the assistant. Keep it concise and conversational (voice call).",
    )
    stream = llm_instance.chat(chat_ctx=chat_ctx)
    parts = []
    try:
        async for chunk in stream:
            if hasattr(chunk, "delta") and chunk.delta:
                content = getattr(chunk.delta, "content", None)
                if content:
                    parts.append(content)
    finally:
        await stream.aclose()
    return "".join(parts)


# ---------------------------------------------------------------------------
#  Pre-init: fast brain should defer action requests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_fast_brain_defers_action_during_init():
    """When the initializing system message is present, the fast brain
    should defer action requests with natural acknowledgment language
    rather than claiming it cannot help or hallucinating a result.
    """
    prompt = _build_prompt()
    messages = [
        {"role": "system", "content": INIT_SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": "Can you check my calendar and tell me what I have today?",
        },
    ]

    response = await _get_response(prompt, messages)

    assert _has_deferral(response), (
        f"Fast brain should defer action requests during initialization "
        f"with natural language like 'let me check' or 'one moment'.\n"
        f"Response: {response}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_fast_brain_handles_greeting_during_init():
    """When the initializing system message is present but the user just
    says hello, the fast brain should greet back normally — no need to
    mention initialization or defer anything.
    """
    prompt = _build_prompt()
    messages = [
        {"role": "system", "content": INIT_SYSTEM_MESSAGE},
        {"role": "user", "content": "Hey, good morning!"},
    ]

    response = await _get_response(prompt, messages)

    low = response.lower()
    assert not any(
        kw in low for kw in ["initializing", "setting up", "not available"]
    ), (
        f"Fast brain should greet normally without mentioning initialization.\n"
        f"Response: {response}"
    )


# ---------------------------------------------------------------------------
#  Post-init: fast brain should behave normally
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_fast_brain_normal_after_init_complete():
    """After the init-complete notification, the fast brain should handle
    action requests without unnecessary hedging about initialization.
    """
    prompt = _build_prompt()
    messages = [
        {"role": "system", "content": INIT_SYSTEM_MESSAGE},
        {"role": "user", "content": "Hey, good morning!"},
        {
            "role": "assistant",
            "content": "Good morning, Alex! How can I help you today?",
        },
        {"role": "system", "content": INIT_COMPLETE_NOTIFICATION},
        {
            "role": "user",
            "content": "Can you check my calendar and tell me what I have today?",
        },
    ]

    response = await _get_response(prompt, messages)

    low = response.lower()
    assert not any(
        kw in low for kw in ["still initializing", "not available yet", "setting up"]
    ), (
        f"Fast brain should NOT mention initialization after init-complete "
        f"notification.\nResponse: {response}"
    )
