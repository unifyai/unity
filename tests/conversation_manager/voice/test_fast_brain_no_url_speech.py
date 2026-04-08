"""
tests/conversation_manager/voice/test_fast_brain_no_url_speech.py
=================================================================

Tests for URL handling on voice calls: the fast brain must distinguish between
simple URLs (which should be spoken phonetically) and complex URLs (which must
only be shared via chat).

- **Simple URLs** like ``console.cloud.google.com`` should be spoken
  phonetically ("console dot cloud dot google dot com") and also pasted
  in the chat with ``https://`` for clickability.
- **Complex URLs** like OAuth scope lists or deep paths with query params
  must NOT be spoken — they produce garbled TTS audio.
- API keys, OAuth scopes, tokens, and other machine-readable strings must
  never be spoken.

Origin: production incident 2026-04-07 (OAuth scope dictation) and
2026-04-08 (simple URL refused when it should have been spoken phonetically).
"""

from __future__ import annotations

import re

import pytest

from livekit.agents import llm

from unity.conversation_manager.livekit_unify_adapter import UnifyLLM
from unity.conversation_manager.prompt_builders import build_voice_agent_prompt
from unity.settings import SETTINGS

pytestmark = pytest.mark.eval

MODEL_TTS = SETTINGS.conversation.FAST_BRAIN_MODEL

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)

SEND_VIA_TEXT_PATTERNS = [
    r"send.{0,30}(chat|message|text)",
    r"paste.{0,30}(chat|message|text)",
    r"put.{0,30}(chat|message|text)",
    r"(chat|message|text).{0,20}(for you|to copy|to paste)",
    r"i'll (send|paste|put|drop|pop)",
    r"in the chat",
    r"in a message",
    r"(send|share|drop|pop).{0,15}(over|those|them|it|that)",
]

PHONETIC_URL_PATTERN = re.compile(
    r"\b\w+\s+dot\s+\w+",
    re.IGNORECASE,
)


def response_contains_url(response: str) -> list[str]:
    return URL_PATTERN.findall(response)


def response_contains_phonetic_url(response: str) -> bool:
    """Check if the response contains a phonetically spelled-out URL."""
    return bool(PHONETIC_URL_PATTERN.search(response))


def response_promises_text(response: str) -> bool:
    lower = response.lower()
    lower = lower.replace("\u2018", "'").replace("\u2019", "'")
    return any(re.search(p, lower) for p in SEND_VIA_TEXT_PATTERNS)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def voice_agent_prompt():
    return build_voice_agent_prompt(
        bio="A helpful and efficient assistant.",
        boss_first_name="Dan",
        boss_surname="Lenton",
        boss_phone_number="+44-7700-900000",
        boss_email_address="dan@unify.ai",
        is_boss_user=True,
        contact_rolling_summary=None,
    ).flatten()


FAST_BRAIN_MODELS = [
    pytest.param(MODEL_TTS, id="tts-" + MODEL_TTS.split("@")[0]),
]


@pytest.fixture(params=FAST_BRAIN_MODELS)
def fast_brain_model(request):
    return request.param


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def get_fast_brain_response(
    system_prompt: str,
    conversation: list[dict[str, str]],
    model: str = MODEL_TTS,
) -> str:
    llm_instance = UnifyLLM(model=model, reasoning_effort="low")

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content=system_prompt)

    for msg in conversation:
        chat_ctx.add_message(role=msg["role"], content=msg["content"])

    chat_ctx.add_message(
        role="user",
        content="Respond as the assistant. Keep it concise and conversational (voice call).",
    )

    stream = llm_instance.chat(chat_ctx=chat_ctx)
    parts: list[str] = []
    try:
        async for chunk in stream:
            if hasattr(chunk, "delta") and chunk.delta:
                content = getattr(chunk.delta, "content", None)
                if content:
                    parts.append(content)
    finally:
        await stream.aclose()

    return "".join(parts)


# ===========================================================================
# Tests
# ===========================================================================


@pytest.mark.asyncio
class TestNoURLSpeech:
    """Complex URLs / machine-readable strings must not be spoken.

    Simple, short URLs should be spoken phonetically instead.
    """

    async def test_simple_url_spoken_phonetically(
        self,
        voice_agent_prompt,
        fast_brain_model,
    ):
        """A simple domain URL should be spoken phonetically on the call,
        e.g. 'console dot cloud dot google dot com'."""
        conversation = [
            {
                "role": "user",
                "content": (
                    "Where do I go to set up the Google Cloud project? "
                    "What's the URL?"
                ),
            },
            {
                "role": "system",
                "content": (
                    "[notification] The Google Cloud Console is at "
                    "console.cloud.google.com"
                ),
            },
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        assert response_contains_phonetic_url(response), (
            f"Fast brain ({fast_brain_model}) did not speak the simple URL "
            f"phonetically!\n"
            f"Response: {response}\n\n"
            f"Simple URLs like 'console.cloud.google.com' should be spoken "
            f"phonetically as 'console dot cloud dot google dot com'."
        )

        raw_urls = response_contains_url(response)
        assert not raw_urls, (
            f"Fast brain ({fast_brain_model}) included a raw URL in speech!\n"
            f"Response: {response}\n"
            f"URLs found: {raw_urls}\n\n"
            f"Even simple URLs should be spoken phonetically (with 'dot'), "
            f"not as raw https:// strings."
        )

    async def test_oauth_scopes_walkthrough(
        self,
        voice_agent_prompt,
        fast_brain_model,
    ):
        """When guiding through OAuth setup, the fast brain should offer to
        send scopes via text rather than dictating URLs."""
        conversation = [
            {
                "role": "user",
                "content": (
                    "I'm setting up domain-wide delegation in Google Workspace "
                    "for you. I've created the service account and copied the "
                    "Client ID. Now I need to paste the OAuth scopes. "
                    "What scopes do I need?"
                ),
            },
            {
                "role": "system",
                "content": (
                    "[notification] The required OAuth scopes for Google "
                    "Workspace domain-wide delegation are:\n"
                    "https://mail.google.com/,"
                    "https://www.googleapis.com/auth/drive,"
                    "https://www.googleapis.com/auth/admin.directory.user,"
                    "https://www.googleapis.com/auth/admin.directory.group,"
                    "https://www.googleapis.com/auth/calendar,"
                    "https://www.googleapis.com/auth/contacts,"
                    "https://www.googleapis.com/auth/gmail.settings.basic,"
                    "https://www.googleapis.com/auth/cloud-platform"
                ),
            },
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        urls = response_contains_url(response)
        assert not urls, (
            f"Fast brain ({fast_brain_model}) tried to speak URLs!\n"
            f"Response: {response}\n"
            f"URLs found: {urls}\n\n"
            f"URLs produce garbled/unintelligible TTS audio. "
            f"The fast brain should send them via text instead."
        )

        assert response_promises_text(response), (
            f"Fast brain ({fast_brain_model}) didn't offer to send via text.\n"
            f"Response: {response}\n\n"
            f"When information contains URLs, the fast brain should tell the "
            f"caller it will send the details in the chat."
        )

    async def test_api_key_notification(
        self,
        voice_agent_prompt,
        fast_brain_model,
    ):
        """When a notification contains an API key / URL, the fast brain
        should not try to read it aloud."""
        conversation = [
            {
                "role": "user",
                "content": "Can you give me the API endpoint and key for the staging server?",
            },
            {
                "role": "system",
                "content": (
                    "[notification] Staging API endpoint: "
                    "https://api.staging.example.com/v2/graphql  "
                    "API key: sk-proj-a1b2c3d4e5f6g7h8i9j0"
                ),
            },
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        urls = response_contains_url(response)
        assert not urls, (
            f"Fast brain ({fast_brain_model}) tried to speak a URL!\n"
            f"Response: {response}\n"
            f"URLs found: {urls}"
        )

        api_key_in_speech = "sk-proj" in response.lower()
        assert not api_key_in_speech, (
            f"Fast brain ({fast_brain_model}) tried to speak an API key!\n"
            f"Response: {response}"
        )

    async def test_long_url_in_search_result(
        self,
        voice_agent_prompt,
        fast_brain_model,
    ):
        """When relaying a search result with a long URL, the fast brain
        should summarise verbally and offer to send the link via text."""
        conversation = [
            {
                "role": "user",
                "content": "Can you find the documentation for setting up Google OAuth?",
            },
            {
                "role": "system",
                "content": (
                    "[notification] Found the official guide: "
                    "https://developers.google.com/identity/protocols/"
                    "oauth2/service-account#delegatingauthority — "
                    "it covers creating a service account, enabling "
                    "domain-wide delegation, and authorising scopes."
                ),
            },
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt,
            conversation,
            model=fast_brain_model,
        )

        urls = response_contains_url(response)
        assert not urls, (
            f"Fast brain ({fast_brain_model}) tried to speak a URL!\n"
            f"Response: {response}\n"
            f"URLs found: {urls}\n\n"
            f"The fast brain should describe the result verbally and "
            f"offer to send the link in the chat."
        )
