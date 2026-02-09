"""
tests/conversation_manager/voice/test_fast_brain_multilingual.py
================================================================

Tests that the fast brain (Voice Agent) matches the caller's language
even though its system prompt and any guidance are in English.

Uses the same ``get_fast_brain_response`` pattern as
``test_fast_brain_deferral.py`` — real LLM calls with cached responses.
"""

from __future__ import annotations

import re

import pytest

from unity.conversation_manager.prompt_builders import build_voice_agent_prompt

# Re-use the LLM helpers and model parameterisation from the deferral tests.
from tests.conversation_manager.voice.test_fast_brain_deferral import (
    get_fast_brain_response,
    FAST_BRAIN_MODELS,
)

pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
#  Language detection helpers (mirrors test_multilingual.py)
# ---------------------------------------------------------------------------

_SPANISH_MARKERS = [
    "hola",
    "gracias",
    "buenos",
    "cómo",
    "número",
    "necesito",
    "reunión",
    "confirmar",
    "información",
    "mañana",
    "también",
    "entendido",
    "perfecto",
    "encantado",
    "disculpe",
    "por favor",
    "bien",
    "buen",
    "mucho",
    "claro",
    "puedo",
    "puede",
    "momento",
    "revisar",
    "novedades",
    "propuesta",
    "cuento",
    "gusto",
    "hablar",
    "contigo",
    "ahora",
    "semana",
    "pasada",
    "enviar",
    "enviaron",
    "enviamos",
    "vamos",
    "dime",
    "cómo",
]

_FRENCH_MARKERS = [
    "bonjour",
    "bonsoir",
    "merci",
    "réunion",
    "monsieur",
    "madame",
    "confirmer",
    "demain",
    "également",
    "absolument",
    "certainement",
    "exactement",
    "cordialement",
    "actuellement",
    "malheureusement",
    "bien sûr",
    "s'il vous plaît",
    "de rien",
    "avec plaisir",
    "bonne journée",
    "enchanté",
    "d'accord",
    "bienvenue",
    "c'est",
    "c'était",
    "très",
    "plaisir",
    "heureux",
    "heureuse",
    "journée",
    "bonne",
    "aider",
    "besoin",
    "aujourd'hui",
]

_CJK_RE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def _has_spanish(text: str) -> bool:
    low = text.lower()
    hits = sum(1 for w in _SPANISH_MARKERS if w in low)
    return hits >= 2 or "¿" in text or "¡" in text


def _has_french(text: str) -> bool:
    low = text.lower()
    return sum(1 for w in _FRENCH_MARKERS if w in low) >= 2


def _has_japanese(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _has_arabic(text: str) -> bool:
    return bool(_ARABIC_RE.search(text))


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def voice_agent_prompt_boss():
    """Voice agent prompt for a call with the boss (contact_id=1)."""
    return build_voice_agent_prompt(
        bio="A helpful and efficient assistant.",
        boss_first_name="Alex",
        boss_surname="Thompson",
        boss_phone_number="+1-555-0100",
        boss_email_address="alex.thompson@example.com",
        is_boss_user=True,
        contact_rolling_summary=None,
    ).flatten()


@pytest.fixture
def voice_agent_prompt_external():
    """Voice agent prompt for a call with an external contact."""
    return build_voice_agent_prompt(
        bio="A helpful and efficient assistant.",
        boss_first_name="Alex",
        boss_surname="Thompson",
        boss_phone_number="+1-555-0100",
        boss_email_address="alex.thompson@example.com",
        is_boss_user=False,
        contact_first_name="María",
        contact_surname="García",
        contact_phone_number="+34-555-0200",
        contact_email="maria.garcia@example.com",
        contact_bio="External client from Madrid.",
        contact_rolling_summary=None,
    ).flatten()


@pytest.fixture(params=FAST_BRAIN_MODELS)
def fast_brain_model(request):
    """Parameterized fixture for fast brain model selection."""
    return request.param


# =====================================================================
#  Tests: Fast brain matches caller's language
# =====================================================================


@pytest.mark.asyncio
class TestFastBrainMatchesCallerLanguage:
    """
    The fast brain should respond in whatever language the caller uses,
    even though the system prompt is entirely in English.
    """

    async def test_responds_in_spanish_to_spanish_caller(
        self,
        voice_agent_prompt_boss,
        fast_brain_model,
    ):
        """Caller speaks Spanish -> fast brain replies in Spanish."""
        conversation = [
            {
                "role": "user",
                "content": (
                    "Hola, muchas gracias por atenderme. "
                    "Quería preguntarte cómo va el proyecto."
                ),
            },
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt_boss,
            conversation,
            model=fast_brain_model,
        )

        assert _has_spanish(response), (
            f"Fast brain ({fast_brain_model}) should reply in Spanish, "
            f"got: {response}"
        )

    async def test_responds_in_french_to_french_caller(
        self,
        voice_agent_prompt_boss,
        fast_brain_model,
    ):
        """Caller speaks French -> fast brain replies in French."""
        conversation = [
            {
                "role": "user",
                "content": (
                    "Bonjour ! Merci de prendre mon appel. "
                    "Comment avancent les choses de votre côté ?"
                ),
            },
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt_boss,
            conversation,
            model=fast_brain_model,
        )

        assert _has_french(response), (
            f"Fast brain ({fast_brain_model}) should reply in French, "
            f"got: {response}"
        )

    async def test_responds_in_japanese_to_japanese_caller(
        self,
        voice_agent_prompt_boss,
        fast_brain_model,
    ):
        """Caller speaks Japanese -> fast brain replies in Japanese."""
        conversation = [
            {
                "role": "user",
                "content": "こんにちは！プロジェクトの進捗はいかがですか？",
            },
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt_boss,
            conversation,
            model=fast_brain_model,
        )

        assert _has_japanese(response), (
            f"Fast brain ({fast_brain_model}) should reply in Japanese, "
            f"got: {response}"
        )

    async def test_responds_in_arabic_to_arabic_caller(
        self,
        voice_agent_prompt_boss,
        fast_brain_model,
    ):
        """Caller speaks Arabic -> fast brain replies in Arabic."""
        conversation = [
            {
                "role": "user",
                "content": ("مرحبا! شكراً على الرد. " "كيف تسير الأمور في المشروع؟"),
            },
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt_boss,
            conversation,
            model=fast_brain_model,
        )

        assert _has_arabic(response), (
            f"Fast brain ({fast_brain_model}) should reply in Arabic, "
            f"got: {response}"
        )


@pytest.mark.asyncio
class TestFastBrainLanguageWithExternalContact:
    """
    When the call is with an external contact (not the boss), the fast
    brain should still match the caller's language.
    """

    async def test_external_contact_spanish_call(
        self,
        voice_agent_prompt_external,
        fast_brain_model,
    ):
        """External contact speaks Spanish -> fast brain replies in Spanish."""
        conversation = [
            {
                "role": "user",
                "content": (
                    "Hola, soy María. Quería saber si hay novedades "
                    "sobre la propuesta que enviamos la semana pasada."
                ),
            },
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt_external,
            conversation,
            model=fast_brain_model,
        )

        assert _has_spanish(response), (
            f"Fast brain ({fast_brain_model}) should reply in Spanish "
            f"to external contact, got: {response}"
        )


@pytest.mark.asyncio
class TestFastBrainLanguageConsistencyAcrossTurns:
    """
    Once the caller establishes a language, the fast brain should
    maintain it across multiple turns — not switch back to English.
    """

    async def test_spanish_stays_spanish_over_two_turns(
        self,
        voice_agent_prompt_boss,
        fast_brain_model,
    ):
        """Two-turn Spanish conversation: both replies in Spanish."""
        # Turn 1
        conversation_t1 = [
            {
                "role": "user",
                "content": (
                    "Hola, muchas gracias por tu ayuda con el informe. "
                    "¡Quedó muy bien!"
                ),
            },
        ]

        response_t1 = await get_fast_brain_response(
            voice_agent_prompt_boss,
            conversation_t1,
            model=fast_brain_model,
        )

        assert _has_spanish(response_t1), (
            f"Fast brain ({fast_brain_model}) turn-1 should be Spanish, "
            f"got: {response_t1}"
        )

        # Turn 2: continue the conversation with turn-1 context
        conversation_t2 = [
            conversation_t1[0],
            {"role": "assistant", "content": response_t1},
            {
                "role": "user",
                "content": (
                    "¡Qué bueno! También quería decirte que el cliente "
                    "quedó muy contento con los resultados."
                ),
            },
        ]

        response_t2 = await get_fast_brain_response(
            voice_agent_prompt_boss,
            conversation_t2,
            model=fast_brain_model,
        )

        assert _has_spanish(response_t2), (
            f"Fast brain ({fast_brain_model}) turn-2 should stay Spanish, "
            f"got: {response_t2}"
        )


@pytest.mark.asyncio
class TestFastBrainLanguageWithEnglishGuidance:
    """
    The fast brain receives guidance (notifications) in English from the
    slow brain.  It should integrate the information but continue
    speaking the caller's language.
    """

    async def test_spanish_caller_english_guidance_stays_spanish(
        self,
        voice_agent_prompt_boss,
        fast_brain_model,
    ):
        """
        Caller speaks Spanish, English guidance arrives -> reply in Spanish.

        The notification contains English data. The fast brain should
        translate/paraphrase it into Spanish when relaying to the caller.
        """
        conversation = [
            {
                "role": "user",
                "content": ("Hola, ¿tienes información sobre la reunión " "de mañana?"),
            },
            {
                "role": "assistant",
                "content": "Déjame verificar eso por ti.",
            },
            {
                # English guidance from slow brain (notification)
                "role": "user",
                "content": (
                    "[notification] The meeting tomorrow is confirmed "
                    "for 3pm in Conference Room B."
                ),
            },
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt_boss,
            conversation,
            model=fast_brain_model,
        )

        assert _has_spanish(response), (
            f"Fast brain ({fast_brain_model}) should relay English guidance "
            f"in Spanish to the caller, got: {response}"
        )

    async def test_japanese_caller_english_guidance_stays_japanese(
        self,
        voice_agent_prompt_boss,
        fast_brain_model,
    ):
        """
        Caller speaks Japanese, English guidance arrives -> reply in Japanese.
        """
        conversation = [
            {
                "role": "user",
                "content": "明日の会議の情報はありますか？",
            },
            {
                "role": "assistant",
                "content": "確認しますので少々お待ちください。",
            },
            {
                # English guidance from slow brain (notification)
                "role": "user",
                "content": (
                    "[notification] The meeting tomorrow is confirmed "
                    "for 3pm in Conference Room B."
                ),
            },
        ]

        response = await get_fast_brain_response(
            voice_agent_prompt_boss,
            conversation,
            model=fast_brain_model,
        )

        assert _has_japanese(response), (
            f"Fast brain ({fast_brain_model}) should relay English guidance "
            f"in Japanese to the caller, got: {response}"
        )
