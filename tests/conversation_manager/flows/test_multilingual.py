"""
tests/conversation_manager/flows/test_multilingual.py
=========================================================

Tests for multilingual conversation handling.

Verifies that the ConversationManager:

1. Responds to contacts in their language (detected from inbound messages)
2. Keeps all internal operations in English:
   - Queries delegated to ``act``
   - Messages sent to other (English-speaking) contacts
   - Confirmations / summaries sent to the boss
"""

import re

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    filter_events_by_type,
    get_exactly_one,
)
from tests.conversation_manager.conftest import TEST_CONTACTS, BOSS
from unity.conversation_manager.events import (
    SMSReceived,
    SMSSent,
    EmailReceived,
    EmailSent,
    UnifyMessageReceived,
    UnifyMessageSent,
    ActorHandleStarted,
)

pytestmark = pytest.mark.eval

# Convenience contact references
ALICE = TEST_CONTACTS[0]  # contact_id 2
BOB = TEST_CONTACTS[1]  # contact_id 3


# ---------------------------------------------------------------------------
#  Language detection helpers
# ---------------------------------------------------------------------------

# Spanish markers — distinctive words unlikely to appear in English text.
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
]

# French markers — distinctive words and contractions unlikely to appear
# in English text.  Includes both formal and conversational vocabulary so
# that natural LLM responses ("De rien, c'était avec plaisir") are caught.
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
_NON_LATIN_RE = re.compile(
    r"[\u0600-\u06FF\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u0400-\u04FF\uAC00-\uD7AF]",
)


def _has_spanish(text: str) -> bool:
    """True if *text* contains clear Spanish language indicators."""
    low = text.lower()
    hits = sum(1 for w in _SPANISH_MARKERS if w in low)
    return hits >= 2 or "¿" in text or "¡" in text


def _has_french(text: str) -> bool:
    """True if *text* contains clear French language indicators."""
    low = text.lower()
    return sum(1 for w in _FRENCH_MARKERS if w in low) >= 2


def _has_japanese(text: str) -> bool:
    """True if *text* contains hiragana, katakana, or kanji."""
    return bool(_CJK_RE.search(text))


def _has_arabic(text: str) -> bool:
    """True if *text* contains Arabic script characters."""
    return bool(_ARABIC_RE.search(text))


def _is_english(text: str) -> bool:
    """True if *text* appears to be written in English.

    Checks that there are no significant non-English markers: non-Latin
    script characters, or multiple distinctive Spanish / French words.
    """
    if _NON_LATIN_RE.search(text):
        return False
    if _has_spanish(text):
        return False
    if _has_french(text):
        return False
    return True


# =====================================================================
#  Group 1 — Response matches the sender's language
#
#  Messages are deliberately simple greetings and thank-yous that
#  require nothing more than a polite reply — no calendar lookups, no
#  information retrieval, no action delegation.
# =====================================================================


@pytest.mark.asyncio
@_handle_project
async def test_spanish_sms_reply_in_spanish(initialized_cm):
    """Alice texts in Spanish -> assistant replies in Spanish."""
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=ALICE,
            content=(
                "¡Hola! Muchas gracias por tu ayuda ayer, fue muy útil. "
                "Espero que tengas un excelente día."
            ),
        ),
    )

    sms = get_exactly_one(result.output_events, SMSSent)
    assert _has_spanish(sms.content), (
        f"Expected Spanish reply to Spanish-speaking Alice, got: {sms.content}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_french_email_reply_in_french(initialized_cm):
    """Alice emails in French -> assistant replies in French."""
    cm = initialized_cm

    result = await cm.step_until_wait(
        EmailReceived(
            contact=ALICE,
            subject="Remerciements",
            body=(
                "Bonjour,\n\n"
                "Je voulais simplement vous remercier pour votre aide "
                "la semaine dernière. Tout s'est très bien passé grâce "
                "à vous.\n\n"
                "Bonne journée !"
            ),
            email_id="french_email_1",
        ),
    )

    email = get_exactly_one(result.output_events, EmailSent)
    assert _has_french(email.body), (
        f"Expected French reply to French-speaking Alice, got: {email.body}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_japanese_unify_message_reply_in_japanese(initialized_cm):
    """Alice sends a Unify message in Japanese -> reply contains Japanese."""
    cm = initialized_cm

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=ALICE,
            content=(
                "こんにちは！先日はお手伝いいただきありがとうございました。"
                "おかげさまでとても助かりました。良い一日をお過ごしください。"
            ),
        ),
    )

    msg = get_exactly_one(result.output_events, UnifyMessageSent)
    assert _has_japanese(msg.content), (
        f"Expected Japanese reply to Japanese-speaking Alice, got: {msg.content}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_arabic_sms_reply_in_arabic(initialized_cm):
    """Alice texts in Arabic -> reply contains Arabic script."""
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=ALICE,
            content=(
                "مرحبا! شكراً جزيلاً على مساعدتك بالأمس. "
                "كان ذلك مفيداً جداً. أتمنى لك يوماً سعيداً."
            ),
        ),
    )

    sms = get_exactly_one(result.output_events, SMSSent)
    assert _has_arabic(sms.content), (
        f"Expected Arabic reply to Arabic-speaking Alice, got: {sms.content}"
    )


# =====================================================================
#  Group 2 — Act queries stay in English
#
#  Alice's messages are simple informational statements (no action
#  needed). The boss then explicitly asks to contact an unknown person
#  (David), which is what forces the ``act`` delegation.
# =====================================================================


@pytest.mark.asyncio
@_handle_project
async def test_act_query_english_when_boss_speaks_spanish(initialized_cm):
    """
    Boss gives instructions in Spanish -> act query must still be English.

    The boss speaks Spanish and asks to email an unknown person (David).
    Even though the entire conversation is in Spanish, the act query
    delegated to the Actor must be in English — act is an internal
    interface, not a user-facing message.
    """
    cm = initialized_cm

    # Alice sends a Spanish informational message (no action required)
    await cm.step_until_wait(
        SMSReceived(
            contact=ALICE,
            content=(
                "Hola, solo quería avisarte que David me pidió que te "
                "dijera que necesita hablar contigo. Dice que es urgente."
            ),
        ),
    )

    # Boss gives instruction in Spanish -> triggers act for unknown David
    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "Envíale un correo electrónico a David sobre lo que "
                "mencionó Alice"
            ),
        ),
    )

    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called (ActorHandleStarted), "
        f"got: {[type(e).__name__ for e in result.output_events]}"
    )

    query = actor_events[0].query
    assert _is_english(query), f"Act query should be in English, got: {query}"


@pytest.mark.asyncio
@_handle_project
async def test_act_query_english_when_boss_speaks_japanese(initialized_cm):
    """
    Boss gives instructions in Japanese -> act query must still be English.

    The boss speaks Japanese and asks to email an unknown person (David).
    Even though the entire conversation is in Japanese, the act query
    must be in English with no CJK character leakage.
    """
    cm = initialized_cm

    # Alice sends a Japanese informational message (no action required)
    await cm.step_until_wait(
        UnifyMessageReceived(
            contact=ALICE,
            content=(
                "こんにちは。デビッドさんから伝言です。"
                "あなたに連絡を取りたいそうです。急ぎだそうです。"
            ),
        ),
    )

    # Boss gives instruction in Japanese -> triggers act for unknown David
    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="アリスが言っていた件について、デビッドにメールを送ってください",
        ),
    )

    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called (ActorHandleStarted), "
        f"got: {[type(e).__name__ for e in result.output_events]}"
    )

    query = actor_events[0].query
    assert not _has_japanese(query), (
        f"Act query must not contain Japanese characters, got: {query}"
    )
    assert _is_english(query), f"Act query should be in English, got: {query}"


# =====================================================================
#  Group 3 — Cross-contact language isolation
#
#  Alice shares simple factual information (statements, not questions).
#  No lookups needed — the assistant already has everything it needs
#  to acknowledge Alice and later relay the info to Bob or the boss.
# =====================================================================


@pytest.mark.asyncio
@_handle_project
async def test_relay_to_bob_in_english_despite_spanish_source(initialized_cm):
    """
    Alice speaks Spanish, boss relays to Bob -> Bob receives English.

    The assistant must translate / paraphrase Alice's Spanish content into
    English when forwarding it to Bob, an English-speaking contact.
    """
    cm = initialized_cm

    # Alice sends Spanish informational message (no action needed)
    await cm.step_until_wait(
        SMSReceived(
            contact=ALICE,
            content=(
                "¡Hola! Solo quería contarte que la cena del viernes será "
                "a las 7 de la noche en el restaurante italiano. ¡Nos vemos allí!"
            ),
        ),
    )

    # Bob sends a simple English greeting (establishes active conversation
    # without mentioning the dinner — we don't want the model to pre-relay
    # Alice's info before the boss asks)
    await cm.step_until_wait(
        SMSReceived(
            contact=BOB,
            content="Hey, just checking in. Hope you're having a good day!",
        ),
    )

    # Boss asks to relay Alice's dinner info to Bob (explicit send instruction)
    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Send Bob a text with what Alice said about the Friday dinner",
        ),
    )

    sms_events = filter_events_by_type(result.output_events, SMSSent)
    bob_sms = [
        s for s in sms_events if s.contact["contact_id"] == BOB["contact_id"]
    ]
    assert len(bob_sms) >= 1, (
        f"Expected SMS to Bob, got SMS to: "
        f"{[s.contact['contact_id'] for s in sms_events]}"
    )

    bob_msg = bob_sms[0].content
    assert _is_english(bob_msg), (
        f"Message to English-speaking Bob should be in English, got: {bob_msg}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_boss_gets_english_summary_of_spanish_message(initialized_cm):
    """
    Alice messages in Spanish, boss asks what she said -> English summary.

    The boss always communicates in English. The response should be an
    English paraphrase / summary, not a copy of Alice's Spanish text.
    """
    cm = initialized_cm

    # Alice sends Spanish informational message (no action needed)
    await cm.step_until_wait(
        SMSReceived(
            contact=ALICE,
            content=(
                "Buenos días. Solo quería informarle que el informe trimestral "
                "ya está terminado y lo envié esta mañana. ¡Gracias por todo!"
            ),
        ),
    )

    # Boss asks about Alice's message
    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="What did Alice just say?",
        ),
    )

    sms_events = filter_events_by_type(result.output_events, SMSSent)
    boss_sms = [
        s for s in sms_events if s.contact["contact_id"] == BOSS["contact_id"]
    ]
    assert len(boss_sms) >= 1, (
        f"Expected SMS reply to boss, got SMS to: "
        f"{[s.contact['contact_id'] for s in sms_events]}"
    )

    boss_msg = boss_sms[0].content
    assert _is_english(boss_msg), (
        f"Summary for the boss should be in English, got: {boss_msg}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_relay_japanese_content_to_bob_in_english(initialized_cm):
    """
    Alice messages in Japanese, boss relays to Bob -> Bob receives English.

    Non-Latin script makes language leakage especially visible. Bob's SMS
    must not contain any Japanese characters.
    """
    cm = initialized_cm

    # Alice sends Japanese informational message (no action needed)
    await cm.step_until_wait(
        UnifyMessageReceived(
            contact=ALICE,
            content=(
                "金曜日のディナーの件ですが、午後7時にイタリアンレストランで"
                "予約を入れました。楽しみにしています！"
            ),
        ),
    )

    # Bob sends a simple English greeting (establishes active conversation
    # without mentioning the dinner — we don't want the model to pre-relay
    # Alice's info before the boss asks)
    await cm.step_until_wait(
        SMSReceived(
            contact=BOB,
            content="Hey, just wanted to say hi. Hope your week is going well!",
        ),
    )

    # Boss relays (explicit send instruction)
    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Send Bob a text with what Alice said about Friday dinner",
        ),
    )

    sms_events = filter_events_by_type(result.output_events, SMSSent)
    bob_sms = [
        s for s in sms_events if s.contact["contact_id"] == BOB["contact_id"]
    ]
    assert len(bob_sms) >= 1, (
        f"Expected SMS to Bob, got SMS to: "
        f"{[s.contact['contact_id'] for s in sms_events]}"
    )

    bob_msg = bob_sms[0].content
    assert not _has_japanese(bob_msg), (
        f"Message to Bob must not contain Japanese, got: {bob_msg}"
    )
    assert _is_english(bob_msg), (
        f"Message to Bob should be in English, got: {bob_msg}"
    )


# =====================================================================
#  Group 4 — Multi-turn language consistency
#
#  Both turns are simple conversational exchanges — greetings and
#  thank-yous — requiring only a polite reply each time.
# =====================================================================


@pytest.mark.asyncio
@_handle_project
async def test_spanish_multi_turn_stays_spanish(initialized_cm):
    """Two-turn Spanish conversation: both replies must be in Spanish."""
    cm = initialized_cm

    # Turn 1: simple greeting and thanks
    result1 = await cm.step_until_wait(
        SMSReceived(
            contact=ALICE,
            content=(
                "¡Hola! Muchas gracias por todo tu trabajo. "
                "Me ha sido de gran ayuda."
            ),
        ),
    )
    sms1 = get_exactly_one(result1.output_events, SMSSent)
    assert _has_spanish(sms1.content), (
        f"Turn-1 reply should be in Spanish, got: {sms1.content}"
    )

    # Turn 2: follow-up thanks and well-wishing
    result2 = await cm.step_until_wait(
        SMSReceived(
            contact=ALICE,
            content=(
                "¡Qué amable! También quería decirte que todo salió "
                "muy bien con el proyecto. ¡Buen trabajo!"
            ),
        ),
    )
    sms2 = get_exactly_one(result2.output_events, SMSSent)
    assert _has_spanish(sms2.content), (
        f"Turn-2 reply should be in Spanish, got: {sms2.content}"
    )


# =====================================================================
#  Group 5 — Different languages per contact
#
#  Simple thank-you messages from each contact.  The assistant must
#  track language preference per contact and reply appropriately.
# =====================================================================


@pytest.mark.asyncio
@_handle_project
async def test_two_contacts_different_languages(initialized_cm):
    """
    Alice speaks Spanish, Bob speaks French -> each gets their language.

    Both contacts message in the same session. The assistant must track
    language preference per contact and reply appropriately.
    """
    cm = initialized_cm

    # Alice messages in Spanish (simple thanks — no action needed)
    result_a = await cm.step_until_wait(
        SMSReceived(
            contact=ALICE,
            content=(
                "¡Hola! Solo quería darte las gracias por tu ayuda. "
                "¡Fue genial!"
            ),
        ),
    )
    alice_sms = get_exactly_one(result_a.output_events, SMSSent)
    assert _has_spanish(alice_sms.content), (
        f"Reply to Alice should be in Spanish, got: {alice_sms.content}"
    )

    # Bob messages in French (simple thanks — no action needed)
    result_b = await cm.step_until_wait(
        SMSReceived(
            contact=BOB,
            content=(
                "Bonjour ! Je voulais vous remercier pour votre aide. "
                "C'était vraiment parfait !"
            ),
        ),
    )
    bob_sms = get_exactly_one(result_b.output_events, SMSSent)
    assert _has_french(bob_sms.content), (
        f"Reply to Bob should be in French, got: {bob_sms.content}"
    )
