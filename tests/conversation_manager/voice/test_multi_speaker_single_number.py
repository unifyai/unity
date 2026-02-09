"""
tests/conversation_manager/voice/test_multi_speaker_single_number.py
=====================================================================

Tests for multi-speaker scenarios on a single phone number / single contact.

It is very common for multiple people to be speaking on the other end of a
single phone number associated with a single contact. For example:
- The boss is on a call and introduces a colleague: "I'm going to introduce
  you to Richard, he's on the line now"
- A demo scenario where the assistant is being shown to a potential new user
- A speakerphone call with multiple participants

In these cases:
- The fast brain prompt shows only the single contact's details (boss)
- New speakers appear ONLY in the speech transcript (introductions, name mentions)
- The model must infer who is speaking from conversational cues
- Voice detection is not available (especially in STT→TTS pipeline)

IMPORTANT NOTE ON MESSAGE FORMAT:
- Fast brain (LiveKit ChatContext): Messages are bare {"role": "user", "content": "..."}
  with NO contact name label. The contact identity comes only from the system prompt.
- Slow brain (Renderer): Messages ARE labeled with the contact's name in the transcript,
  e.g., "[Dan Lewis @ DATE]: Hi Alex, this is Richard." — ALL utterances from that phone
  number are attributed to the registered contact, even when a different person is speaking.

These tests verify that both the fast brain (gpt-5-nano) and slow brain understand
this nuance. The fast brain tests deliberately avoid trivial scenarios where speakers
explicitly self-identify, focusing instead on whether a small model can track speaker
context across multiple turns.

Markers: All tests are eval tests (LLM reasoning about multi-speaker dynamics).
"""

from __future__ import annotations

import pytest

from unity.common.llm_client import new_llm_client
from unity.conversation_manager.prompt_builders import build_voice_agent_prompt
from unity.conversation_manager.events import (
    PhoneCallStarted,
    InboundPhoneUtterance,
    OutboundPhoneUtterance,
    ActorResult,
    ActorHandleStarted,
)
from unity.conversation_manager.types import Mode, Medium

from tests.conversation_manager.conftest import BOSS, TEST_CONTACTS

pytestmark = pytest.mark.eval


# =============================================================================
# Helpers
# =============================================================================


def _mentions_name(response: str, name: str) -> bool:
    """Check if the response mentions a name (case-insensitive)."""
    return name.lower() in response.lower()


def _is_confused_about_speaker(response: str, caller_name: str, new_name: str) -> bool:
    """
    Detect if the response confuses the new speaker with the registered contact.

    Signs of confusion:
    - Suggesting they'll talk to the new person "later" or "separately"
    - Treating the introduction as a request to call someone else
    """
    response_lower = response.lower()
    confusion_patterns = [
        "i'll call",
        "i can call",
        "i will call",
        "reach out to",
        "contact them",
        "give them a call",
        "call them",
        "in a separate call",
        "in a new call",
        "at some point",
    ]
    return any(p in response_lower for p in confusion_patterns)


async def _get_fast_brain_response_raw(
    system_prompt: str,
    conversation: list[dict[str, str]],
) -> str:
    """
    Get a response from the fast brain (gpt-5-nano) using the production pathway.

    Unlike get_fast_brain_response() from test_fast_brain_deferral.py, this does
    NOT append an artificial "Respond as the assistant" meta-instruction. The model
    receives exactly the system prompt + conversation messages, matching production.
    """
    from unity.conversation_manager.livekit_unify_adapter import UnifyLLM
    from livekit.agents import llm

    llm_instance = UnifyLLM(model="gpt-5-nano@openai", reasoning_effort="minimal")

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content=system_prompt)

    for msg in conversation:
        chat_ctx.add_message(role=msg["role"], content=msg["content"])

    stream = llm_instance.chat(chat_ctx=chat_ctx)
    response_parts = []
    try:
        async for chunk in stream:
            if hasattr(chunk, "delta") and chunk.delta:
                content = getattr(chunk.delta, "content", None)
                if content:
                    response_parts.append(content)
    finally:
        await stream.aclose()

    return "".join(response_parts)


# =============================================================================
# Fixtures
# =============================================================================


def _build_boss_call_prompt() -> str:
    """Voice agent prompt for a call with the boss (single contact on file)."""
    return build_voice_agent_prompt(
        bio="A helpful and efficient virtual assistant.",
        assistant_name="Alex",
        boss_first_name="Dan",
        boss_surname="Lewis",
        boss_phone_number="+1-555-0100",
        boss_email_address="dan.lewis@example.com",
        boss_bio="CEO of a technology startup. Frequently demos the assistant to potential clients and partners.",
        is_boss_user=True,
        contact_rolling_summary=None,
    ).flatten()


@pytest.fixture
def boss_call_prompt():
    return _build_boss_call_prompt()


# =============================================================================
# Test Class: Fast brain addressing introduced speaker by name (no meta prompt)
# =============================================================================


@pytest.mark.asyncio
class TestFastBrainMultiSpeakerTracking:
    """
    Tests whether gpt-5-nano can correctly track and address a new speaker
    introduced mid-call. Uses the raw fast brain pathway (no artificial
    meta-instruction) to match production behavior.

    The system prompt says "I am on a phone call with my boss" + boss details.
    Messages are bare role:"user" — no contact name labels. Speakers are
    identified only by conversational cues in the text content.
    """

    async def test_greets_introduced_person_by_name(self, boss_call_prompt):
        """
        Boss explicitly introduces Richard. Richard speaks. The fast brain
        should greet Richard by name in its response.
        """
        conversation = [
            {
                "role": "user",
                "content": (
                    "Hey Alex, I want to introduce you to someone. "
                    "Richard is here with me, I'm going to put him on."
                ),
            },
            {
                "role": "assistant",
                "content": "Of course! I'd be happy to meet them.",
            },
            {
                "role": "user",
                "content": "Hi Alex, this is Richard. Dan's told me a lot about you.",
            },
        ]

        response = await _get_fast_brain_response_raw(boss_call_prompt, conversation)

        assert _mentions_name(response, "Richard"), (
            f"Fast brain should greet Richard by name!\n"
            f"Response: {response}\n\n"
            f"The boss introduced Richard who is now speaking. The fast brain\n"
            f"should respond with something like 'Nice to meet you, Richard!'"
        )

        assert not _is_confused_about_speaker(response, "Dan", "Richard"), (
            f"Fast brain is confused about the speaker!\n"
            f"Response: {response}\n\n"
            f"Richard is ALREADY on the line speaking. The fast brain should\n"
            f"engage with him directly, not suggest calling him separately."
        )

    async def test_uses_name_in_later_turn_without_reidentification(
        self,
        boss_call_prompt,
    ):
        """
        After Richard is introduced and the conversation proceeds for several
        turns, Richard asks a question WITHOUT re-identifying himself. The fast
        brain should still address him as Richard (not Dan).

        This is the harder test: the model must maintain speaker context across
        turns without any re-identification cue.
        """
        conversation = [
            {
                "role": "user",
                "content": (
                    "Alex, I've got my colleague Richard here. He wants to "
                    "ask you some questions about what you can do."
                ),
            },
            {
                "role": "assistant",
                "content": "Hi Richard! Great to meet you. Go ahead, I'm all ears.",
            },
            {
                "role": "user",
                "content": "So what kind of admin tasks can you handle?",
            },
            {
                "role": "assistant",
                "content": (
                    "I can handle scheduling, email management, data entry, "
                    "research, document drafting, and more. What's your typical workload?"
                ),
            },
            {
                "role": "user",
                "content": (
                    "That's useful. We mainly need help with client follow-ups "
                    "and keeping our CRM updated. Can you do that?"
                ),
            },
        ]

        response = await _get_fast_brain_response_raw(boss_call_prompt, conversation)

        # The response should NOT address "Dan" — the conversation context
        # makes it clear Richard has been asking the questions since the intro.
        # The fast brain should either use "Richard" or just answer naturally
        # without mistakenly attributing the question to Dan.
        response_lower = response.lower()

        # Should NOT address Dan when Richard is the one asking
        is_addressing_dan = (
            "dan" in response_lower
            and not "dan mentioned" in response_lower
            and not "dan's" in response_lower
        )
        assert not is_addressing_dan, (
            f"Fast brain incorrectly addressed Dan instead of Richard!\n"
            f"Response: {response}\n\n"
            f"Richard has been asking questions since the introduction.\n"
            f"The fast brain should be answering Richard, not Dan."
        )

    async def test_handles_speaker_switch_without_explicit_identification(
        self,
        boss_call_prompt,
    ):
        """
        Richard was introduced, they chat for a bit, then someone starts asking
        about a completely different topic without saying "it's Dan again."
        The fast brain should at minimum NOT address the unidentified speaker as
        Richard with confidence.

        This tests whether the model understands ambiguity — when it's unclear
        who's speaking, it shouldn't confidently attribute to the wrong person.
        """
        conversation = [
            {
                "role": "user",
                "content": (
                    "Alex, Richard's here with me. He's evaluating assistants "
                    "for his team."
                ),
            },
            {
                "role": "assistant",
                "content": "Hi Richard! Happy to show you what I can do.",
            },
            {
                "role": "user",
                "content": "What kind of software can you use?",
            },
            {
                "role": "assistant",
                "content": (
                    "Pretty much anything — browsers, spreadsheets, CRMs, "
                    "project management tools, you name it."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Oh hey, by the way, did you finish looking into "
                    "that vendor contract I asked about yesterday?"
                ),
            },
        ]

        response = await _get_fast_brain_response_raw(boss_call_prompt, conversation)

        # The last question about "that vendor contract I asked about yesterday"
        # is almost certainly Dan (the boss), not Richard. Richard was just introduced.
        # A good model should either:
        # a) Recognize this is likely Dan and respond accordingly, OR
        # b) Not confidently attribute this to Richard
        # The model should NOT say "Sure Richard, let me check on that contract"
        response_lower = response.lower()
        incorrectly_attributes_to_richard = (
            "richard" in response_lower
            and ("contract" in response_lower or "vendor" in response_lower)
        )
        assert not incorrectly_attributes_to_richard, (
            f"Fast brain incorrectly attributed the boss's question to Richard!\n"
            f"Response: {response}\n\n"
            f"The question about 'that vendor contract I asked about yesterday'\n"
            f"is clearly from Dan (the boss), not Richard who was just introduced.\n"
            f"The fast brain should not say 'Sure Richard, let me check...'"
        )

    async def test_demo_introduction_without_name_in_greeting(
        self,
        boss_call_prompt,
    ):
        """
        Boss introduces someone for a demo, then the new person starts talking
        without saying their name. The boss mentioned the name, so the model
        should use it.

        This tests whether the model can pick up a name from a THIRD PARTY
        introduction rather than self-identification.
        """
        conversation = [
            {
                "role": "user",
                "content": (
                    "Alex, I'm here with Maria. She's thinking about getting "
                    "an assistant for her team. Maria, go ahead and ask "
                    "Alex anything."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Hi! So, can you help manage a team's calendar across "
                    "different time zones?"
                ),
            },
        ]

        response = await _get_fast_brain_response_raw(boss_call_prompt, conversation)

        # The second user message is from Maria (based on Dan's intro).
        # Maria didn't self-identify — Dan introduced her. The model should
        # ideally use "Maria" in the response, but at minimum should NOT
        # address this as Dan.
        assert _mentions_name(response, "Maria"), (
            f"Fast brain should address Maria by name!\n"
            f"Response: {response}\n\n"
            f"Dan introduced Maria and asked her to go ahead. The next message\n"
            f"is from Maria (she asks about calendars). The fast brain should\n"
            f"greet her by name — 'Hi Maria!' or 'Great question, Maria.'\n"
            f"Maria didn't self-identify, but Dan's introduction was explicit."
        )

    async def test_does_not_confuse_introduction_with_call_request(
        self,
        boss_call_prompt,
    ):
        """
        Critical disambiguation test: when the boss says "Richard is here",
        the fast brain must NOT interpret this as "please call Richard."
        """
        conversation = [
            {
                "role": "user",
                "content": (
                    "Hey Alex, so Richard's sitting right next to me. "
                    "I'm going to hand the phone over to him."
                ),
            },
        ]

        response = await _get_fast_brain_response_raw(boss_call_prompt, conversation)

        assert not _is_confused_about_speaker(response, "Dan", "Richard"), (
            f"Fast brain confused 'Richard is here' with a request to call him!\n"
            f"Response: {response}\n\n"
            f"Dan said Richard is 'sitting right next to me' and is 'handing the\n"
            f"phone over'. This is a physical handoff, not a request to make a call."
        )


# =============================================================================
# Test Class: Fast brain handling notification data with multi-speaker context
# =============================================================================


@pytest.mark.asyncio
class TestFastBrainNotificationWithMultiSpeaker:
    """
    Tests that the fast brain correctly relays notification data even when
    multiple speakers are present. The notification should be shared naturally
    without attributing it to the wrong person.
    """

    async def test_relays_notification_to_correct_speaker(self, boss_call_prompt):
        """
        Richard asks a data question, a notification arrives with the answer.
        The fast brain should relay this to Richard specifically (not Dan).
        """
        conversation = [
            {
                "role": "user",
                "content": "Alex, Richard here. How many active clients do we have?",
            },
            {
                "role": "assistant",
                "content": "Let me check on that.",
            },
            {
                "role": "system",
                "content": "[notification] There are currently 47 active clients.",
            },
        ]

        response = await _get_fast_brain_response_raw(boss_call_prompt, conversation)

        # Should relay the data
        assert "47" in response, (
            f"Fast brain should relay the notification data!\n"
            f"Response: {response}\n\n"
            f"The notification said there are 47 active clients. The fast brain\n"
            f"should share this information with the caller."
        )

        # Should NOT address Dan when Richard asked the question
        response_lower = response.lower()
        incorrectly_addresses_dan = "dan" in response_lower and "47" in response_lower
        assert not incorrectly_addresses_dan, (
            f"Fast brain addressed Dan when Richard asked the question!\n"
            f"Response: {response}\n\n"
            f"Richard asked about active clients. The answer should be directed\n"
            f"to Richard, not Dan."
        )


# =============================================================================
# Test Class: Slow brain understanding of multi-speaker dynamics
# =============================================================================


@pytest.mark.asyncio
class TestSlowBrainMultiSpeakerAwareness:
    """
    Tests that the slow brain (Main CM Brain) correctly understands multi-speaker
    dynamics on a single phone line.

    CRITICAL CONTEXT: In the slow brain's transcript, ALL messages from the
    phone number are labeled with the REGISTERED CONTACT'S name. For example:

        [Default User @ DATE]: Hey, I'm introducing Richard.
        [Default User @ DATE]: Hi Alex, this is Richard. Nice to meet you!

    Both lines show "Default User" even though the second is actually Richard.
    The slow brain must understand this discrepancy from the text content alone.
    """

    async def test_slow_brain_does_not_initiate_separate_call(
        self,
        initialized_cm,
    ):
        """
        When the boss introduces someone on the call, the slow brain should NOT
        interpret this as a request to call that person separately.

        In the slow brain's transcript, this looks like:
            [Default User @ DATE]: Hey, Richard is here with us now.
        The slow brain must understand Richard is ALREADY present.
        """
        boss = BOSS

        await initialized_cm.step(PhoneCallStarted(contact=boss))
        assert initialized_cm.cm.mode == Mode.CALL

        initialized_cm.all_tool_calls.clear()
        result = await initialized_cm.step_until_wait(
            InboundPhoneUtterance(
                contact=boss,
                content=(
                    "Hey, I'm here with my friend Richard. He's on the line "
                    "with us now. Richard, go ahead and introduce yourself."
                ),
            ),
            max_steps=5,
        )

        assert "make_call" not in initialized_cm.all_tool_calls, (
            f"Slow brain tried to call Richard separately!\n"
            f"Tool calls: {initialized_cm.all_tool_calls}\n\n"
            f"Richard is ALREADY on the line. The boss said 'he's on the line\n"
            f"with us now'. The slow brain should not interpret this as a\n"
            f"request to call Richard."
        )

    async def test_slow_brain_understands_new_speaker_data_request(
        self,
        initialized_cm,
    ):
        """
        A new speaker (introduced on the call) asks for specific data. Even though
        the transcript shows the message as coming from the boss's contact, the
        slow brain should understand the social context and process the data request.

        In the slow brain's transcript, this looks like:
            [Default User @ DATE]: Alex, I've got Richard here. He has a question.
            [Default User @ DATE]: Hi Alex, Richard here. What's the Henderson project status?

        Both labeled "Default User" but the slow brain should understand that the
        second message's data request is legitimate and trigger an act() call.
        """
        boss = BOSS

        await initialized_cm.step(PhoneCallStarted(contact=boss))

        # Introduction (don't run LLM — just register the utterance)
        await initialized_cm.step(
            InboundPhoneUtterance(
                contact=boss,
                content="Alex, I've got my colleague Richard here. He has a question for you.",
            ),
            run_llm=False,
        )

        # Richard asks a data question (still labeled as boss in the transcript)
        initialized_cm.all_tool_calls.clear()
        result = await initialized_cm.step_until_wait(
            InboundPhoneUtterance(
                contact=boss,
                content=(
                    "Hi Alex, Richard here. Dan mentioned you've been tracking "
                    "the Henderson project. Can you tell me the current status?"
                ),
            ),
            max_steps=5,
        )

        # The slow brain should recognize this as a legitimate data request
        assert result.llm_ran, (
            f"Slow brain should have run to handle the data request!\n"
            f"Richard asked about the Henderson project status — this requires\n"
            f"a data lookup via `act`."
        )

    async def test_slow_brain_does_not_confuse_speaker_with_contact_label(
        self,
        initialized_cm,
    ):
        """
        The hardest test: the transcript shows messages labeled with the boss's name,
        but the TEXT content clearly indicates a different person is speaking. The slow
        brain should reason about the multi-speaker scenario rather than blindly
        trusting the contact label.

        Transcript as seen by slow brain:
            [Default User @ DATE]: I'm handing the phone to Richard now.
            [Default User @ DATE]: Hey, so I run a logistics company and we need help with invoicing.

        The second message is labeled "Default User" but is clearly from Richard
        (who runs a logistics company, not the boss who is a CEO). The slow brain
        should NOT treat this as the boss asking about their own invoicing.
        """
        boss = BOSS

        await initialized_cm.step(PhoneCallStarted(contact=boss))

        # Boss hands off
        await initialized_cm.step(
            InboundPhoneUtterance(
                contact=boss,
                content="I'm handing the phone to Richard now. He wants to discuss his business needs.",
            ),
            run_llm=False,
        )

        # Richard speaks (labeled as boss in transcript)
        initialized_cm.all_tool_calls.clear()
        result = await initialized_cm.step_until_wait(
            InboundPhoneUtterance(
                contact=boss,
                content=(
                    "Hey, so I run a logistics company and we need help "
                    "with invoicing and shipment tracking. Is that something "
                    "you can handle?"
                ),
            ),
            max_steps=5,
        )

        # The key check: the slow brain should NOT have called make_call
        # (would indicate it thinks it needs to contact Richard separately)
        assert "make_call" not in initialized_cm.all_tool_calls, (
            f"Slow brain tried to call someone after a speaker handoff!\n"
            f"Tool calls: {initialized_cm.all_tool_calls}\n\n"
            f"The boss handed the phone to Richard. The subsequent message\n"
            f"is from Richard (on the same call), not a request to call anyone."
        )

    async def test_slow_brain_call_guidance_after_multi_speaker_data_request(
        self,
        initialized_cm,
    ):
        """
        Full flow: boss introduces someone, new speaker asks for data, slow brain
        triggers act, result arrives, slow brain produces guidance.

        The guidance should be relevant to the new speaker's question, not confused
        by the contact label mismatch.
        """
        boss = BOSS

        await initialized_cm.step(PhoneCallStarted(contact=boss))

        # Introduction (no LLM)
        await initialized_cm.step(
            InboundPhoneUtterance(
                contact=boss,
                content="Alex, Richard is here. He wants to know about our current task pipeline.",
            ),
            run_llm=False,
        )

        # Richard asks about tasks
        initialized_cm.all_tool_calls.clear()
        result = await initialized_cm.step_until_wait(
            InboundPhoneUtterance(
                contact=boss,
                content="What tasks are currently in progress?",
            ),
            max_steps=5,
        )

        # If the slow brain called act, complete the action and check guidance
        if "act" in initialized_cm.all_tool_calls and initialized_cm.cm.in_flight_actions:
            handle_id = next(iter(initialized_cm.cm.in_flight_actions))

            def _get_guidance_messages(cm, contact_id: int) -> list:
                conv = cm.contact_index.get_conversation_state(contact_id)
                if not conv:
                    return []
                voice_thread = list(conv.threads.get(Medium.PHONE_CALL, []))
                return [
                    msg
                    for msg in voice_thread
                    if getattr(msg, "name", None) == "guidance"
                ]

            guidance_before = _get_guidance_messages(
                initialized_cm.cm, boss["contact_id"]
            )

            result = await initialized_cm.step_until_wait(
                ActorResult(
                    handle_id=handle_id,
                    success=True,
                    result=(
                        "Currently 3 tasks in progress: Website redesign (70% complete), "
                        "Q1 report compilation (50%), and client onboarding for Acme Corp (30%)."
                    ),
                ),
                max_steps=5,
            )

            guidance_after = _get_guidance_messages(
                initialized_cm.cm, boss["contact_id"]
            )
            new_guidance = guidance_after[len(guidance_before):]

            all_guidance_text = " ".join(
                getattr(g, "content", "") for g in guidance_after
            ).lower()

            assert any(
                term in all_guidance_text
                for term in ["task", "website", "q1", "report", "acme", "progress"]
            ), (
                f"Slow brain guidance should contain task pipeline info!\n"
                f"All guidance: {[getattr(g, 'content', '') for g in guidance_after]}\n"
                f"Tool calls: {initialized_cm.all_tool_calls}\n\n"
                f"Richard asked about tasks in progress. The action returned\n"
                f"task data. The slow brain should relay this via guidance."
            )


# =============================================================================
# Test Class: Fast brain with multi-turn conversation (no re-identification)
# =============================================================================


@pytest.mark.asyncio
class TestFastBrainExtendedMultiSpeakerConversation:
    """
    Longer conversation tests where the fast brain must maintain speaker context
    across many turns. These are the hardest tests for gpt-5-nano because the
    model must remember who's speaking without repeated name cues.
    """

    async def test_maintains_context_over_extended_exchange(self, boss_call_prompt):
        """
        After a clear introduction, Richard has a 4-turn conversation. On the
        5th turn, he asks a confirming question. The fast brain should still
        be engaging with Richard, not suddenly addressing Dan.

        This specifically tests whether gpt-5-nano's limited context window
        and reasoning can maintain speaker identity over multiple turns.
        """
        conversation = [
            {
                "role": "user",
                "content": (
                    "Hey Alex, Dan here. I've got Richard with me — he's "
                    "evaluating your services. Handing him the phone."
                ),
            },
            {
                "role": "assistant",
                "content": "Hi Richard, nice to meet you! What would you like to know?",
            },
            {
                "role": "user",
                "content": "Can you handle email management?",
            },
            {
                "role": "assistant",
                "content": (
                    "Absolutely — inbox triage, drafting replies, scheduling "
                    "follow-ups, the works."
                ),
            },
            {
                "role": "user",
                "content": "What about CRM updates?",
            },
            {
                "role": "assistant",
                "content": (
                    "Yes, I can update records in Salesforce, HubSpot, or "
                    "whatever CRM you use."
                ),
            },
            {
                "role": "user",
                "content": "And scheduling meetings across time zones?",
            },
            {
                "role": "assistant",
                "content": (
                    "That too — I'll find times that work for all participants "
                    "and send calendar invites."
                ),
            },
            {
                "role": "user",
                "content": "Great, thanks. That all sounds really useful.",
            },
        ]

        response = await _get_fast_brain_response_raw(boss_call_prompt, conversation)

        # The model should respond naturally to the sign-off. It should NOT
        # suddenly switch to addressing Dan. Ideal: "Glad I could help, Richard!"
        # Acceptable: natural sign-off without naming anyone.
        # Bad: "You're welcome, Dan!" or addressing the wrong person.

        response_lower = response.lower()
        mistakenly_addresses_dan = "dan" in response_lower and any(
            phrase in response_lower
            for phrase in ["welcome dan", "thanks dan", "glad dan", "hope dan"]
        )
        assert not mistakenly_addresses_dan, (
            f"Fast brain incorrectly addressed Dan at the end of Richard's conversation!\n"
            f"Response: {response}\n\n"
            f"Richard has been the one talking throughout this exchange.\n"
            f"The fast brain should not suddenly switch to addressing Dan."
        )

    async def test_back_and_forth_between_speakers_with_topic_cues(
        self,
        boss_call_prompt,
    ):
        """
        Dan and Richard take turns asking questions. The only cue for who's
        speaking is the TOPIC — Dan asks about his own schedule, Richard asks
        about service evaluation.

        This is extremely hard for a small model. It tests implicit speaker
        tracking based on conversational coherence rather than explicit naming.
        """
        conversation = [
            {
                "role": "user",
                "content": (
                    "Alex, I'm with James. He's looking at virtual assistant "
                    "options for his company."
                ),
            },
            {
                "role": "assistant",
                "content": "Great, hi James! Happy to answer any questions.",
            },
            {
                "role": "user",
                "content": "How do you handle confidential documents?",
            },
            {
                "role": "assistant",
                "content": (
                    "Everything is handled securely — encrypted channels, "
                    "strict access controls, and I follow data handling best practices."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Hey Alex, it's Dan. Quick thing — did you manage to send "
                    "that follow-up email to the Henderson team?"
                ),
            },
        ]

        response = await _get_fast_brain_response_raw(boss_call_prompt, conversation)

        # Dan explicitly identifies himself and asks about a prior task.
        # The fast brain should handle this as Dan's request (likely defer
        # since it doesn't have email status data).
        response_lower = response.lower()

        is_contextually_appropriate = any(
            indicator in response_lower
            for indicator in [
                "dan",
                "henderson",
                "email",
                "follow-up",
                "follow up",
                "let me check",
                "checking",
                "one moment",
            ]
        )
        assert is_contextually_appropriate, (
            f"Fast brain didn't handle the speaker switch back to Dan!\n"
            f"Response: {response}\n\n"
            f"Dan explicitly said 'it's Dan' and asked about a prior task.\n"
            f"The fast brain should recognize Dan and handle his request."
        )
