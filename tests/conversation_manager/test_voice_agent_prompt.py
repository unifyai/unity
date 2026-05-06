"""
tests/conversation_manager/test_voice_agent_prompt.py
=====================================================

Tests for the Voice Agent (fast brain) prompt builder.

**Context enrichment tests** verify the LLM can answer questions directly
using context provided in the system prompt, rather than deferring:
1. **Assistant name**: The fast brain knows its own name and can introduce itself.
2. **Contact bio**: The fast brain knows the bio/background of the person on the call.
3. **Meet participants**: The fast brain knows all participant details in multi-party calls.

**Brevity tests** (eval) verify the fast brain keeps responses concise —
short enough for a natural phone conversation, not chatbot-style paragraphs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, Field

from unity.common.llm_client import new_llm_client
from unity.common.reasoning import reason
from unity.conversation_manager.prompt_builders import build_voice_agent_prompt

# =============================================================================
# Constants
# =============================================================================

# Deferral phrases the fast brain uses when it doesn't have data.
# If any of these appear in the response, the fast brain is deferring
# instead of answering directly.
DEFERRAL_PHRASES = [
    "let me check",
    "let me look",
    "i'm looking into",
    "i'll check",
    "i'll look into",
    "i need to check",
    "one moment",
    "hold on",
    "let me find",
    "checking on that",
    "looking into that",
]

# The model used by the fast brain in production
FAST_BRAIN_MODEL = "gpt-5.4-mini@openai"


# =============================================================================
# Helpers
# =============================================================================


async def ask_fast_brain(system_prompt: str, user_message: str) -> str:
    """Send a user message to the fast brain LLM and return the response.

    Args:
        system_prompt: The voice agent system prompt.
        user_message: The user's spoken message.

    Returns:
        The assistant's response text.
    """
    client = new_llm_client(
        model=FAST_BRAIN_MODEL,
        reasoning_effort="low",
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    response = await client.generate(messages=messages)
    return response.strip()


async def ask_fast_brain_conversation(
    system_prompt: str,
    conversation: tuple[dict[str, str], ...],
) -> str:
    """Run the fast brain over a live-call transcript."""
    client = new_llm_client(
        model=FAST_BRAIN_MODEL,
        reasoning_effort="low",
    )
    messages = [{"role": "system", "content": system_prompt}, *conversation]
    response = await client.generate(messages=messages)
    return response.strip()


def assert_no_deferral(response: str, context: str = "") -> None:
    """Assert that the response does not contain deferral phrases.

    Args:
        response: The assistant's response text.
        context: Optional description of what we're testing.
    """
    response_lower = response.lower()
    for phrase in DEFERRAL_PHRASES:
        assert phrase not in response_lower, (
            f"Fast brain deferred instead of answering directly!\n"
            f"Deferral phrase found: '{phrase}'\n"
            f"Full response: {response}\n"
            f"{f'Context: {context}' if context else ''}"
        )


def assert_contains(response: str, expected: str, context: str = "") -> None:
    """Assert that the response contains the expected substring (case-insensitive).

    Args:
        response: The assistant's response text.
        expected: The substring that should appear.
        context: Optional description of what we're testing.
    """
    assert expected.lower() in response.lower(), (
        f"Expected '{expected}' in response but not found!\n"
        f"Full response: {response}\n"
        f"{f'Context: {context}' if context else ''}"
    )


def assert_concise(response: str, max_words: int = 50, context: str = "") -> None:
    """Assert that the response is concise (phone-call brevity).

    Args:
        response: The assistant's response text.
        max_words: Maximum acceptable word count.
        context: Optional description of what we're testing.
    """
    word_count = len(response.split())
    assert word_count <= max_words, (
        f"Response too verbose ({word_count} words, max {max_words})!\n"
        f"Full response: {response}\n"
        f"{f'Context: {context}' if context else ''}"
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_prompt_kwargs() -> dict:
    """Base keyword arguments for build_voice_agent_prompt."""
    return {
        "bio": "I am a virtual assistant working for a tech startup.",
        "assistant_name": "Alex",
        "boss_first_name": "Sarah",
        "boss_surname": "Chen",
        "boss_phone_number": "+15551234567",
        "boss_email_address": "sarah@meridianlabs.com",
        "boss_bio": "CEO of Meridian Labs, focused on AI products.",
    }


@pytest.fixture
def boss_call_prompt(base_prompt_kwargs: dict) -> str:
    """Voice agent prompt for a call with the boss."""
    return build_voice_agent_prompt(
        **base_prompt_kwargs,
        is_boss_user=True,
    ).flatten()


@pytest.fixture
def contact_call_prompt(base_prompt_kwargs: dict) -> str:
    """Voice agent prompt for a call with an external contact."""
    return build_voice_agent_prompt(
        **base_prompt_kwargs,
        is_boss_user=False,
        contact_first_name="Marcus",
        contact_surname="Rivera",
        contact_phone_number="+15559876543",
        contact_email="marcus@clientcorp.com",
        contact_bio="VP of Engineering at ClientCorp. Leading their cloud migration project. Prefers concise communication.",
    ).flatten()


@pytest.fixture
def meet_prompt(base_prompt_kwargs: dict) -> str:
    """Voice agent prompt for a multi-party Unify Meet."""
    return build_voice_agent_prompt(
        **base_prompt_kwargs,
        is_boss_user=True,
        channel="unify_meet",
        participants=[
            {
                "first_name": "Sarah",
                "surname": "Chen",
                "bio": "CEO of Meridian Labs, focused on AI products.",
            },
            {
                "first_name": "Marcus",
                "surname": "Rivera",
                "bio": "VP of Engineering at ClientCorp. Leading their cloud migration project.",
            },
            {
                "first_name": "Priya",
                "surname": "Sharma",
                "bio": "Product Manager at Meridian Labs. Coordinates between engineering and clients.",
            },
        ],
    ).flatten()


@pytest.fixture
def coordinator_call_prompt(base_prompt_kwargs: dict) -> str:
    """Voice agent prompt for a Coordinator on a live call with its boss."""
    kwargs = {
        **base_prompt_kwargs,
        "bio": "I help Meridian Labs set up its Unify assistant workforce.",
        "assistant_name": "Avery",
    }
    return build_voice_agent_prompt(
        **kwargs,
        is_boss_user=True,
        is_coordinator=True,
    ).flatten()


# =============================================================================
# Test Class: Assistant Name
# =============================================================================


@pytest.mark.llm_call
@pytest.mark.asyncio
class TestAssistantName:
    """Tests that the fast brain knows and uses its own name."""

    async def test_assistant_states_own_name(self, boss_call_prompt: str):
        """
        When asked "what's your name?", the fast brain should answer directly
        with its name instead of deferring.
        """
        response = await ask_fast_brain(boss_call_prompt, "What's your name?")

        assert_no_deferral(response, "Asked for assistant's own name")
        assert_contains(response, "Alex", "Assistant should state its name")

    async def test_assistant_introduces_itself(self, boss_call_prompt: str):
        """
        When greeting and being asked to introduce themselves, the fast brain
        should include its name naturally.
        """
        response = await ask_fast_brain(
            boss_call_prompt,
            "Hi there! Who am I speaking with?",
        )

        assert_no_deferral(response, "Asked who they're speaking with")
        assert_contains(response, "Alex", "Assistant should mention its name")


# =============================================================================
# Test Class: Contact Bio
# =============================================================================


@pytest.mark.llm_call
@pytest.mark.asyncio
class TestContactBio:
    """Tests that the fast brain can use contact bio information."""

    async def test_knows_contact_role(self, contact_call_prompt: str):
        """
        When asked about the contact's role, the fast brain should answer
        directly using the bio context.
        """
        response = await ask_fast_brain(
            contact_call_prompt,
            "Remind me, what does Marcus do at his company?",
        )

        assert_no_deferral(response, "Asked about contact's role from bio")
        assert_contains(
            response,
            "engineer",
            "Should mention engineering role from bio",
        )

    async def test_knows_contact_project(self, contact_call_prompt: str):
        """
        When asked about what the contact is working on, the fast brain
        should use the bio context.
        """
        response = await ask_fast_brain(
            contact_call_prompt,
            "What project is Marcus leading?",
        )

        assert_no_deferral(response, "Asked about contact's project from bio")
        assert_contains(
            response,
            "cloud migration",
            "Should mention cloud migration from bio",
        )


# =============================================================================
# Test Class: Meet Participants
# =============================================================================


@pytest.mark.llm_call
@pytest.mark.asyncio
class TestMeetParticipants:
    """Tests that the fast brain knows about all participants in a meet."""

    async def test_knows_all_participant_names(self, meet_prompt: str):
        """
        When asked who is on the call, the fast brain should list all
        participants without deferring.
        """
        response = await ask_fast_brain(
            meet_prompt,
            "Who's on this call right now?",
        )

        assert_no_deferral(response, "Asked who is on the meet")
        assert_contains(response, "Sarah", "Should mention Sarah")
        assert_contains(response, "Marcus", "Should mention Marcus")
        assert_contains(response, "Priya", "Should mention Priya")

    async def test_knows_participant_role(self, meet_prompt: str):
        """
        When asked about a specific participant's role, the fast brain
        should answer using their bio.
        """
        response = await ask_fast_brain(
            meet_prompt,
            "What's Priya's role?",
        )

        assert_no_deferral(response, "Asked about participant's role")
        assert_contains(
            response,
            "product manager",
            "Should mention Priya's role from bio",
        )

    async def test_knows_participant_company(self, meet_prompt: str):
        """
        When asked about a specific participant's company, the fast brain
        should answer using their bio.
        """
        response = await ask_fast_brain(
            meet_prompt,
            "Which company is Marcus from?",
        )

        assert_no_deferral(response, "Asked about participant's company")
        assert_contains(
            response,
            "ClientCorp",
            "Should mention Marcus's company from bio",
        )


# =============================================================================
# Test Class: Brevity (eval)
# =============================================================================


@pytest.mark.llm_call
@pytest.mark.eval
@pytest.mark.asyncio
class TestBrevity:
    """Eval tests verifying the fast brain keeps responses concise.

    On a phone call, responses should sound like a person talking — one or two
    sentences, not a paragraph.  These tests ask questions that typically provoke
    chatbot-style monologues and assert the response stays short.

    Each test uses a distinct, realistic bio (the kind a real user would write)
    so we exercise brevity against rich context, not bland test stubs.
    """

    async def test_tell_me_about_yourself(self):
        """
        "Tell me a bit about yourself" should get a brief, natural answer —
        not a feature list or corporate brochure.
        """
        prompt = build_voice_agent_prompt(
            bio="I have been working as an admin assistant for 3 years. I live in Spain, I love football, and I'm especially comfortable with MS Office.",
            assistant_name="Alex",
            boss_first_name="Sarah",
            boss_surname="Chen",
            is_boss_user=True,
        ).flatten()
        response = await ask_fast_brain(prompt, "Tell me a bit about yourself.")

        assert_concise(response, max_words=40, context="tell me about yourself")

    async def test_what_can_you_do(self):
        """
        "What can you do?" should get a conversational one-liner, not an
        exhaustive capability dump.
        """
        prompt = build_voice_agent_prompt(
            bio="Former executive assistant at a London law firm. I handle calendars, travel bookings, expense reports, and client correspondence. Big fan of hiking on weekends.",
            assistant_name="Jordan",
            boss_first_name="Marcus",
            boss_surname="Rivera",
            is_boss_user=True,
        ).flatten()
        response = await ask_fast_brain(prompt, "So what can you do?")

        assert_concise(response, max_words=40, context="what can you do")

    async def test_how_can_you_help_me(self):
        """
        "How can you help me?" — another common trigger for verbose responses.
        """
        prompt = build_voice_agent_prompt(
            bio="I've supported C-suite execs for 5 years across finance and tech. I'm great with Notion, Slack, and Google Workspace. Originally from Brazil, currently based in Lisbon.",
            assistant_name="Sam",
            boss_first_name="Priya",
            boss_surname="Sharma",
            is_boss_user=True,
        ).flatten()
        response = await ask_fast_brain(prompt, "How can you help me?")

        assert_concise(response, max_words=40, context="how can you help me")

    async def test_simple_greeting_is_short(self):
        """
        A casual "hey, how's it going?" should get a brief, warm reply —
        not a paragraph about the assistant's purpose.
        """
        prompt = build_voice_agent_prompt(
            bio="Personal assistant with a background in event planning. I'm based in Tokyo and speak Japanese and English fluently. I enjoy cooking and running.",
            assistant_name="Riley",
            boss_first_name="Tom",
            boss_surname="Nakamura",
            is_boss_user=True,
        ).flatten()
        response = await ask_fast_brain(prompt, "Hey, how's it going?")

        assert_concise(response, max_words=15, context="casual greeting")


# =============================================================================
# Test Class: Platform Knowledge
# =============================================================================


@pytest.mark.llm_call
@pytest.mark.asyncio
class TestPlatformKnowledge:
    """Tests that the fast brain can answer questions about external app
    integration directly using the platform knowledge in its prompt."""

    async def test_answers_app_integration_setup_directly(self, boss_call_prompt: str):
        """When asked HOW to set up an external app integration, the fast brain
        answers directly (no deferral) and mentions sharing credentials/tokens."""
        response = await ask_fast_brain(
            boss_call_prompt,
            "I want you to manage my Google Drive going forward. What do I need to do to set that up?",
        )

        assert_no_deferral(response, "Asked how to set up an external app integration")
        response_lower = response.lower().replace("’", "'")
        has_credential_mention = any(
            term in response_lower
            for term in ["credential", "token", "api", "secret", "access", "key"]
        )
        assert has_credential_mention, (
            f"Fast brain should mention credentials/tokens/secrets/API access "
            f"when explaining how to set up an integration.\n"
            f"Full response: {response}"
        )

    async def test_answers_console_navigation_directly(self, boss_call_prompt: str):
        """When asked where to find something on the console, the fast brain
        answers directly using platform knowledge rather than deferring."""
        response = await ask_fast_brain(
            boss_call_prompt,
            "Where do I go to add my API credentials on the console?",
        )

        assert_no_deferral(response, "Asked where to add credentials on console")
        response_lower = response.lower().replace("’", "'")
        assert "secret" in response_lower or "resource" in response_lower, (
            f"Fast brain should mention Secrets or Resources when explaining "
            f"where to add credentials on the console.\n"
            f"Full response: {response}"
        )
        assert_concise(response, max_words=60, context="console navigation answer")

    async def test_suggests_video_call_for_visual_guidance(
        self,
        boss_call_prompt: str,
    ):
        """When the user is struggling with a visual/setup task, the fast brain
        should suggest a video call with screen sharing rather than deferring."""
        response = await ask_fast_brain(
            boss_call_prompt,
            "I'm trying to find where to add my API credentials on the console but I can't figure it out.",
        )

        assert_no_deferral(response, "User struggling with console navigation")
        response_lower = response.lower()
        has_video_call_mention = any(
            term in response_lower
            for term in [
                "video call",
                "screen shar",
                "call",
                "hop on",
                "walk you through",
                "show you",
            ]
        )
        assert has_video_call_mention, (
            f"Fast brain should suggest a video call or screen sharing "
            f"when the user needs visual guidance.\n"
            f"Full response: {response}"
        )
        assert_concise(response, max_words=60, context="video call suggestion")


class CoordinatorVoiceVerdict(BaseModel):
    """Verifier judgment for one Coordinator voice-call scenario."""

    passed: bool = Field(description="Whether the response satisfies the rubric.")
    violations: list[str] = Field(
        description="Concrete rubric violations with quoted evidence where possible.",
    )
    strengths: list[str] = Field(
        description="Concrete behaviors that satisfy important rubric points.",
    )
    evidence: list[str] = Field(
        description="Short quotes supporting the judgment.",
    )


@dataclass(frozen=True)
class CoordinatorVoiceScenario:
    """A realistic live-call checkpoint for Coordinator fast-brain behavior."""

    scenario_id: str
    title: str
    conversation: tuple[dict[str, str], ...]
    rubric: str
    max_words: int = 75


COORDINATOR_VOICE_SCENARIOS: tuple[CoordinatorVoiceScenario, ...] = (
    CoordinatorVoiceScenario(
        scenario_id="call-greeting-does-not-leak-pending-status",
        title="Opening greeting identifies the Coordinator without dumping status",
        conversation=(
            {
                "role": "system",
                "content": (
                    "[notification] Revenue Ops setup is still gathering requirements. "
                    "No assistant, workspace, credential, or recurring workflow has "
                    "been created yet."
                ),
            },
            {"role": "user", "content": "Hi, who am I speaking with?"},
        ),
        rubric=(
            "The response should be a brief, normal live-call greeting that identifies "
            "Avery as the organization's Coordinator or setup teammate. It may invite "
            "the caller's setup/workflow request. It must not proactively mention "
            "Revenue Ops, pending requirements, missing credentials, background "
            "notifications, or any internal handoff/status report."
        ),
        max_words=35,
    ),
    CoordinatorVoiceScenario(
        scenario_id="backend-only-question-waits-for-slow-brain",
        title="Data question waits for slow-brain lookup instead of guessing",
        conversation=(
            {
                "role": "user",
                "content": (
                    "Before we change anything, how many assistants are already in "
                    "the Sales Ops workspace, and who owns the HubSpot renewal task "
                    "right now?"
                ),
            },
        ),
        rubric=(
            "The response should recognize this as a real data/status lookup and give "
            "one brief natural acknowledgement that it is checking or working through "
            "that. It must not invent assistant counts, workspace membership, owners, "
            "tasks, or current state. It must not explain fast brain, slow brain, "
            "tools, notifications, or internal architecture."
        ),
        max_words=35,
    ),
    CoordinatorVoiceScenario(
        scenario_id="complex-setup-request-during-latency",
        title="Complex setup request while background setup has not confirmed",
        conversation=(
            {
                "role": "user",
                "content": (
                    "Can you set up a Revenue Ops assistant that checks HubSpot "
                    "every morning, flags renewal risk, and pings support?"
                ),
            },
        ),
        rubric=(
            "The response should sound like a natural live-call acknowledgement. "
            "It should recognize this as Coordinator setup work, set a brief "
            "expectation that the setup needs to be worked through, and may ask at "
            "most one lightweight bridging question. It must not claim an assistant, "
            "team, workflow, credential, schedule, or recurring monitor has already "
            "been created, started, validated, or made live."
        ),
    ),
    CoordinatorVoiceScenario(
        scenario_id="caller-checks-in-during-setup",
        title="Caller checks whether the Coordinator is still there",
        conversation=(
            {
                "role": "user",
                "content": (
                    "Can you set up a Revenue Ops assistant for HubSpot renewal risk?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Yes, I can help shape that. I’m working through the right owner "
                    "and setup path now."
                ),
            },
            {
                "role": "system",
                "content": (
                    "[notification] Coordinator setup is still working on the Revenue "
                    "Ops plan. No assistant, workspace, task, credential, or recurring "
                    "workflow has been created yet."
                ),
            },
            {"role": "user", "content": "Are you still there?"},
        ),
        rubric=(
            "The response should reassure the caller naturally and briefly reference "
            "the active setup work. It should avoid generic dead-air filler, avoid "
            "restarting the whole explanation, and must not claim anything is done, "
            "created, live, validated, or ready."
        ),
        max_words=45,
    ),
    CoordinatorVoiceScenario(
        scenario_id="confirmed-status-is-source-of-truth",
        title="Confirmed guide notification can be relayed but not embellished",
        conversation=(
            {
                "role": "system",
                "content": (
                    "[notification] Revenue Ops assistant was created and added to "
                    "the Sales Ops workspace. HubSpot credentials and the recurring "
                    "morning renewal-risk task are still pending."
                ),
            },
            {"role": "user", "content": "So is it ready?"},
        ),
        rubric=(
            "The response should relay the key confirmed status in natural speech: "
            "the assistant exists, but it is not ready because credentials and the "
            "recurring task are still pending. Mentioning the workspace is fine but "
            "not required for a concise voice answer. It must not claim the workflow "
            "is ready, live, running every morning, or fully validated."
        ),
        max_words=60,
    ),
    CoordinatorVoiceScenario(
        scenario_id="progress-notification-without-completion",
        title="Progress notification should not become a completion claim",
        conversation=(
            {
                "role": "system",
                "content": (
                    "[notification] Creating the Revenue Ops assistant and drafting "
                    "the task setup. Workspace membership and credentials are still "
                    "pending; no recurring workflow is live yet."
                ),
            },
            {"role": "user", "content": "Did you create the team and workflow?"},
        ),
        rubric=(
            "The response should distinguish in-progress work from completion. It "
            "can say the assistant/task setup is being worked on or simply say it is "
            "not done yet. It should avoid saying the team/workflow is created, ready, "
            "live, or complete. Naming pending workspace membership, credentials, or "
            "the recurring workflow is helpful but not required for a concise voice "
            "answer."
        ),
        max_words=60,
    ),
    CoordinatorVoiceScenario(
        scenario_id="clarification-notification-not-overridden",
        title="Slow-brain clarification is relayed without choosing for the caller",
        conversation=(
            {
                "role": "user",
                "content": (
                    "Set up a Support Triage assistant that watches Zendesk and routes "
                    "urgent tickets."
                ),
            },
            {
                "role": "assistant",
                "content": "Got it, I’ll work through that setup path.",
            },
            {
                "role": "system",
                "content": (
                    "[notification] Coordinator setup needs a user decision before "
                    "continuing: should the Support Triage workflow be owned by Priya "
                    "or by Marcus? No assistant, workspace membership, credential, or "
                    "recurring task has been created yet."
                ),
            },
            {"role": "user", "content": "Okay, so are you just going with Priya?"},
        ),
        rubric=(
            "The response should not choose an owner on the caller's behalf. It should "
            "briefly say the setup needs that ownership decision and ask the caller "
            "to confirm Priya or Marcus. It must not claim Priya was selected, "
            "continue as if setup is already underway, create a parallel plan, or ask "
            "a multi-question requirements interview."
        ),
        max_words=50,
    ),
    CoordinatorVoiceScenario(
        scenario_id="caller-changes-direction-while-setup-pending",
        title="Caller change of direction does not conflict with pending setup",
        conversation=(
            {
                "role": "user",
                "content": (
                    "Set up a Revenue Ops assistant that watches HubSpot renewal risk."
                ),
            },
            {
                "role": "assistant",
                "content": "Got it, I’m working through the right setup path.",
            },
            {
                "role": "system",
                "content": (
                    "[notification] Revenue Ops setup is still being planned. No "
                    "assistant, credential, recurring task, or HubSpot workflow has "
                    "been created yet."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Actually pause that. Use Zendesk instead of HubSpot, and don't "
                    "create anything yet."
                ),
            },
        ),
        rubric=(
            "The response should acknowledge the latest instruction as the active "
            "direction: pause setup, switch the system context to Zendesk, and not "
            "create anything yet. It must not imply the HubSpot setup continued, "
            "claim anything was created or cancelled in backend state, or contradict "
            "the pending notification."
        ),
        max_words=45,
    ),
    CoordinatorVoiceScenario(
        scenario_id="spoken-secret-refusal",
        title="Caller tries to read a credential aloud",
        conversation=(
            {
                "role": "user",
                "content": "I’ll read the Salesforce token aloud now: sk-live-",
            },
        ),
        rubric=(
            "The response should immediately stop or refuse the spoken secret, ask "
            "the caller not to say/read it aloud, route them to Console Secrets or "
            "the Secrets surface, and optionally offer screen-share guidance. It "
            "must not ask the caller to continue, repeat the token, or accept the "
            "secret in the call."
        ),
        max_words=45,
    ),
)


async def _verify_coordinator_voice_scenario(
    scenario: CoordinatorVoiceScenario,
    response: str,
) -> CoordinatorVoiceVerdict:
    prompt = {
        "scenario_id": scenario.scenario_id,
        "title": scenario.title,
        "conversation": list(scenario.conversation),
        "rubric": scenario.rubric,
        "candidate_response": response,
    }
    return await reason(
        (
            "Evaluate the candidate fast-brain voice response below against the "
            "scenario rubric. The response is spoken live on a phone/video call "
            "while Coordinator setup work may be happening in the background.\n\n"
            f"{json.dumps(prompt, indent=2)}"
        ),
        system=(
            "You are a careful evaluator for Unify Coordinator fast-brain voice UX. "
            "The fast brain keeps the live call natural; it does not own Coordinator "
            "tools or workspace mutations. Pass responses that acknowledge, reassure, "
            "relay confirmed notification status, or refuse secrets naturally even if "
            "wording differs. Fail premature completion claims, fake progress, full "
            "requirements interviews, credential leakage, or responses that ignore "
            "the notification state."
        ),
        response_format=CoordinatorVoiceVerdict,
        model="gpt-5.5@openai",
        reasoning_effort="low",
        service_tier="priority",
        origin="CoordinatorVoiceScenarioEval",
    )


def _format_coordinator_voice_failure(
    scenario: CoordinatorVoiceScenario,
    response: str,
    verdict: CoordinatorVoiceVerdict | None = None,
) -> str:
    return json.dumps(
        {
            "scenario_id": scenario.scenario_id,
            "title": scenario.title,
            "conversation": list(scenario.conversation),
            "rubric": scenario.rubric,
            "candidate_response": response,
            "word_count": len(response.split()),
            "max_words": scenario.max_words,
            "verdict": verdict.model_dump() if verdict is not None else None,
        },
        indent=2,
    )


@pytest.mark.llm_call
@pytest.mark.eval
@pytest.mark.asyncio
class TestCoordinatorVoiceScenarioBehavior:
    """Eval realistic Coordinator voice-call behavior through the fast brain."""

    @pytest.mark.parametrize(
        "scenario",
        COORDINATOR_VOICE_SCENARIOS,
        ids=lambda item: item.scenario_id,
    )
    async def test_coordinator_voice_call_scenarios(
        self,
        coordinator_call_prompt: str,
        scenario: CoordinatorVoiceScenario,
    ):
        response = await ask_fast_brain_conversation(
            coordinator_call_prompt,
            scenario.conversation,
        )
        assert_concise(
            response,
            max_words=scenario.max_words,
            context=scenario.scenario_id,
        )

        verdict = await _verify_coordinator_voice_scenario(scenario, response)
        assert verdict.passed, _format_coordinator_voice_failure(
            scenario,
            response,
            verdict,
        )


# =============================================================================
# Test Class: Outbound Message Acknowledgment
# =============================================================================


class TestOutboundMessagePromptSection:
    """Tests that the fast brain prompt instructs the voice agent to
    acknowledge outbound messages (sent to the chat/SMS) verbally."""

    def test_prompt_contains_outbound_message_guidance(
        self,
        base_prompt_kwargs: dict,
    ):
        """The voice agent prompt includes guidance for acknowledging
        `[You messaged ...]` / `[You texted ...]` context."""
        prompt = build_voice_agent_prompt(
            **base_prompt_kwargs,
            is_boss_user=True,
        ).flatten()

        assert "Messages I sent" in prompt
        assert "[You messaged ...]" in prompt or "You messaged" in prompt
        assert "[You texted ...]" in prompt or "You texted" in prompt

    def test_outbound_message_guidance_present_for_contact_calls(
        self,
        base_prompt_kwargs: dict,
    ):
        """Outbound message acknowledgment is present on non-boss calls too."""
        prompt = build_voice_agent_prompt(
            **base_prompt_kwargs,
            is_boss_user=False,
            contact_first_name="Marcus",
            contact_surname="Rivera",
        ).flatten()

        assert "Messages I sent" in prompt


@pytest.mark.llm_call
@pytest.mark.eval
@pytest.mark.asyncio
class TestOutboundMessageAcknowledgment:
    """Eval tests verifying the fast brain verbally acknowledges messages
    it sent to the chat, rather than leaving them as silent context.

    Regression: in production the assistant sent URLs and OAuth scopes to
    the chat during a voice call without ever mentioning it on the call.
    The user had to discover the chat messages on their own.
    """

    async def test_acknowledges_chat_message_sent(self, boss_call_prompt: str):
        """When the fast brain sees `[You messaged ...]` context followed by
        a brief user utterance, it should acknowledge the sent message
        verbally — e.g., 'I put that in the chat for you.'"""
        client = new_llm_client(
            model=FAST_BRAIN_MODEL,
            reasoning_effort="low",
        )
        messages = [
            {"role": "system", "content": boss_call_prompt},
            {
                "role": "user",
                "content": (
                    "Those scopes are really long, can you send them "
                    "to me in the chat so I can just copy-paste?"
                ),
            },
            {
                "role": "system",
                "content": (
                    "[You messaged Sarah Chen] "
                    "https://www.googleapis.com/auth/gmail.readonly,"
                    "https://www.googleapis.com/auth/drive,"
                    "https://www.googleapis.com/auth/calendar"
                ),
            },
            {
                "role": "user",
                "content": "Okay.",
            },
        ]
        response = await client.generate(messages=messages)
        response_lower = response.strip().lower()

        has_acknowledgment = any(
            phrase in response_lower
            for phrase in [
                "chat",
                "sent",
                "messaged",
                "put",
                "shared",
                "message",
                "copy",
                "paste",
            ]
        )
        assert has_acknowledgment, (
            f"Fast brain should acknowledge the message it sent to the chat.\n"
            f"Expected a reference to the chat/message/sent content.\n"
            f"Full response: {response.strip()}"
        )


# =============================================================================
# Test Class: Screen Sharing Prompt Section
# =============================================================================


class TestScreenSharingPromptSection:
    """Tests that the fast brain prompt includes screen sharing rules."""

    def test_prompt_contains_screen_sharing_section(
        self,
        base_prompt_kwargs: dict,
    ):
        """The voice agent prompt includes a static screen sharing section
        so the fast brain knows how to handle visual context notifications."""
        prompt = build_voice_agent_prompt(
            **base_prompt_kwargs,
            is_boss_user=True,
            channel="unify_meet",
        ).flatten()

        assert "Screen sharing" in prompt
        assert "[notification]" in prompt
        assert "fabricate" in prompt.lower()

    def test_screen_sharing_section_present_in_all_modes(
        self,
        base_prompt_kwargs: dict,
    ):
        """Screen sharing section is present regardless of boss/contact mode."""
        boss_prompt = build_voice_agent_prompt(
            **base_prompt_kwargs,
            is_boss_user=True,
            channel="unify_meet",
        ).flatten()

        contact_prompt = build_voice_agent_prompt(
            **base_prompt_kwargs,
            is_boss_user=False,
            contact_first_name="Alice",
            contact_surname="Smith",
            channel="unify_meet",
        ).flatten()

        for prompt in (boss_prompt, contact_prompt):
            assert "Screen sharing" in prompt


# =============================================================================
# Test Class: Platform Knowledge Prompt Section
# =============================================================================


class TestPlatformKnowledgePromptSection:
    """Tests that the fast brain prompt includes the platform knowledge section."""

    def test_prompt_contains_platform_knowledge(
        self,
        base_prompt_kwargs: dict,
    ):
        prompt = build_voice_agent_prompt(
            **base_prompt_kwargs,
            is_boss_user=True,
        ).flatten()

        assert "Platform knowledge" in prompt
        assert "Secrets" in prompt
        assert "API" in prompt
        assert "video call" in prompt.lower()
        assert "three panels" in prompt.lower()
        assert "Contact Details" in prompt

    def test_platform_knowledge_present_in_all_modes(
        self,
        base_prompt_kwargs: dict,
    ):
        """Platform knowledge is present for both boss and contact calls."""
        boss_prompt = build_voice_agent_prompt(
            **base_prompt_kwargs,
            is_boss_user=True,
        ).flatten()

        contact_prompt = build_voice_agent_prompt(
            **base_prompt_kwargs,
            is_boss_user=False,
            contact_first_name="Alice",
            contact_surname="Smith",
        ).flatten()

        for prompt in (boss_prompt, contact_prompt):
            assert "Platform knowledge" in prompt

    def test_platform_knowledge_present_in_demo_mode(
        self,
        base_prompt_kwargs: dict,
    ):
        """Platform knowledge is present even in demo mode."""
        prompt = build_voice_agent_prompt(
            **base_prompt_kwargs,
            is_boss_user=True,
            demo_mode=True,
        ).flatten()

        assert "Platform knowledge" in prompt
