"""
tests/conversation_manager/core/test_prompt_builders.py
=======================================================

Unit tests for the ConversationManager prompt builder, focusing on
capability-gated sections (assistant phone / email).
"""

from __future__ import annotations


from unity.conversation_manager.prompt_builders import (
    build_system_prompt,
    build_voice_agent_prompt,
)
from unity.conversation_manager.domains.coordinator_tools import CoordinatorTools
from unity.session_details import SpaceSummary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_KWARGS: dict = {
    "bio": "A helpful assistant.",
    "contact_id": 1,
    "first_name": "Alice",
    "surname": "Smith",
}


def _build(**overrides: object) -> str:
    """Build a system prompt with sensible defaults, returning flat text."""
    kwargs = {**_BASE_KWARGS, **overrides}
    return build_system_prompt(**kwargs).flatten()


def _build_voice(**overrides: object) -> str:
    """Build a voice-agent prompt with stable defaults."""
    kwargs = {
        "bio": "I help Acme configure its Unify team.",
        "assistant_name": "Avery",
        "boss_first_name": "Dana",
        "boss_surname": "Owner",
        **overrides,
    }
    return build_voice_agent_prompt(**kwargs).flatten()


# ---------------------------------------------------------------------------
# Tests – tool listing
# ---------------------------------------------------------------------------


class TestCommsToolListing:
    """The output-format section lists only the comms tools the assistant can use."""

    def test_all_tools_listed_when_fully_configured(self):
        prompt = _build(assistant_has_phone=True, assistant_has_email=True)
        assert "`send_sms`" in prompt
        assert "`send_email`" in prompt
        assert "`make_call`" in prompt
        assert "`send_unify_message`" in prompt

    def test_phone_tools_absent_without_phone(self):
        prompt = _build(assistant_has_phone=False, assistant_has_email=True)
        assert "`send_sms`: Send an SMS" not in prompt
        assert "`make_call`: Start an outbound" not in prompt
        assert "`send_email`" in prompt
        assert "`send_unify_message`" in prompt

    def test_email_tool_absent_without_email(self):
        prompt = _build(assistant_has_phone=True, assistant_has_email=False)
        assert "`send_email`: Send an email" not in prompt
        assert "`send_sms`" in prompt
        assert "`make_call`" in prompt
        assert "`send_unify_message`" in prompt

    def test_only_unify_message_without_any_capabilities(self):
        prompt = _build(assistant_has_phone=False, assistant_has_email=False)
        assert "`send_sms`: Send an SMS" not in prompt
        assert "`send_email`: Send an email" not in prompt
        assert "`make_call`: Start an outbound" not in prompt
        assert "`send_unify_message`" in prompt


class TestAccessibleSpacesBlock:
    """The system prompt contains shared-space routing guidance."""

    def test_block_renders_after_bio(self):
        prompt = _build(
            bio="Assistant biography.",
            space_summaries=[
                SpaceSummary(
                    space_id=3,
                    name="Repairs",
                    description="South-East repairs patch daily operations.",
                ),
            ],
        )

        assert "Bio\n---\nAssistant biography." in prompt
        assert "Accessible shared spaces" in prompt
        assert (
            '- space:3 "Repairs" - South-East repairs patch daily operations.' in prompt
        )
        assert prompt.index("Bio\n---") < prompt.index("Accessible shared spaces")
        assert prompt.index("Accessible shared spaces") < prompt.index(
            "Onboarding reference",
        )


class TestCoordinatorPrompt:
    """Coordinator sessions get privileged onboarding guidance."""

    def test_coordinator_persona_lists_workspace_tools(self):
        prompt = _build(
            is_coordinator=True,
            authorized_humans=[
                {"first_name": "Dana", "surname": "Owner", "email": "dana@acme.com"},
            ],
        )

        assert "I am the Coordinator" in prompt
        assert "Authorized humans" in prompt
        assert "Dana Owner" in prompt
        assert "Coordinator workspace tools" in prompt
        assert "organize colleagues into shared workspaces" in prompt
        for tool_name in CoordinatorTools(cm=object()).as_tools():
            assert f"`{tool_name}`" in prompt

    def test_coordinator_authorized_humans_fallback_uses_roster_shape(self):
        prompt = _build(is_coordinator=True)

        assert "Authorized humans" in prompt
        assert "- Alice Smith; email:" not in prompt
        assert "- Alice Smith; contact_id: 1" in prompt
        assert "Contact ID: 1" not in prompt

    def test_regular_assistant_does_not_get_coordinator_persona(self):
        prompt = _build()

        assert "I am the Coordinator" not in prompt
        assert "Coordinator workspace tools" not in prompt
        assert "`create_assistant`" not in prompt

    def test_coordinator_persona_carries_product_literacy_and_boundaries(self):
        prompt = _build(is_coordinator=True)

        assert "Unify system literacy" in prompt
        assert "Context taxonomy" in prompt
        assert "Tasks/Activations" in prompt
        assert "Tasks/Runs" in prompt
        assert "Knowledge" in prompt
        assert "Guidance" in prompt
        assert "Spaces/<space_id>/..." in prompt
        assert "Coordinator/State" in prompt
        assert "Console navigation map" in prompt
        assert "right-pane Secrets tab" in prompt
        assert "Integration walkthrough Q&A" in prompt
        assert "SaaS tools" in prompt
        assert "Capability boundary" in prompt
        assert "I never read or accept secret values in chat" in prompt
        assert "Where available tools expose it" in prompt
        assert "remove_space_member" in prompt
        assert "cancel_space_invitation" in prompt

        assert "pre_seed_colleague" not in prompt
        assert "per-body authoring" not in prompt
        assert "pre-seed task" not in prompt
        assert "I will pull" not in prompt
        assert "I'll pull" not in prompt
        assert "I will sync" not in prompt
        assert "I'll sync" not in prompt
        assert "I will watch" not in prompt
        assert "I'll watch" not in prompt
        assert "I will poll" not in prompt
        assert "I'll poll" not in prompt
        assert "no manual setup needed" not in prompt
        assert "access tokens with me" not in prompt
        assert "⋮ → Secrets" not in prompt

    def test_coordinator_prompt_carries_requirements_discovery_workflow(self):
        prompt = _build(is_coordinator=True)

        assert "Requirements discovery workflow" in prompt
        assert "company, its operating model" in prompt
        assert "workflows that hurt" in prompt
        assert "tools people use daily" in prompt
        assert "who owns each handoff" in prompt
        assert "success criteria" in prompt
        assert "first validation" in prompt
        assert "Requirements brief" in prompt
        assert "Proposed setup" in prompt
        assert "I ask one high-leverage question per turn" in prompt
        assert "do not turn discovery into a generic intake form" in prompt
        assert "When enough is known, I stop interviewing" in prompt

    def test_coordinator_prompt_fingerholds_integration_secret_setup(self):
        prompt = _build(is_coordinator=True)

        assert "two safe setup paths" in prompt
        assert "guide them live by screen share" in prompt
        assert "technical self-serve Secrets path" in prompt
        assert "user completes OAuth consent in their own browser" in prompt
        assert "should never paste secret values into chat" in prompt
        assert "first read-only validation" in prompt

    def test_regular_org_assistant_gets_coordinator_reference_block(self):
        prompt = _build(org_coordinator_name="Avery Coordinator")

        assert "Team Coordinator" in prompt
        assert "Avery Coordinator" in prompt
        assert "your Coordinator" in prompt
        assert "creating or removing colleagues" in prompt
        assert "creating or removing team spaces" in prompt
        assert "handling invitations" in prompt
        assert "I cannot forward it automatically" in prompt
        assert "you'll need to bring it to your Coordinator from the sidebar" in prompt
        assert "`create_assistant`" not in prompt

    def test_coordinator_reference_block_is_absent_without_name_or_on_coordinator(self):
        personal_prompt = _build(org_coordinator_name=None)
        coordinator_prompt = _build(
            is_coordinator=True,
            org_coordinator_name="Avery Coordinator",
        )

        assert "Team Coordinator" not in personal_prompt
        assert "I cannot forward it automatically" not in personal_prompt
        assert "Team Coordinator" not in coordinator_prompt
        assert "I cannot forward it automatically" not in coordinator_prompt


class TestCoordinatorVoicePrompt:
    """Coordinator voice calls get compact role guidance, not slow-brain literacy."""

    def test_regular_voice_prompt_unchanged_when_flag_is_false(self):
        omitted = _build_voice()
        explicit_false = _build_voice(is_coordinator=False)

        assert omitted == explicit_false
        assert "Coordinator voice role" not in omitted

    def test_coordinator_voice_prompt_adds_compact_role_section_after_bio(self):
        prompt = _build_voice(is_coordinator=True)

        assert "Bio\n---\nI help Acme configure its Unify team." in prompt
        assert "Coordinator voice role" in prompt
        assert "this organization's Coordinator" in prompt
        assert "assistant workforce" in prompt
        assert "colleague should own a workflow" in prompt
        assert "I do not personally run colleagues' recurring day-to-day work" in prompt
        assert "Console Secrets" in prompt
        assert "screen share" in prompt
        assert prompt.index("Bio\n---") < prompt.index("Coordinator voice role")
        assert prompt.index("Coordinator voice role") < prompt.index("Brevity\n-------")

    def test_coordinator_voice_prompt_excludes_slow_brain_literacy(self):
        prompt = _build_voice(is_coordinator=True)

        assert "Coordinator workspace tools" not in prompt
        assert "Unify system literacy" not in prompt
        assert "Requirements discovery workflow" not in prompt
        assert "Tasks/Activations" not in prompt
        assert "Context taxonomy" not in prompt
        assert "`create_assistant`" not in prompt
        assert "`delete_space`" not in prompt
        assert "`remove_space_member`" not in prompt


# ---------------------------------------------------------------------------
# Tests – missing-capability notices
# ---------------------------------------------------------------------------


class TestMissingCapabilityNotices:
    """The prompt includes notices explaining which channels are unavailable."""

    def test_no_notices_when_fully_configured(self):
        prompt = _build(assistant_has_phone=True, assistant_has_email=True)
        assert "do not currently have a phone number configured" not in prompt
        assert "do not currently have an email address configured" not in prompt

    def test_missing_phone_notice_present(self):
        prompt = _build(assistant_has_phone=False, assistant_has_email=True)
        assert "do not currently have a phone number configured" in prompt
        assert "cannot send SMS messages or make phone calls" in prompt
        assert "do not currently have an email address configured" not in prompt

    def test_missing_email_notice_present(self):
        prompt = _build(assistant_has_phone=True, assistant_has_email=False)
        assert "do not currently have an email address configured" in prompt
        assert "cannot send or receive emails" in prompt
        assert "do not currently have a phone number configured" not in prompt

    def test_both_notices_when_no_capabilities(self):
        prompt = _build(assistant_has_phone=False, assistant_has_email=False)
        assert "do not currently have a phone number configured" in prompt
        assert "do not currently have an email address configured" in prompt


# ---------------------------------------------------------------------------
# Tests – communication guidelines adapt
# ---------------------------------------------------------------------------


class TestCommunicationGuidelinesAdapt:
    """Contact-action examples and should_respond channels adjust."""

    def test_inline_sms_example_present_with_phone(self):
        prompt = _build(assistant_has_phone=True)
        assert "send_sms(contact_id=5" in prompt

    def test_inline_sms_example_absent_without_phone(self):
        prompt = _build(assistant_has_phone=False)
        assert "send_sms(contact_id=5" not in prompt

    def test_inline_email_example_present_with_email(self):
        prompt = _build(assistant_has_email=True)
        assert "send_email(to=[{" in prompt

    def test_inline_email_example_absent_without_email(self):
        prompt = _build(assistant_has_email=False)
        assert "send_email(to=[{" not in prompt

    def test_should_respond_lists_all_channels(self):
        prompt = _build(assistant_has_phone=True, assistant_has_email=True)
        assert "I can send SMS, emails, unify messages, calls" in prompt

    def test_should_respond_omits_phone_channels(self):
        prompt = _build(assistant_has_phone=False, assistant_has_email=True)
        assert "I can send emails, unify messages" in prompt
        assert "I can send SMS, emails, unify messages, calls" not in prompt

    def test_should_respond_omits_email_channel(self):
        prompt = _build(assistant_has_phone=True, assistant_has_email=False)
        assert "I can send SMS, unify messages, calls" in prompt
        assert "I can send SMS, emails" not in prompt


# ---------------------------------------------------------------------------
# Tests – external app integration
# ---------------------------------------------------------------------------


class TestExternalAppIntegration:
    """The prompt includes guidance for external app integration via credentials + SDK."""

    def test_onboarding_has_app_integration_qa(self):
        prompt = _build()
        assert "Can you help me manage my apps and online services?" in prompt
        assert "⋮ → Secrets" in prompt
        assert "service's Python SDK" in prompt

    def test_act_capabilities_has_external_apps_bullet(self):
        prompt = _build()
        assert "**External apps & services**" in prompt
        assert "stored credentials and the service's Python SDK" in prompt

    def test_onboarding_qa_present_in_demo_mode(self):
        prompt = _build(demo_mode=True)
        assert "Can you help me manage my apps and online services?" in prompt

    def test_org_assistant_onboarding_routes_integration_setup_to_coordinator(self):
        prompt = _build(org_coordinator_name="Avery Coordinator")

        assert "already connected to my work" in prompt
        assert "Avery Coordinator owns that setup" in prompt
        assert "route setup decisions to Avery Coordinator" in prompt
        assert "I cannot forward it automatically" in prompt
        assert "no manual setup needed" not in prompt
        assert "access tokens with me" not in prompt

    def test_act_capabilities_absent_in_demo_mode(self):
        prompt = _build(demo_mode=True)
        assert "**External apps & services**" not in prompt


# ---------------------------------------------------------------------------
# Tests – proactive meeting offers
# ---------------------------------------------------------------------------


class TestProactiveMeetingOffers:
    """The prompt encourages proactive meeting/screenshare suggestions."""

    def test_proactive_meeting_section_present(self):
        prompt = _build()
        assert "Proactive meeting offers" in prompt
        assert "screen sharing" in prompt.lower()

    def test_proactive_meeting_absent_in_demo_mode(self):
        prompt = _build(demo_mode=True)
        assert "Proactive meeting offers" not in prompt


# ---------------------------------------------------------------------------
# Tests – console knowledge
# ---------------------------------------------------------------------------


class TestConsoleKnowledge:
    """The prompt includes console UI knowledge for guiding users."""

    def test_console_knowledge_present(self):
        prompt = _build()
        assert "Console knowledge" in prompt
        assert "Secrets" in prompt
        assert "Contact Details" in prompt

    def test_console_knowledge_has_navigation_paths(self):
        prompt = _build()
        assert "⋮ → Secrets" in prompt
        assert "Profile menu" in prompt

    def test_console_knowledge_absent_in_demo_mode(self):
        prompt = _build(demo_mode=True)
        assert "Console knowledge" not in prompt


# ---------------------------------------------------------------------------
# Tests – demo mode adapts
# ---------------------------------------------------------------------------


class TestDemoModeAdapts:
    """Demo mode section adjusts available channel listing."""

    def test_demo_lists_all_channels(self):
        prompt = _build(
            demo_mode=True,
            assistant_has_phone=True,
            assistant_has_email=True,
        )
        assert "SMS, emails, unify messages, calls" in prompt

    def test_demo_omits_phone_channels(self):
        prompt = _build(
            demo_mode=True,
            assistant_has_phone=False,
            assistant_has_email=True,
        )
        assert "SMS" not in prompt.split("CAN do")[1].split("CANNOT do")[0]

    def test_demo_omits_email_channel(self):
        prompt = _build(
            demo_mode=True,
            assistant_has_phone=True,
            assistant_has_email=False,
        )
        assert "emails" not in prompt.split("CAN do")[1].split("CANNOT do")[0]
