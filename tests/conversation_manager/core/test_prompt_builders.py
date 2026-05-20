"""
tests/conversation_manager/core/test_prompt_builders.py
=======================================================

Unit tests for the ConversationManager prompt builder, focusing on
capability-gated sections (assistant phone / email).
"""

from __future__ import annotations

import pytest

from unity.conversation_manager.prompt_builders import (
    build_system_prompt,
    build_voice_agent_prompt,
)
from unity.session_details import SpaceSummary

pytestmark = pytest.mark.no_unify_context

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
    """Coordinator sessions use a unified base prompt plus org-context surfaces."""

    def test_org_coordinator_prompt_lists_org_roster_and_workspace_tools(self):
        prompt = _build(
            is_coordinator=True,
            authorized_humans=[
                {
                    "first_name": "Dana",
                    "surname": "Owner",
                    "email": "dana@acme.com",
                    "is_admin": True,
                },
                {
                    "first_name": "Francis",
                    "surname": "Lead",
                    "email": "francis@acme.com",
                    "is_admin": False,
                },
            ],
        )

        assert "Authorized humans" in prompt
        assert "Dana Owner; email: dana@acme.com; role: admin" in prompt
        assert "Francis Lead; email: francis@acme.com; role: member" in prompt
        assert "**Coordinator workspace tools:**" in prompt
        assert (
            "`primitives.coordinator.list_org_members` and "
            "`primitives.coordinator.invite_org_member` are available for org membership changes"
            in prompt
        )
        assert "Team Coordinator\n----------------" not in prompt

    def test_personal_coordinator_uses_boss_details_and_hides_org_tools(self):
        prompt = _build(is_coordinator=True, is_org_workspace=False)

        assert "Boss details" in prompt
        assert "Authorized humans" not in prompt
        assert "list_org_members" not in prompt
        assert "invite_org_member" not in prompt

    def test_regular_assistant_gets_updated_coordinator_reference_block(self):
        prompt = _build(workspace_coordinator_name="Avery Coordinator")

        assert "Team Coordinator" in prompt
        assert "Escalate to Avery Coordinator" in prompt
        assert "create or remove colleagues" in prompt
        assert "create or remove team spaces" in prompt
        assert "add or remove space members" in prompt
        assert "invite org members" in prompt
        assert "I cannot forward it automatically" not in prompt

    def test_coordinator_reference_block_is_absent_without_name_or_on_coordinator(self):
        personal_prompt = _build(workspace_coordinator_name=None)
        coordinator_prompt = _build(
            is_coordinator=True,
            workspace_coordinator_name="Avery Coordinator",
        )

        assert "Team Coordinator" not in personal_prompt
        assert "Team Coordinator" not in coordinator_prompt

    def test_coordinator_prompt_removes_legacy_setup_sections(self):
        prompt = _build(is_coordinator=True)

        for marker in (
            "Coordinator workspace ontology",
            "Requirements discovery workflow",
            "Coordinator checklist and state usage",
            "Coordinator disposition",
            "Coordinator conversational cadence",
            "Coordinator parallel tool discipline",
            "Unify system literacy",
        ):
            assert marker not in prompt
        assert "Conversational restraint" in prompt
        assert "Intent vs verified outcomes" in prompt
        assert "Parallel tool discipline" in prompt

    def test_base_and_coordinator_share_restraint_but_keep_role_specific_sections(self):
        base_prompt = _build()
        coordinator_prompt = _build(is_coordinator=True)

        assert "Intent vs verified outcomes" in base_prompt
        assert "Intent vs verified outcomes" in coordinator_prompt
        assert "Console knowledge" in base_prompt
        assert "Console knowledge" not in coordinator_prompt
        assert "Concurrent action and acknowledgment" in base_prompt
        assert "Concurrent action and acknowledgment" not in coordinator_prompt


class TestPromptSectionOwnershipMatrix:
    """Role/mode/org combinations keep prompt section ownership boundaries stable."""

    def test_system_prompt_section_ownership_matrix(self):
        cases = (
            {
                "name": "regular_non_demo_no_org",
                "kwargs": {},
                "present": (
                    "Act capabilities\n----------------",
                    "Concurrent action and acknowledgment\n------------------------------------",
                ),
                "absent": (
                    "**Coordinator workspace tools:**",
                    "Authorized humans\n-----------------",
                    "Team Coordinator\n----------------",
                    "Demo mode\n---------",
                ),
            },
            {
                "name": "regular_non_demo_with_org",
                "kwargs": {"workspace_coordinator_name": "Avery Coordinator"},
                "present": (
                    "Act capabilities\n----------------",
                    "Concurrent action and acknowledgment\n------------------------------------",
                    "Team Coordinator\n----------------",
                ),
                "absent": (
                    "**Coordinator workspace tools:**",
                    "Authorized humans\n-----------------",
                    "Demo mode\n---------",
                ),
            },
            {
                "name": "regular_demo_no_org",
                "kwargs": {"demo_mode": True},
                "present": ("Demo mode\n---------",),
                "absent": (
                    "**Coordinator workspace tools:**",
                    "Authorized humans\n-----------------",
                    "Act capabilities\n----------------",
                    "Team Coordinator\n----------------",
                    "Concurrent action and acknowledgment\n------------------------------------",
                ),
            },
            {
                "name": "regular_demo_with_org",
                "kwargs": {
                    "demo_mode": True,
                    "workspace_coordinator_name": "Avery Coordinator",
                },
                "present": (
                    "Demo mode\n---------",
                    "Team Coordinator\n----------------",
                ),
                "absent": (
                    "**Coordinator workspace tools:**",
                    "Authorized humans\n-----------------",
                    "Act capabilities\n----------------",
                    "Concurrent action and acknowledgment\n------------------------------------",
                ),
            },
            {
                "name": "coordinator_non_demo_with_org",
                "kwargs": {
                    "is_coordinator": True,
                    "workspace_coordinator_name": "Avery Coordinator",
                },
                "present": (
                    "**Coordinator workspace tools:**",
                    "Authorized humans\n-----------------",
                    "Act capabilities\n----------------",
                ),
                "absent": (
                    "Team Coordinator\n----------------",
                    "Demo mode\n---------",
                    "Concurrent action and acknowledgment\n------------------------------------",
                ),
            },
            {
                "name": "coordinator_demo_with_org",
                "kwargs": {
                    "is_coordinator": True,
                    "demo_mode": True,
                    "workspace_coordinator_name": "Avery Coordinator",
                },
                "present": (
                    "Authorized humans\n-----------------",
                    "Demo mode\n---------",
                ),
                "absent": (
                    "**Coordinator workspace tools:**",
                    "Team Coordinator\n----------------",
                    "Act capabilities\n----------------",
                    "Concurrent action and acknowledgment\n------------------------------------",
                ),
            },
            {
                "name": "coordinator_non_demo_personal_workspace",
                "kwargs": {
                    "is_coordinator": True,
                    "is_org_workspace": False,
                },
                "present": (
                    "**Coordinator workspace tools:**",
                    "Boss details\n------------",
                ),
                "absent": (
                    "Authorized humans\n-----------------",
                    "Team Coordinator\n----------------",
                    "list_org_members",
                    "invite_org_member",
                ),
            },
        )

        for case in cases:
            prompt = _build(**case["kwargs"])
            for marker in case["present"]:
                assert (
                    marker in prompt
                ), f"{case['name']} missing expected marker: {marker}"
            for marker in case["absent"]:
                assert (
                    marker not in prompt
                ), f"{case['name']} unexpectedly contains marker: {marker}"


class TestCoordinatorVoicePrompt:
    """Coordinator voice calls rely on shared bio and base voice scaffolding."""

    def test_regular_voice_prompt_unchanged_when_flag_is_false(self):
        omitted = _build_voice()
        explicit_false = _build_voice(is_coordinator=False)

        assert omitted == explicit_false
        assert "Coordinator voice role" not in omitted

    def test_coordinator_voice_prompt_does_not_add_extra_role_block_after_bio(self):
        prompt = _build_voice(is_coordinator=True)

        assert "Bio\n---\nI help Acme configure its Unify team." in prompt
        assert "Coordinator voice role" not in prompt
        assert prompt.index("Bio\n---") < prompt.index("Brevity\n-------")

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

    def test_teams_workspace_actions_are_not_marked_contact_addressed(self):
        prompt = _build(
            assistant_has_phone=True,
            assistant_has_email=True,
            assistant_has_teams=True,
        )
        contact_actions = prompt.split("**Contact actions:**")[1].split(
            "- If the contact is NOT in active_conversations at all",
        )[0]
        contact_addressed_line = next(
            line
            for line in contact_actions.splitlines()
            if "Contact-addressed communication tools" in line
        )

        assert "send_teams_message" in contact_addressed_line
        assert "create_teams_channel" not in contact_addressed_line
        assert "create_teams_meet" not in contact_addressed_line
        assert (
            "`create_teams_channel` and `create_teams_meet` are Teams workspace actions"
            in contact_actions
        )


# ---------------------------------------------------------------------------
# Tests – external app integration
# ---------------------------------------------------------------------------


class TestExternalAppIntegration:
    """The prompt includes guidance for external app integration via credentials + SDK."""

    def test_onboarding_has_app_integration_qa(self):
        prompt = _build()
        assert "Can you help me manage my apps and online services?" in prompt
        assert "secure page on the console" in prompt
        assert "API credentials or access tokens" in prompt
        assert "⋮ → Secrets" in prompt
        assert "service's Python SDK" in prompt

    def test_act_capabilities_has_external_apps_bullet(self):
        prompt = _build()
        assert "**External apps & services**" in prompt
        assert "stored credentials and the service's Python SDK" in prompt

    def test_onboarding_qa_present_in_demo_mode(self):
        prompt = _build(demo_mode=True)
        assert "Can you help me manage my apps and online services?" in prompt

    def test_org_assistant_onboarding_allows_direct_setup_with_shared_handoff(self):
        prompt = _build(workspace_coordinator_name="Avery Coordinator")

        assert "I can walk through app setup and day-to-day usage directly" in prompt
        assert (
            "If a credential needs to be shared across the team or org, I can route "
            "that handoff to Avery Coordinator."
        ) in prompt
        assert "Avery Coordinator owns that setup" not in prompt
        assert "I cannot forward it automatically" not in prompt

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

    def test_proactive_meeting_absent_for_coordinator(self):
        prompt = _build(is_coordinator=True)
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
        assert "Hover over my name in the assistant list → ⋮ → Secrets" in prompt
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
