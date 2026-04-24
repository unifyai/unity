"""
tests/conversation_manager/core/test_prompt_builders.py
=======================================================

Unit tests for the ConversationManager prompt builder, focusing on
capability-gated sections (assistant phone / email).
"""

from __future__ import annotations


from unity.conversation_manager.prompt_builders import build_system_prompt

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
        assert "Resources → Secrets" in prompt
        assert "service's SDK" in prompt

    def test_act_capabilities_has_external_apps_bullet(self):
        prompt = _build()
        assert "**External apps & services**" in prompt
        assert "stored credentials and the service's Python SDK" in prompt

    def test_onboarding_qa_present_in_demo_mode(self):
        prompt = _build(demo_mode=True)
        assert "Can you help me manage my apps and online services?" in prompt

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
        assert "Resources" in prompt
        assert "Secrets" in prompt
        assert "Contact Details" in prompt

    def test_console_knowledge_has_navigation_paths(self):
        prompt = _build()
        assert "Resources → Secrets" in prompt
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


# ---------------------------------------------------------------------------
# Tests – None-safe boss_contact_id
# ---------------------------------------------------------------------------


class TestNoneSafeBossContactId:
    """``boss_contact_id=None`` must omit the Contact ID line from the
    boss-details block and suppress the ``(contact_id=...)`` interpolation
    inside demo-mode and tool-instruction examples — otherwise the LLM
    reads ``Contact ID: None`` and invokes tools with the literal ``None``.
    """

    def test_none_contact_id_omits_contact_id_line(self) -> None:
        prompt = _build(contact_id=None)
        # Neither literal nor magic-number leakage.
        assert "Contact ID: None" not in prompt
        assert "Contact ID: 1" not in prompt
        assert "Contact ID: 0" not in prompt
        # First-name / surname still flow through — only the id is
        # elided.
        assert "First Name: Alice" in prompt
        assert "Surname: Smith" in prompt

    def test_none_contact_id_in_demo_mode_never_interpolates_none(
        self,
    ) -> None:
        """Demo mode must not interpolate the literal ``None`` into the
        ``make_call(...)`` instruction, the boss-id phrase, or the ack
        example. Any remaining ``make_call(contact_id=...)`` must use
        a human-readable placeholder like ``<boss_id>`` rather than
        the Python ``None`` string.
        """

        prompt = _build(
            contact_id=None,
            demo_mode=True,
            assistant_has_phone=True,
            assistant_has_email=True,
        )
        assert "contact_id=None" not in prompt
        # "(contact_id=...)" in the demo-mode intro phrase must also
        # not render the literal None.
        assert " (contact_id=None)" not in prompt

    def test_concrete_contact_id_still_renders(self) -> None:
        """Happy-path sanity check: a resolved id still appears in the
        boss block."""

        prompt = _build(contact_id=42)
        assert "Contact ID: 42" in prompt

    def test_concrete_contact_id_renders_in_demo_mode(self) -> None:
        prompt = _build(
            contact_id=42,
            demo_mode=True,
            assistant_has_phone=True,
        )
        assert "contact_id=42" in prompt
