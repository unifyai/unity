"""
tests/conversation_manager/core/test_renderer.py
================================================

Unit tests for the Renderer class in `domains/renderer.py`.

These are symbolic tests that verify rendering logic without invoking the LLM.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from unity.conversation_manager.domains.contact_index import EmailMessage
from unity.conversation_manager.domains.renderer import (
    Renderer,
    _get_assistant_email_role,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def renderer():
    """Create a Renderer instance."""
    return Renderer()


@pytest.fixture
def sample_received_email():
    """Create a sample received email where assistant is in To."""
    return EmailMessage(
        name="Alice Smith",
        subject="Project Update",
        body="Here's the latest update on the project.",
        email_id="CAKx7fQ_test@mail.gmail.com",
        timestamp=datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc),
        role="user",
        attachments=[],
        to=["assistant@unify.ai"],
        cc=["bob@example.com"],
        bcc=[],
        contact_role="sender",
    )


@pytest.fixture
def sample_sent_email():
    """Create a sample sent email from the assistant."""
    return EmailMessage(
        name="You",
        subject="Re: Project Update",
        body="Thanks for the update!",
        email_id="CAKx7fQ_test@mail.gmail.com",
        timestamp=datetime(2025, 6, 13, 12, 5, 0, tzinfo=timezone.utc),
        role="assistant",
        attachments=[],
        to=["alice@example.com"],
        cc=["bob@example.com"],
        bcc=[],
        contact_role="to",
    )


# =============================================================================
# Tests for _get_assistant_email_role
# =============================================================================


class TestGetAssistantEmailRole:
    """Tests for the _get_assistant_email_role helper function."""

    def test_assistant_is_direct_recipient_to(self):
        """When assistant's email is in To field, returns 'direct recipient'."""
        email = EmailMessage(
            name="Alice Smith",
            subject="Test",
            body="Test body",
            email_id="test@mail.gmail.com",
            timestamp=datetime.now(timezone.utc),
            role="user",
            to=["assistant@unify.ai", "other@example.com"],
            cc=[],
            bcc=[],
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = "assistant@unify.ai"
            result = _get_assistant_email_role(email)
            assert result == "You were a direct recipient (To)"

    def test_assistant_is_cc_recipient(self):
        """When assistant's email is in Cc field, returns 'CC'd'."""
        email = EmailMessage(
            name="Alice Smith",
            subject="Test",
            body="Test body",
            email_id="test@mail.gmail.com",
            timestamp=datetime.now(timezone.utc),
            role="user",
            to=["bob@example.com"],
            cc=["assistant@unify.ai"],
            bcc=[],
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = "assistant@unify.ai"
            result = _get_assistant_email_role(email)
            assert result == "You were CC'd"

    def test_assistant_is_bcc_recipient(self):
        """When assistant's email is in Bcc field, returns 'BCC'd'."""
        email = EmailMessage(
            name="Alice Smith",
            subject="Test",
            body="Test body",
            email_id="test@mail.gmail.com",
            timestamp=datetime.now(timezone.utc),
            role="user",
            to=["bob@example.com"],
            cc=[],
            bcc=["assistant@unify.ai"],
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = "assistant@unify.ai"
            result = _get_assistant_email_role(email)
            assert result == "You were BCC'd"

    def test_assistant_sent_email(self):
        """When assistant sent the email (role=assistant), returns 'sent'."""
        email = EmailMessage(
            name="You",
            subject="Test",
            body="Test body",
            email_id="test@mail.gmail.com",
            timestamp=datetime.now(timezone.utc),
            role="assistant",
            to=["alice@example.com"],
            cc=[],
            bcc=[],
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = "assistant@unify.ai"
            result = _get_assistant_email_role(email)
            assert result == "You sent this email"

    def test_assistant_not_in_email(self):
        """When assistant's email is not in any field, returns None."""
        email = EmailMessage(
            name="Alice Smith",
            subject="Test",
            body="Test body",
            email_id="test@mail.gmail.com",
            timestamp=datetime.now(timezone.utc),
            role="user",
            to=["bob@example.com"],
            cc=["charlie@example.com"],
            bcc=[],
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = "assistant@unify.ai"
            result = _get_assistant_email_role(email)
            assert result is None

    def test_case_insensitive_email_matching(self):
        """Email matching should be case-insensitive."""
        email = EmailMessage(
            name="Alice Smith",
            subject="Test",
            body="Test body",
            email_id="test@mail.gmail.com",
            timestamp=datetime.now(timezone.utc),
            role="user",
            to=["ASSISTANT@UNIFY.AI"],  # Uppercase
            cc=[],
            bcc=[],
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = "assistant@unify.ai"  # Lowercase
            result = _get_assistant_email_role(email)
            assert result == "You were a direct recipient (To)"

    def test_no_assistant_email_configured(self):
        """When assistant email is not configured, returns None."""
        email = EmailMessage(
            name="Alice Smith",
            subject="Test",
            body="Test body",
            email_id="test@mail.gmail.com",
            timestamp=datetime.now(timezone.utc),
            role="user",
            to=["assistant@unify.ai"],
            cc=[],
            bcc=[],
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = None
            result = _get_assistant_email_role(email)
            assert result is None


# =============================================================================
# Tests for Renderer.render_message with email assistant role
# =============================================================================


class TestRendererEmailAssistantRole:
    """Tests for email rendering with assistant role context."""

    def test_render_email_shows_assistant_role_when_direct_recipient(self, renderer):
        """Rendered email includes '[Your role: ...]' when assistant is To recipient."""
        email = EmailMessage(
            name="Alice Smith",
            subject="Important Update",
            body="Please review this.",
            email_id="test123@mail.gmail.com",
            timestamp=datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc),
            role="user",
            to=["assistant@unify.ai"],
            cc=[],
            bcc=[],
            contact_role="sender",
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = "assistant@unify.ai"
            # Use a timestamp before the message to mark it as NEW
            last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)
            result = renderer.render_message(email, last_snapshot)

            # Should contain assistant role line
            assert "[Your role: You were a direct recipient (To)]" in result
            # Should also contain contact role line
            assert "[Context: This contact SENT this email]" in result

    def test_render_email_shows_assistant_role_when_sender(self, renderer):
        """Rendered email includes '[Your role: You sent this email]' for outgoing."""
        email = EmailMessage(
            name="You",
            subject="Re: Important Update",
            body="Got it, thanks!",
            email_id="test123@mail.gmail.com",
            timestamp=datetime(2025, 6, 13, 12, 5, 0, tzinfo=timezone.utc),
            role="assistant",
            to=["alice@example.com"],
            cc=[],
            bcc=[],
            contact_role="to",
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = "assistant@unify.ai"
            last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)
            result = renderer.render_message(email, last_snapshot)

            # Should contain assistant role line for sent email
            assert "[Your role: You sent this email]" in result

    def test_render_email_no_assistant_role_when_not_involved(self, renderer):
        """Rendered email does not include assistant role when not in email."""
        email = EmailMessage(
            name="Alice Smith",
            subject="FYI",
            body="Forwarding this for reference.",
            email_id="test456@mail.gmail.com",
            timestamp=datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc),
            role="user",
            to=["bob@example.com"],  # Not assistant
            cc=["charlie@example.com"],  # Not assistant
            bcc=[],
            contact_role="sender",
        )
        with patch(
            "unity.conversation_manager.domains.renderer.SESSION_DETAILS",
        ) as mock_session:
            mock_session.assistant.email = "assistant@unify.ai"
            last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)
            result = renderer.render_message(email, last_snapshot)

            # Should NOT contain assistant role line
            assert "[Your role:" not in result
            # But should still contain contact role line
            assert "[Context: This contact SENT this email]" in result
