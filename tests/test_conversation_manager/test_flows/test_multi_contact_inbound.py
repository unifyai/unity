"""
tests/test_conversation_manager/test_multi_contact_inbound.py
=====================================================

Tests for multi-contact inbound conversation routing.

These tests verify that the ConversationManager can handle conversations
with multiple contacts. We first establish contact (receive a message
from them) so they appear in active_conversations, then verify the
assistant can route replies correctly.

This implicitly tests multi-turn behavior since managing multiple
contacts requires maintaining context about who said what.
"""

import pytest

from tests.helpers import _handle_project
from tests.test_conversation_manager.cm_helpers import filter_events_by_type
from tests.test_conversation_manager.conftest import TEST_CONTACTS
from unity.conversation_manager.events import (
    SMSReceived,
    SMSSent,
    EmailReceived,
    EmailSent,
)

pytestmark = pytest.mark.eval

# Convenience references to test contacts
# Note: contact_id 0 (assistant) and 1 (user) are system contacts.
# Test contacts start from contact_id 2.
ALICE = TEST_CONTACTS[0]  # contact_id 2
BOB = TEST_CONTACTS[1]  # contact_id 3
BOSS = TEST_CONTACTS[2]  # contact_id 4 - used as command sender in multi-contact tests


# ---------------------------------------------------------------------------
#  Multiple active conversations - reply to correct contact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_reply_to_alice_after_her_message(initialized_cm):
    """
    Alice sends SMS, boss asks to reply -> reply should go to Alice.

    This establishes Alice in active_conversations first.
    """
    cm = initialized_cm

    # Step 1: Alice sends a message (establishes her in active_conversations)
    await cm.step_until_wait(
        SMSReceived(
            contact=ALICE,
            content="Hi, can we meet tomorrow?",
        ),
    )

    # Step 2: Boss asks to reply to Alice
    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Reply to Alice saying yes, 10am works",
        ),
    )

    sms_events = filter_events_by_type(result.output_events, SMSSent)
    alice_sms = [
        s for s in sms_events if s.contact["contact_id"] == ALICE["contact_id"]
    ]

    assert len(alice_sms) >= 1, (
        f"Expected reply to Alice (contact_id={ALICE['contact_id']}), "
        f"got SMS to: {[s.contact['contact_id'] for s in sms_events]}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_reply_to_bob_after_his_message(initialized_cm):
    """
    Bob sends SMS, boss asks to reply -> reply should go to Bob.
    """
    cm = initialized_cm

    # Step 1: Bob sends a message
    await cm.step_until_wait(
        SMSReceived(
            contact=BOB,
            content="Project status update needed",
        ),
    )

    # Step 2: Boss asks to reply to Bob
    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Reply to Bob: project is on track",
        ),
    )

    sms_events = filter_events_by_type(result.output_events, SMSSent)
    bob_sms = [s for s in sms_events if s.contact["contact_id"] == BOB["contact_id"]]

    assert len(bob_sms) >= 1, (
        f"Expected reply to Bob (contact_id={BOB['contact_id']}), "
        f"got SMS to: {[s.contact['contact_id'] for s in sms_events]}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_two_contacts_reply_to_correct_one(initialized_cm):
    """
    Both Alice and Bob message, boss specifies who to reply to.

    Tests that the assistant can distinguish between multiple active contacts.
    """
    cm = initialized_cm

    # Step 1: Alice sends a message
    await cm.step_until_wait(
        SMSReceived(
            contact=ALICE,
            content="Meeting reminder for tomorrow",
        ),
    )

    # Step 2: Bob sends a message
    await cm.step_until_wait(
        SMSReceived(
            contact=BOB,
            content="Invoice ready for review",
        ),
    )

    # Step 3: Boss asks to reply specifically to Bob (not Alice)
    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Reply to Bob about the invoice: approved",
        ),
    )

    sms_events = filter_events_by_type(result.output_events, SMSSent)
    bob_sms = [s for s in sms_events if s.contact["contact_id"] == BOB["contact_id"]]

    assert len(bob_sms) >= 1, (
        f"Expected reply to Bob (contact_id={BOB['contact_id']}), "
        f"got SMS to: {[s.contact['contact_id'] for s in sms_events]}"
    )


# ---------------------------------------------------------------------------
#  Email conversations with multiple contacts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_email_reply_to_alice(initialized_cm):
    """
    Alice emails, boss asks to reply -> reply should go to Alice.
    """
    cm = initialized_cm

    # Step 1: Alice sends an email
    await cm.step_until_wait(
        EmailReceived(
            contact=ALICE,
            subject="Quarterly Report",
            body="Please review the attached report",
            email_id="alice_email_1",
        ),
    )

    # Step 2: Boss asks to reply to Alice
    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Email Alice back saying report looks good",
        ),
    )

    email_events = filter_events_by_type(result.output_events, EmailSent)
    alice_emails = [
        e for e in email_events if e.contact["contact_id"] == ALICE["contact_id"]
    ]

    assert len(alice_emails) >= 1, (
        f"Expected email reply to Alice (contact_id={ALICE['contact_id']}), "
        f"got emails to: {[e.contact['contact_id'] for e in email_events]}"
    )
