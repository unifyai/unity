"""
tests/conversation_manager/flows/test_multi_thread.py
=====================================================

Tests for multi-thread email scenarios where the system must distinguish
between multiple email threads with the same contact.

The current implementation auto-infers the email_id for threading based on
subject matching against the most recent inbound email. This fails when:
- Multiple threads exist with the same subject
- The user wants to reply to an older thread, not the most recent one

These tests verify and document this limitation.
"""

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    filter_events_by_type,
)
from tests.conversation_manager.conftest import (
    TEST_CONTACTS,
    BOSS,
    HELPFUL_RESPONSE_POLICY,
)
from unity.conversation_manager.events import (
    EmailReceived,
    EmailSent,
    UnifyMessageReceived,
)

pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def use_helpful_response_policy(initialized_cm):
    """Override response_policy for test contacts to be more permissive."""
    cm = initialized_cm.cm
    if cm.contact_manager is not None:
        for contact in TEST_CONTACTS:
            cm.contact_manager.update_contact(
                contact_id=contact["contact_id"],
                response_policy=HELPFUL_RESPONSE_POLICY,
            )
    yield


# ---------------------------------------------------------------------------
#  Multi-thread email tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_reply_to_older_thread_with_same_subject(initialized_cm):
    """
    Bug reproduction: system picks wrong thread when multiple threads have same subject.

    Scenario:
    1. Alice sends email on Thread A (subject: "Budget Discussion", about Q1)
    2. Alice sends email on Thread B (subject: "Budget Discussion", about Q2) - MORE RECENT
    3. Boss tells assistant to reply to Alice about Q1, using subject "Budget Discussion"
    4. EXPECTED: Reply should thread to Thread A (email_id_replied_to = thread_a_email_id)
    5. ACTUAL BUG: System picks Thread B because it's most recent with matching subject

    The bug trigger conditions:
    - Reply subject EXACTLY matches inbound email subjects (auto-inference kicks in)
    - Multiple inbound emails have the same subject
    - The desired thread is NOT the most recent one

    The current auto-inference logic in brain_action_tools.py:
    - Iterates through email thread in reverse (most recent first)
    - Finds first inbound message where subject matches the reply subject
    - Uses that email_id, IGNORING the LLM's explicit email_id_to_reply_to

    Note: If the LLM uses "Re: Budget Discussion" instead of "Budget Discussion",
    the auto-inference won't find a match and the LLM's choice will be used.
    This test explicitly instructs the LLM to use the exact subject to trigger the bug.
    """
    cm = initialized_cm
    alice = TEST_CONTACTS[0]  # Alice Smith

    # Use realistic RFC Message-ID format (opaque identifiers from email servers)
    # These are NOT email addresses - they're unique message identifiers like:
    # <CABx+abc123@mail.gmail.com> or <1234567890.123456@smtp.example.com>
    # Note: angle brackets are stripped by the system, so we omit them here
    thread_a_email_id = "CAKx7fQ1a2b3c4d5@mail.gmail.com"
    thread_b_email_id = "CAKx7fQ9z8y7w6v5@mail.gmail.com"

    # --- Step 1: Alice sends email on Thread A (Q1 budget, older) ---
    await cm.step(
        EmailReceived(
            contact=alice,
            subject="Budget Discussion",
            body="Hi, I wanted to discuss the Q1 budget allocation. Can we review the marketing spend?",
            email_id=thread_a_email_id,
            attachments=[],
        ),
    )

    # --- Step 2: Alice sends email on Thread B (Q2 budget, more recent) ---
    # Same subject as Thread A - this is the key condition that triggers the bug
    await cm.step(
        EmailReceived(
            contact=alice,
            subject="Budget Discussion",  # Same subject as Thread A!
            body="Following up on a separate matter - the Q2 budget projections need review. Very different from Q1.",
            email_id=thread_b_email_id,
            attachments=[],
        ),
    )

    # --- Step 3: Boss asks assistant to reply specifically to Thread A (Q1) ---
    # CRITICAL: We explicitly tell the assistant to use the SAME subject (not "Re: ...")
    # This forces the auto-inference to kick in and pick the wrong thread.
    # The auto-inference matches by exact subject and picks the most recent email.
    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Reply to Alice's email about the Q1 budget (the first one she sent, not the Q2 one). "
                "Use the subject line 'Budget Discussion' (same as her emails). "
                "Tell her we'll review the Q1 marketing spend next week."
            ),
        ),
    )

    # --- Step 4: Verify the reply went to the CORRECT thread ---
    email_events = filter_events_by_type(result.output_events, EmailSent)
    assert len(email_events) >= 1, (
        f"Expected at least 1 EmailSent event, got {len(email_events)}. "
        f"Output events: {[type(e).__name__ for e in result.output_events]}"
    )

    sent_email = email_events[0]

    # THE ASSERTION THAT SHOULD FAIL (documenting the bug):
    # The email should reply to Thread A (Q1), but the system will pick Thread B (Q2)
    # because Thread B is more recent and has the same subject.
    assert sent_email.email_id_replied_to == thread_a_email_id, (
        f"Expected reply to Thread A (Q1 budget)\n"
        f"  email_id: {thread_a_email_id}\n"
        f"But system replied to Thread B (Q2 budget)\n"
        f"  email_id: {sent_email.email_id_replied_to}\n"
        f"\n"
        f"This documents a bug: the system auto-infers the reply thread based on "
        f"the most recent email with matching subject, OVERRIDING the LLM's explicit "
        f"email_id_to_reply_to choice."
    )
