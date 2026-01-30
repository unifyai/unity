"""
tests/conversation_manager/flows/test_multi_thread.py
=====================================================

Tests for multi-thread email scenarios where the system must distinguish
between multiple email threads with the same contact.

The current implementation auto-infers the email_id for threading based on
subject matching against the most recent inbound email. This fails when:
- Multiple threads exist with the same subject
- The user wants to reply to an older thread, not the most recent one

Also includes tests for:
- Reply-all functionality with multiple recipients
- Email threading with to/cc/bcc fields

These tests verify and document these features.
"""

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    filter_events_by_type,
    assert_has_one,
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


# ---------------------------------------------------------------------------
#  Reply-all tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_reply_all_preserves_recipients(initialized_cm):
    """
    Reply-all should preserve all original recipients.

    Scenario:
    1. Alice sends email to assistant with Bob and Charlie CC'd
    2. Boss asks assistant to reply-all
    3. EXPECTED: Reply should have Alice in TO, Bob and Charlie in CC
    """
    cm = initialized_cm
    alice = TEST_CONTACTS[0]  # Alice Smith
    bob = TEST_CONTACTS[1]  # Bob Johnson
    charlie = TEST_CONTACTS[2]  # Charlie Davis

    email_id = "CAKx7fQ_reply_all_test@mail.gmail.com"

    # Alice sends email with Bob and Charlie CC'd
    await cm.step(
        EmailReceived(
            contact=alice,
            subject="Team Sync",
            body="Hi team, let's sync up on the project status.",
            email_id=email_id,
            attachments=[],
            to=[],  # Assistant is the recipient (implicit)
            cc=[bob["email_address"], charlie["email_address"]],
        ),
    )

    # Boss asks to reply-all
    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Reply all to Alice's team sync email. "
                "Tell them I'll send the status update by end of day."
            ),
        ),
    )

    # Should have exactly one email sent
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]

    # Verify Alice (original sender) is in TO
    assert (
        alice["email_address"] in email.to
    ), f"Expected Alice (original sender) in 'to', got to={email.to}"

    # Verify Bob and Charlie (original CC) are in CC
    # Note: They might be in 'to' or 'cc' depending on LLM interpretation
    all_recipients = email.to + email.cc
    assert (
        bob["email_address"] in all_recipients
    ), f"Expected Bob in recipients, got to={email.to}, cc={email.cc}"
    assert (
        charlie["email_address"] in all_recipients
    ), f"Expected Charlie in recipients, got to={email.to}, cc={email.cc}"


@pytest.mark.asyncio
@_handle_project
async def test_reply_all_to_email_with_multiple_to_recipients(initialized_cm):
    """
    Reply-all to an email with multiple TO recipients.

    Scenario:
    1. Alice sends email to assistant AND Diana (both in TO)
    2. Boss asks assistant to reply-all
    3. EXPECTED: Reply includes Alice (sender) and Diana in recipients
    """
    cm = initialized_cm
    alice = TEST_CONTACTS[0]  # Alice Smith
    diana = TEST_CONTACTS[3]  # Diana Evans

    email_id = "CAKx7fQ_multi_to_test@mail.gmail.com"

    # Alice sends email with Diana also in TO
    await cm.step(
        EmailReceived(
            contact=alice,
            subject="Dual Recipients Test",
            body="Hi, this email is sent to both the assistant and Diana.",
            email_id=email_id,
            attachments=[],
            to=[diana["email_address"]],  # Diana explicitly in TO
            cc=[],
        ),
    )

    # Boss asks to reply-all
    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content="Reply all to Alice's email about dual recipients. Confirm receipt.",
        ),
    )

    # Should have exactly one email sent
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]

    # Verify Alice (sender) and Diana are both in recipients
    all_recipients = email.to + email.cc
    assert (
        alice["email_address"] in all_recipients
    ), f"Expected Alice (sender) in recipients, got to={email.to}, cc={email.cc}"
    assert (
        diana["email_address"] in all_recipients
    ), f"Expected Diana in recipients, got to={email.to}, cc={email.cc}"


@pytest.mark.asyncio
@_handle_project
async def test_email_thread_fork_with_different_recipients(initialized_cm):
    """
    Fork an email thread by replying with different recipients.

    Scenario:
    1. Alice sends email about project (just to assistant)
    2. Boss asks to reply and add Bob and Charlie (using inline emails)
    3. EXPECTED: Reply includes Alice, Bob, and Charlie
    """
    cm = initialized_cm
    alice = TEST_CONTACTS[0]  # Alice Smith

    email_id = "CAKx7fQ_fork_test@mail.gmail.com"

    # Alice sends initial email (just to assistant)
    await cm.step(
        EmailReceived(
            contact=alice,
            subject="Project Proposal",
            body="Here's my proposal for the new feature. What do you think?",
            email_id=email_id,
            attachments=[],
        ),
    )

    # Boss asks to reply but add more recipients (using inline emails)
    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Reply to Alice's project proposal email and tell her the proposal "
                "looks great and we're moving forward with it. "
                "Also CC bob@example.com and charlie@example.com "
                "so they can provide their technical input."
            ),
        ),
    )

    # Should have exactly one email sent
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]

    # Verify Alice (original sender) is in recipients
    all_recipients = email.to + email.cc
    assert (
        alice["email_address"] in all_recipients
    ), f"Expected Alice in recipients, got to={email.to}, cc={email.cc}"
    # Verify Bob and Charlie were added
    assert (
        "bob@example.com" in all_recipients
    ), f"Expected bob@example.com in recipients, got to={email.to}, cc={email.cc}"
    assert (
        "charlie@example.com" in all_recipients
    ), f"Expected charlie@example.com in recipients, got to={email.to}, cc={email.cc}"


@pytest.mark.asyncio
@_handle_project
async def test_reply_removes_recipient_from_thread(initialized_cm):
    """
    Reply to thread but explicitly exclude a recipient.

    Scenario:
    1. Alice sends email with Bob CC'd
    2. Boss asks to reply to Alice only, WITHOUT Bob
    3. EXPECTED: Reply goes only to Alice, Bob is not included
    """
    cm = initialized_cm
    alice = TEST_CONTACTS[0]  # Alice Smith
    bob = TEST_CONTACTS[1]  # Bob Johnson

    email_id = "CAKx7fQ_exclude_test@mail.gmail.com"

    # Alice sends email with Bob CC'd
    await cm.step(
        EmailReceived(
            contact=alice,
            subject="Sensitive Topic",
            body="Can we discuss the reorganization privately?",
            email_id=email_id,
            attachments=[],
            cc=[bob["email_address"]],
        ),
    )

    # Boss asks to reply only to Alice, excluding Bob
    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Reply to Alice's email about the sensitive topic. "
                "Reply ONLY to Alice - do NOT include Bob. This is confidential."
            ),
        ),
    )

    # Should have exactly one email sent
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]

    # Verify Alice is in recipients
    all_recipients = email.to + email.cc + email.bcc
    assert (
        alice["email_address"] in all_recipients
    ), f"Expected Alice in recipients, got to={email.to}, cc={email.cc}, bcc={email.bcc}"

    # Verify Bob is NOT in recipients
    assert bob["email_address"] not in all_recipients, (
        f"Expected Bob NOT in recipients, but found him in "
        f"to={email.to}, cc={email.cc}, bcc={email.bcc}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_email_thread_with_external_recipients(initialized_cm):
    """
    Reply to email thread and add external email addresses (not contacts).

    Scenario:
    1. Alice sends email
    2. Boss asks to reply via email and add external partners
    3. EXPECTED: Reply includes Alice and the external email addresses
    """
    cm = initialized_cm
    alice = TEST_CONTACTS[0]  # Alice Smith

    email_id = "CAKx7fQ_external_test@mail.gmail.com"

    # Alice sends initial email
    await cm.step(
        EmailReceived(
            contact=alice,
            subject="Partnership Opportunity",
            body="I've been in touch with Acme Corp about a partnership.",
            email_id=email_id,
            attachments=[],
        ),
    )

    # Boss asks to reply via email and add external partners
    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Send an email reply to Alice about her partnership opportunity email. "
                "CC partner@acme-corp.com and legal@acme-corp.com "
                "so they can join the discussion. Tell her we're interested."
            ),
        ),
    )

    # Should have exactly one email sent
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]

    # Verify Alice is in recipients
    all_recipients = email.to + email.cc
    assert (
        alice["email_address"] in all_recipients
    ), f"Expected Alice in recipients, got to={email.to}, cc={email.cc}"

    # Verify at least one external email is included
    external_emails = ["partner@acme-corp.com", "legal@acme-corp.com"]
    found_external = any(ext in all_recipients for ext in external_emails)
    assert found_external, (
        f"Expected at least one external email in recipients, "
        f"got to={email.to}, cc={email.cc}"
    )


# ---------------------------------------------------------------------------
#  Multi-thread with different recipient sets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_multiple_threads_different_recipients(initialized_cm):
    """
    Multiple email threads with the same contact but different CC lists.

    Scenario:
    1. Alice sends email Thread A with Bob CC'd (technical discussion)
    2. Alice sends email Thread B with Charlie CC'd (budget discussion)
    3. Boss asks to reply to the technical thread (with Bob)
    4. EXPECTED: Reply should be threaded to Thread A and include Bob
    """
    cm = initialized_cm
    alice = TEST_CONTACTS[0]  # Alice Smith
    bob = TEST_CONTACTS[1]  # Bob Johnson
    charlie = TEST_CONTACTS[2]  # Charlie Davis

    thread_a_email_id = "CAKx7fQ_tech_thread@mail.gmail.com"
    thread_b_email_id = "CAKx7fQ_budget_thread@mail.gmail.com"

    # Thread A: Technical discussion with Bob
    await cm.step(
        EmailReceived(
            contact=alice,
            subject="Technical Architecture Review",
            body="Let's review the microservices architecture with the tech team.",
            email_id=thread_a_email_id,
            attachments=[],
            cc=[bob["email_address"]],
        ),
    )

    # Thread B: Budget discussion with Charlie (more recent)
    await cm.step(
        EmailReceived(
            contact=alice,
            subject="Budget Planning",
            body="We need to finalize the Q2 budget allocations.",
            email_id=thread_b_email_id,
            attachments=[],
            cc=[charlie["email_address"]],
        ),
    )

    # Boss asks to reply to the TECHNICAL thread (not the most recent)
    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Reply to Alice's email about the technical architecture review. "
                "Keep Bob in the loop. Tell them we'll schedule a review meeting."
            ),
        ),
    )

    # Should have exactly one email sent
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]

    # Verify Alice is in recipients
    all_recipients = email.to + email.cc
    assert (
        alice["email_address"] in all_recipients
    ), f"Expected Alice in recipients, got to={email.to}, cc={email.cc}"

    # Verify Bob (from tech thread) is in recipients
    assert bob["email_address"] in all_recipients, (
        f"Expected Bob (from tech thread) in recipients, "
        f"got to={email.to}, cc={email.cc}"
    )

    # Verify reply is threaded to the technical thread (Thread A)
    assert email.email_id_replied_to == thread_a_email_id, (
        f"Expected reply to technical thread (Thread A), "
        f"but got email_id_replied_to={email.email_id_replied_to}"
    )
