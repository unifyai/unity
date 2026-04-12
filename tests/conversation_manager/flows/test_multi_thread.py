"""
tests/conversation_manager/flows/test_multi_thread.py
=====================================================

Tests for multi-thread email scenarios where the system must distinguish
between multiple email threads with the same contact.

Also includes tests for:
- Reply-all functionality with multiple recipients
- Email threading with to/cc/bcc fields
- Re: prefix and subject-based auto-inference

All tests use a boss-first pattern: the boss pre-instructs the assistant,
then the triggering email(s) arrive.  This avoids non-determinism from
the LLM auto-replying to inbound emails before the boss has spoken.
"""

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    filter_events_by_type,
    assert_has_one,
    make_contacts_visible,
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
#  Multi-thread email tests (require multiple prior emails)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_reply_to_older_thread_with_same_subject(initialized_cm):
    """
    Bug reproduction: system picks wrong thread when multiple threads have same subject.

    Scenario:
    1. Boss pre-instructs: "When Alice emails about Q1 budget, reply about
       reviewing marketing spend.  Use subject 'Budget Discussion'."
    2. Alice sends Thread A (subject: "Budget Discussion", about Q1)
    3. Alice sends Thread B (subject: "Budget Discussion", about Q2) - MORE RECENT
    4. EXPECTED: Reply should thread to Thread A (email_id_replied_to = thread_a_email_id)
    """
    cm = initialized_cm
    alice = TEST_CONTACTS[0]  # Alice Smith

    thread_a_email_id = "CAKx7fQ1a2b3c4d5@mail.gmail.com"
    thread_b_email_id = "CAKx7fQ9z8y7w6v5@mail.gmail.com"

    # --- Boss pre-instructs ---
    await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Alice is going to send two emails both with subject 'Budget Discussion'. "
                "The first will be about Q1 budget, the second about Q2. "
                "Reply to the Q1 one specifically — tell her we'll review the "
                "Q1 marketing spend next week. Use subject 'Budget Discussion'."
            ),
        ),
    )

    # --- Alice sends Thread A (Q1 budget, older) ---
    result_a = await cm.step_until_wait(
        EmailReceived(
            contact=alice,
            subject="Budget Discussion",
            body="Hi, I wanted to discuss the Q1 budget allocation. Can we review the marketing spend?",
            email_id=thread_a_email_id,
            attachments=[],
        ),
    )

    # --- Alice sends Thread B (Q2 budget, more recent, same subject) ---
    result_b = await cm.step_until_wait(
        EmailReceived(
            contact=alice,
            subject="Budget Discussion",
            body="Following up on a separate matter - the Q2 budget projections need review. Very different from Q1.",
            email_id=thread_b_email_id,
            attachments=[],
        ),
    )

    # --- Verify the reply went to the CORRECT thread ---
    all_emails = filter_events_by_type(
        result_a.output_events + result_b.output_events,
        EmailSent,
    )
    assert len(all_emails) >= 1, (
        f"Expected at least 1 EmailSent event, got {len(all_emails)}. "
        f"Output events: "
        f"{[type(e).__name__ for e in result_a.output_events + result_b.output_events]}"
    )

    sent_email = all_emails[0]

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
    1. Boss pre-instructs: reply-all to Alice's upcoming team sync email
    2. Alice sends email with Bob and Charlie CC'd
    3. EXPECTED: Reply should have Alice in TO, Bob and Charlie in CC
    """
    cm = initialized_cm
    alice = TEST_CONTACTS[0]  # Alice Smith
    bob = TEST_CONTACTS[1]  # Bob Johnson
    charlie = TEST_CONTACTS[2]  # Charlie Davis

    email_id = "CAKx7fQ_reply_all_test@mail.gmail.com"

    # Boss pre-instructs
    await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Alice is going to email about a team sync with Bob and Charlie CC'd. "
                "When she does, reply all — tell everyone I'll send the status "
                "update by end of day."
            ),
        ),
    )

    # Alice sends email with Bob and Charlie CC'd
    result = await cm.step_until_wait(
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
    1. Boss pre-instructs: reply-all to Alice's upcoming email (Diana also in TO)
    2. Alice sends email to assistant AND Diana (both in TO)
    3. EXPECTED: Reply includes Alice (sender) and Diana in recipients
    """
    cm = initialized_cm
    alice = TEST_CONTACTS[0]  # Alice Smith
    diana = TEST_CONTACTS[3]  # Diana Evans

    email_id = "CAKx7fQ_multi_to_test@mail.gmail.com"

    # Boss pre-instructs
    await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Alice is going to send an email with Diana also in the TO field. "
                "When it arrives, reply all to confirm receipt."
            ),
        ),
    )

    # Alice sends email with Diana also in TO
    result = await cm.step_until_wait(
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
    1. Boss pre-instructs: when Alice emails about project proposal, reply and
       CC Bob and Charlie
    2. Alice sends email about project (just to assistant)
    3. EXPECTED: Reply includes Alice, Bob, and Charlie
    """
    cm = initialized_cm
    alice = TEST_CONTACTS[0]  # Alice Smith

    make_contacts_visible(cm, 2, 3, 4)  # Alice, Bob, Charlie

    email_id = "CAKx7fQ_fork_test@mail.gmail.com"

    # Boss pre-instructs
    await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "When Alice emails about her project proposal, reply telling her "
                "the proposal looks great and we're moving forward with it. "
                "Also CC Bob and Charlie so they can provide their technical input."
            ),
        ),
    )

    # Alice sends initial email (just to assistant)
    result = await cm.step_until_wait(
        EmailReceived(
            contact=alice,
            subject="Project Proposal",
            body="Here's my proposal for the new feature. What do you think?",
            email_id=email_id,
            attachments=[],
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
    1. Boss pre-instructs: when Alice emails about a sensitive topic (Bob CC'd),
       reply ONLY to Alice — do NOT include Bob
    2. Alice sends email with Bob CC'd
    3. EXPECTED: Reply goes only to Alice, Bob is not included
    """
    cm = initialized_cm
    alice = TEST_CONTACTS[0]  # Alice Smith
    bob = TEST_CONTACTS[1]  # Bob Johnson

    email_id = "CAKx7fQ_exclude_test@mail.gmail.com"

    # Boss pre-instructs
    await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Alice is going to email about a sensitive topic. Bob will be CC'd. "
                "When it arrives, reply ONLY to Alice — do NOT include Bob. "
                "This is confidential."
            ),
        ),
    )

    # Alice sends email with Bob CC'd
    result = await cm.step_until_wait(
        EmailReceived(
            contact=alice,
            subject="Sensitive Topic",
            body="Can we discuss the reorganization privately?",
            email_id=email_id,
            attachments=[],
            cc=[bob["email_address"]],
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
    Reply to email thread and add additional recipients.

    Scenario:
    1. Boss pre-instructs: when Alice emails about the partnership, reply and
       CC Bob and Charlie
    2. Alice sends email
    3. EXPECTED: Reply includes Alice, Bob, and Charlie
    """
    cm = initialized_cm
    alice = TEST_CONTACTS[0]  # Alice Smith
    bob = TEST_CONTACTS[1]  # Bob Johnson
    charlie = TEST_CONTACTS[2]  # Charlie Davis

    make_contacts_visible(cm, 2, 3, 4)  # Alice, Bob, Charlie

    email_id = "CAKx7fQ_external_test@mail.gmail.com"

    # Boss pre-instructs
    await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "When Alice emails about the partnership opportunity, reply to her "
                "and CC Bob and Charlie so they can join the discussion. "
                "Tell her we're interested."
            ),
        ),
    )

    # Alice sends initial email
    result = await cm.step_until_wait(
        EmailReceived(
            contact=alice,
            subject="Partnership Opportunity",
            body="I've been in touch with Acme Corp about a partnership.",
            email_id=email_id,
            attachments=[],
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

    # Verify Bob and Charlie are included
    assert (
        bob["email_address"] in all_recipients
    ), f"Expected Bob in recipients, got to={email.to}, cc={email.cc}"
    assert (
        charlie["email_address"] in all_recipients
    ), f"Expected Charlie in recipients, got to={email.to}, cc={email.cc}"


# ---------------------------------------------------------------------------
#  Multi-thread with different recipient sets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_multiple_threads_different_recipients(initialized_cm):
    """
    Multiple email threads with the same contact but different CC lists.

    Scenario:
    1. Boss pre-instructs: when Alice emails about technical architecture
       (with Bob CC'd) and budget planning (with Charlie CC'd), reply to the
       technical one and keep Bob in the loop
    2. Alice sends Thread A with Bob CC'd (technical discussion)
    3. Alice sends Thread B with Charlie CC'd (budget discussion)
    4. EXPECTED: Reply should be threaded to Thread A and include Bob
    """
    cm = initialized_cm
    alice = TEST_CONTACTS[0]  # Alice Smith
    bob = TEST_CONTACTS[1]  # Bob Johnson
    charlie = TEST_CONTACTS[2]  # Charlie Davis

    thread_a_email_id = "CAKx7fQ_tech_thread@mail.gmail.com"
    thread_b_email_id = "CAKx7fQ_budget_thread@mail.gmail.com"

    # Boss pre-instructs
    await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Alice is going to send two emails: one about technical architecture "
                "with Bob CC'd, and one about budget planning with Charlie CC'd. "
                "Reply to the technical one — keep Bob in the loop and tell them "
                "we'll schedule a review meeting."
            ),
        ),
    )

    # Thread A: Technical discussion with Bob
    result_a = await cm.step_until_wait(
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
    result_b = await cm.step_until_wait(
        EmailReceived(
            contact=alice,
            subject="Budget Planning",
            body="We need to finalize the Q2 budget allocations.",
            email_id=thread_b_email_id,
            attachments=[],
            cc=[charlie["email_address"]],
        ),
    )

    # Check across both email steps
    all_emails = filter_events_by_type(
        result_a.output_events + result_b.output_events,
        EmailSent,
    )
    assert len(all_emails) >= 1, (
        f"Expected at least 1 EmailSent across both steps, got {len(all_emails)}. "
        f"Output events: "
        f"{[type(e).__name__ for e in result_a.output_events + result_b.output_events]}"
    )

    email = all_emails[0]

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


@pytest.mark.asyncio
@_handle_project
async def test_reply_adds_re_prefix_to_subject(initialized_cm):
    """
    Reply to email should auto-prefix subject with "Re:" if not present.

    Scenario:
    1. Boss pre-instructs: when Alice emails about a meeting request, reply
    2. Alice sends email with subject "Meeting Request"
    3. EXPECTED: Reply subject should be "Re: Meeting Request"
    """
    cm = initialized_cm
    alice = TEST_CONTACTS[0]

    email_id = "CAKx7fQ_re_prefix_test@mail.gmail.com"

    # Boss pre-instructs
    await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "When Alice emails about a meeting request, reply telling her "
                "Tuesday at 2pm works."
            ),
        ),
    )

    # Alice sends email
    result = await cm.step_until_wait(
        EmailReceived(
            contact=alice,
            subject="Meeting Request",
            body="Can we schedule a meeting for next week?",
            email_id=email_id,
            attachments=[],
        ),
    )

    # Should have exactly one email sent
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]

    # Verify subject has "Re:" prefix
    assert email.subject.startswith(
        "Re:",
    ), f"Expected subject to start with 'Re:', got '{email.subject}'"
    # Verify it's threaded correctly
    assert (
        email.email_id_replied_to == email_id
    ), f"Expected email_id_replied_to={email_id}, got {email.email_id_replied_to}"


@pytest.mark.asyncio
@_handle_project
async def test_subject_auto_inference_threads_correctly(initialized_cm):
    """
    When no explicit email_id is provided, system should auto-infer
    based on subject matching and thread to the correct email.

    Scenario:
    1. Boss pre-instructs: when Alice emails about Project Alpha, reply
    2. Alice sends email about "Project Alpha"
    3. EXPECTED: Reply should be threaded to Alice's email
    """
    cm = initialized_cm
    alice = TEST_CONTACTS[0]

    email_id = "CAKx7fQ_auto_infer_test@mail.gmail.com"

    # Boss pre-instructs
    await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "When Alice emails about Project Alpha, reply telling her "
                "the updates look good and we should proceed."
            ),
        ),
    )

    # Alice sends email
    result = await cm.step_until_wait(
        EmailReceived(
            contact=alice,
            subject="Project Alpha",
            body="Here's the latest update on Project Alpha.",
            email_id=email_id,
            attachments=[],
        ),
    )

    # Should have exactly one email sent
    assert_has_one(result.output_events, EmailSent)
    email = filter_events_by_type(result.output_events, EmailSent)[0]

    # Verify it's correctly threaded via auto-inference
    assert email.email_id_replied_to == email_id, (
        f"Expected auto-inference to thread to email_id={email_id}, "
        f"got email_id_replied_to={email.email_id_replied_to}"
    )


# ---------------------------------------------------------------------------
#  Multi-contact email visibility tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_received_email_appears_in_all_cc_recipient_threads(initialized_cm):
    """
    When an email is received with CC recipients, it should appear in ALL
    contacts' threads (sender AND CC'd contacts), not just the sender's thread.

    This ensures the LLM has complete context when viewing any contact's thread.
    """
    from unity.conversation_manager.cm_types import Medium
    from unity.conversation_manager.domains.contact_index import EmailMessage

    cm = initialized_cm
    alice = TEST_CONTACTS[0]  # Alice Smith - sender
    bob = TEST_CONTACTS[1]  # Bob Johnson - CC'd
    charlie = TEST_CONTACTS[2]  # Charlie Davis - CC'd

    email_id = "CAKx7fQ_multi_contact_received@mail.gmail.com"

    # Alice sends email with Bob and Charlie CC'd
    # run_llm=False: this test verifies thread indexing infrastructure (role
    # tagging, cross-contact fan-out), not LLM response behavior.
    await cm.step(
        EmailReceived(
            contact=alice,
            subject="Team Update",
            body="Hi everyone, here's the weekly update.",
            email_id=email_id,
            attachments=[],
            to=[],  # Assistant is the direct recipient
            cc=[bob["email_address"], charlie["email_address"]],
        ),
        run_llm=False,
    )

    # Verify the email appears in Alice's thread (sender)
    alice_emails = cm.cm.contact_index.get_messages_for_contact(
        alice["contact_id"],
        Medium.EMAIL,
    )
    assert (
        len(alice_emails) == 1
    ), f"Expected 1 email in Alice's thread, got {len(alice_emails)}"
    alice_email = alice_emails[0]
    assert isinstance(alice_email, EmailMessage)
    assert (
        alice_email.contact_role == "sender"
    ), f"Expected 'sender' role, got '{alice_email.contact_role}'"

    # Verify the email appears in Bob's thread (CC'd)
    bob_emails = cm.cm.contact_index.get_messages_for_contact(
        bob["contact_id"],
        Medium.EMAIL,
    )
    assert (
        len(bob_emails) == 1
    ), f"Expected 1 email in Bob's thread, got {len(bob_emails)}"
    bob_email = bob_emails[0]
    assert isinstance(bob_email, EmailMessage)
    assert (
        bob_email.contact_role == "cc"
    ), f"Expected 'cc' role for Bob, got '{bob_email.contact_role}'"

    # Verify the email appears in Charlie's thread (CC'd)
    charlie_emails = cm.cm.contact_index.get_messages_for_contact(
        charlie["contact_id"],
        Medium.EMAIL,
    )
    assert (
        len(charlie_emails) == 1
    ), f"Expected 1 email in Charlie's thread, got {len(charlie_emails)}"
    charlie_email = charlie_emails[0]
    assert isinstance(charlie_email, EmailMessage)
    assert (
        charlie_email.contact_role == "cc"
    ), f"Expected 'cc' role for Charlie, got '{charlie_email.contact_role}'"


@pytest.mark.asyncio
@_handle_project
async def test_sent_email_appears_in_all_recipient_threads(initialized_cm):
    """
    When the assistant sends an email with multiple recipients (to, cc, bcc),
    it should appear in ALL contacts' threads with appropriate role markers.

    This ensures the LLM knows what each contact has seen when viewing
    their conversation thread.
    """
    from unity.conversation_manager.cm_types import Medium
    from unity.conversation_manager.domains.contact_index import EmailMessage

    cm = initialized_cm
    alice = TEST_CONTACTS[0]  # TO recipient
    bob = TEST_CONTACTS[1]  # CC recipient
    charlie = TEST_CONTACTS[2]  # BCC recipient

    # Simulate an EmailSent event (as would be generated by send_email action)
    from unity.conversation_manager.events import EmailSent

    # run_llm=False: this test verifies thread indexing infrastructure (role
    # tagging, cross-contact fan-out), not LLM response behavior.
    await cm.step(
        EmailSent(
            contact=alice,  # Primary contact (first in TO)
            subject="Important Announcement",
            body="Please review the attached document.",
            email_id_replied_to=None,
            attachments=[],
            to=[alice["email_address"]],
            cc=[bob["email_address"]],
            bcc=[charlie["email_address"]],
        ),
        run_llm=False,
    )

    # Verify the email appears in Alice's thread (TO recipient)
    alice_emails = cm.cm.contact_index.get_messages_for_contact(
        alice["contact_id"],
        Medium.EMAIL,
    )
    assert (
        len(alice_emails) == 1
    ), f"Expected 1 email in Alice's thread, got {len(alice_emails)}"
    alice_email = alice_emails[0]
    assert isinstance(alice_email, EmailMessage)
    assert (
        alice_email.contact_role == "to"
    ), f"Expected 'to' role for Alice, got '{alice_email.contact_role}'"
    assert alice_email.name == "You", "Sent emails should show 'You' as sender"

    # Verify the email appears in Bob's thread (CC recipient)
    bob_emails = cm.cm.contact_index.get_messages_for_contact(
        bob["contact_id"],
        Medium.EMAIL,
    )
    assert (
        len(bob_emails) == 1
    ), f"Expected 1 email in Bob's thread, got {len(bob_emails)}"
    bob_email = bob_emails[0]
    assert isinstance(bob_email, EmailMessage)
    assert (
        bob_email.contact_role == "cc"
    ), f"Expected 'cc' role for Bob, got '{bob_email.contact_role}'"

    # Verify the email appears in Charlie's thread (BCC recipient)
    charlie_emails = cm.cm.contact_index.get_messages_for_contact(
        charlie["contact_id"],
        Medium.EMAIL,
    )
    assert (
        len(charlie_emails) == 1
    ), f"Expected 1 email in Charlie's thread, got {len(charlie_emails)}"
    charlie_email = charlie_emails[0]
    assert isinstance(charlie_email, EmailMessage)
    assert (
        charlie_email.contact_role == "bcc"
    ), f"Expected 'bcc' role for Charlie, got '{charlie_email.contact_role}'"


@pytest.mark.asyncio
@_handle_project
async def test_email_appears_in_global_thread_for_all_contacts(initialized_cm):
    """
    When an email involves multiple contacts, it should also appear in each
    contact's global thread (not just the email-specific thread).
    """
    from unity.conversation_manager.domains.contact_index import EmailMessage

    cm = initialized_cm
    alice = TEST_CONTACTS[0]
    bob = TEST_CONTACTS[1]

    email_id = "CAKx7fQ_global_thread_test@mail.gmail.com"

    # Alice sends email with Bob CC'd
    await cm.step(
        EmailReceived(
            contact=alice,
            subject="Quick Question",
            body="Can you help me with something?",
            email_id=email_id,
            attachments=[],
            cc=[bob["email_address"]],
        ),
    )

    # Verify the original email is in Alice's messages
    # Filter by role='user' to find the received email (not any replies the LLM may have sent)
    alice_all = cm.cm.contact_index.get_messages_for_contact(alice["contact_id"])
    alice_received_emails = [
        m
        for m in alice_all
        if isinstance(m, EmailMessage)
        and m.email_id == email_id
        and m.contact_role == "sender"
        and m.role == "user"
    ]
    assert len(alice_received_emails) == 1, (
        f"Expected 1 received email in Alice's messages, "
        f"found {len(alice_received_emails)}: {alice_received_emails}"
    )

    # Verify the original email is also in Bob's messages
    # Filter by role='user' to find the received email (not any replies the LLM may have sent)
    bob_all = cm.cm.contact_index.get_messages_for_contact(bob["contact_id"])
    bob_received_emails = [
        m
        for m in bob_all
        if isinstance(m, EmailMessage)
        and m.email_id == email_id
        and m.contact_role == "cc"
        and m.role == "user"
    ]
    assert len(bob_received_emails) == 1, (
        f"Expected 1 received email in Bob's messages (CC'd), "
        f"found {len(bob_received_emails)}: {bob_received_emails}"
    )


@pytest.mark.asyncio
@_handle_project
async def test_email_with_to_and_cc_shows_correct_roles(initialized_cm):
    """
    When a received email has both TO and CC recipients, each contact should
    see the email with their correct role (to vs cc).
    """
    from unity.conversation_manager.cm_types import Medium

    cm = initialized_cm
    alice = TEST_CONTACTS[0]  # Sender
    bob = TEST_CONTACTS[1]  # TO recipient
    charlie = TEST_CONTACTS[2]  # CC recipient

    email_id = "CAKx7fQ_to_cc_roles@mail.gmail.com"

    # Alice sends email to Bob with Charlie CC'd
    # run_llm=False: this test verifies role assignment (sender/to/cc)
    # on the contact_index, not LLM response behavior.
    await cm.step(
        EmailReceived(
            contact=alice,
            subject="For Bob (CC Charlie)",
            body="Bob, please handle this. Charlie, FYI.",
            email_id=email_id,
            attachments=[],
            to=[bob["email_address"]],  # Bob is in TO
            cc=[charlie["email_address"]],  # Charlie is in CC
        ),
        run_llm=False,
    )

    # Check Alice (sender)
    alice_emails = cm.cm.contact_index.get_messages_for_contact(
        alice["contact_id"],
        Medium.EMAIL,
    )
    assert alice_emails[0].contact_role == "sender"

    # Check Bob (TO)
    bob_emails = cm.cm.contact_index.get_messages_for_contact(
        bob["contact_id"],
        Medium.EMAIL,
    )
    assert (
        bob_emails[0].contact_role == "to"
    ), f"Bob should have 'to' role, got '{bob_emails[0].contact_role}'"

    # Check Charlie (CC)
    charlie_emails = cm.cm.contact_index.get_messages_for_contact(
        charlie["contact_id"],
        Medium.EMAIL,
    )
    assert (
        charlie_emails[0].contact_role == "cc"
    ), f"Charlie should have 'cc' role, got '{charlie_emails[0].contact_role}'"


@pytest.mark.asyncio
@_handle_project
async def test_no_duplicate_email_when_contact_in_multiple_fields(initialized_cm):
    """
    If a contact appears in multiple fields (e.g., both sender and TO), they
    should only get ONE copy of the email in their thread (no duplicates).
    """
    from unity.conversation_manager.cm_types import Medium

    cm = initialized_cm
    alice = TEST_CONTACTS[0]

    email_id = "CAKx7fQ_no_dupe_test@mail.gmail.com"

    # Alice sends email, also lists herself in TO (unusual but possible)
    # run_llm=False: this test verifies deduplication in the contact_index,
    # not LLM response behavior.
    await cm.step(
        EmailReceived(
            contact=alice,
            subject="Self-loop Test",
            body="Testing self-loop scenario.",
            email_id=email_id,
            attachments=[],
            to=[alice["email_address"]],  # Alice is both sender and in TO
        ),
        run_llm=False,
    )

    # Alice should only have ONE email in her thread (not duplicated)
    alice_emails = cm.cm.contact_index.get_messages_for_contact(
        alice["contact_id"],
        Medium.EMAIL,
    )
    assert (
        len(alice_emails) == 1
    ), f"Expected 1 email (no duplicate), got {len(alice_emails)}"

    # First role should be "sender" (processed first)
    assert alice_emails[0].contact_role == "sender"
