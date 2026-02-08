"""
tests/conversation_manager/test_take_action.py
===================================================

Tests that verify ConversationManager correctly delegates to `act` for various
types of requests that require access to knowledge, resources, or the world.

These tests use the same categories of requests as tests/actor/test_state_managers
but phrased as natural conversational scenarios. At this level we don't verify
which inner state manager is reached - we simply verify that the request lands
on the `act` method with a SimulatedActor under the hood.

Each scenario presents a natural conversational request from the boss that should
trigger the assistant to call `act`.
"""

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    assert_act_triggered,
    assert_efficient,
    filter_events_by_type,
)
from tests.conversation_manager.conftest import BOSS
from unity.conversation_manager.events import (
    SMSReceived,
    EmailReceived,
    UnifyMessageReceived,
    ActorHandleStarted,
)

pytestmark = pytest.mark.eval

# Note: BOSS (contact_id=1) is imported from conftest.py


# ---------------------------------------------------------------------------
#  Contact-related requests -> should trigger act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_contact_lookup_triggers_act(initialized_cm):
    """
    Boss asks about contact preferences -> should call act to search contacts.

    Natural scenario: Boss wants to know how to reach someone.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Does Sarah prefer phone or email?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Contact preference lookup should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_contact_search_by_location_triggers_act(initialized_cm):
    """
    Boss asks about contacts in a location -> should call act.

    Natural scenario: Boss planning a trip and wants to meet contacts.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="I'm heading to Berlin next week. Do we know anyone there?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Location-based contact search should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_create_contact_triggers_act(initialized_cm):
    """
    Boss asks to save a new contact -> should call act.

    Natural scenario: Boss met someone and wants to save their details.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Just met Jane Doe at the conference. Can you save her email jane.d@example.com?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Contact creation should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_save_service_number_act_query_describes_entity(initialized_cm):
    """
    Boss shares a service/support number -> act query should describe the
    organisation, not the name of whoever answered the phone.

    Scenario: boss says "Save this number 8005551234 — it's the Acme billing
    support line. Sarah answered when I called." The act query should frame
    the contact as a service number for Acme, not as "add Sarah".
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content=(
                "Save this number 8005551234 - it's the Acme billing support "
                "line. Sarah answered when I called."
            ),
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Saving a service number should trigger act",
        cm=cm,
    )

    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    act_query = actor_events[0].query.lower()

    # The query should reference the service/company, not "add Sarah"
    assert "8005551234" in act_query, (
        f"act query should include the phone number, got: {actor_events[0].query}"
    )
    assert "acme" in act_query or "billing" in act_query or "support" in act_query, (
        f"act query should describe the service/organisation, got: {actor_events[0].query}"
    )
    # Must NOT frame this as adding a person named Sarah
    query_words = act_query.split()
    # Check that "sarah" doesn't appear as a contact name in typical patterns
    sarah_as_contact = (
        "add sarah" in act_query
        or "create sarah" in act_query
        or "save sarah" in act_query
        or "contact sarah" in act_query
        or ("sarah" in act_query and "first_name" in act_query)
    )
    assert not sarah_as_contact, (
        f"act query should NOT frame Sarah as the contact name — she is a "
        f"transient representative, got: {actor_events[0].query}"
    )

    assert_efficient(result, 3)


# ---------------------------------------------------------------------------
#  Knowledge-related requests -> should trigger act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_knowledge_query_triggers_act(initialized_cm):
    """
    Boss asks about company policy -> should call act to search knowledge.

    Natural scenario: Boss needs to know a policy detail.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="What are our office hours again?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Knowledge query should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_knowledge_about_product_triggers_act(initialized_cm):
    """
    Boss asks about product information -> should call act.

    Natural scenario: Boss needs warranty/product details for a customer.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="A customer is asking about Tesla warranty. What do we have on file?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Product knowledge query should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_store_knowledge_triggers_act(initialized_cm):
    """
    Boss asks to remember some information -> should call act.

    Natural scenario: Boss wants to store a piece of information.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Make a note that our refund window is 30 days for unopened items.",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Storing knowledge should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


# ---------------------------------------------------------------------------
#  Task-related requests -> should trigger act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_task_query_triggers_act(initialized_cm):
    """
    Boss asks about scheduled tasks -> should call act.

    Natural scenario: Boss checking their schedule.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="What do I have on my plate today?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Task query should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_create_task_triggers_act(initialized_cm):
    """
    Boss asks to schedule something -> should call act.

    Natural scenario: Boss wants to create a reminder/task.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Remind me to call Alice about the Q3 budget tomorrow at 9am.",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Task creation should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_priority_task_query_triggers_act(initialized_cm):
    """
    Boss asks about high-priority items -> should call act.

    Natural scenario: Boss wants to focus on urgent matters.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="What's the most urgent thing I need to deal with?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Priority task query should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


# ---------------------------------------------------------------------------
#  Transcript-related requests -> should trigger act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_transcript_search_triggers_act(initialized_cm):
    """
    Boss asks about a past conversation -> should call act.

    Natural scenario: Boss trying to remember what was discussed.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="What did David say about the project deadline last week?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Transcript search should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_recent_messages_search_triggers_act(initialized_cm):
    """
    Boss asks about recent messages -> should call act.

    Natural scenario: Boss checking for updates from someone.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Has Alice messaged me in the last day or so?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Recent messages search should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_specific_topic_search_triggers_act(initialized_cm):
    """
    Boss asks about messages on a specific topic -> should call act.

    Natural scenario: Boss looking for a specific discussion.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Can you find the last message where someone mentioned the Q3 budget?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Topic-based message search should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


# ---------------------------------------------------------------------------
#  Web search requests -> should trigger act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_weather_query_triggers_act(initialized_cm):
    """
    Boss asks about current weather -> should call act for web search.

    Natural scenario: Boss planning travel or outdoor activity.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="What's the weather like in Berlin today?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Weather query should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_news_query_triggers_act(initialized_cm):
    """
    Boss asks about current news -> should call act for web search.

    Natural scenario: Boss wants to stay informed.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="What's happening in the news today? Any major headlines?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "News query should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_current_events_query_triggers_act(initialized_cm):
    """
    Boss asks about a recent event -> should call act for web search.

    Natural scenario: Boss following industry developments.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Any notable AI announcements this week I should know about?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Current events query should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


# ---------------------------------------------------------------------------
#  Guidance-related requests -> should trigger act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_guidance_query_triggers_act(initialized_cm):
    """
    Boss asks for guidance on a process -> should call act.

    Natural scenario: Boss needs to follow a procedure.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="We might have a security incident. What's the protocol?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Guidance query should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


# ---------------------------------------------------------------------------
#  Combined/complex requests -> should trigger act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_find_and_action_triggers_act(initialized_cm):
    """
    Boss asks to find something and do something with it -> should call act.

    Natural scenario: Boss wants information found and acted upon.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="Find Bob's latest invoice and let me know if it's been paid.",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Find-and-action request should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_research_request_triggers_act(initialized_cm):
    """
    Boss asks for research on a topic -> should call act.

    Natural scenario: Boss needs background information compiled.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        SMSReceived(
            contact=BOSS,
            content="I'm meeting with Contoso tomorrow. Can you pull together some background on them?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Research request should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


# ---------------------------------------------------------------------------
#  File/attachment-related requests -> should trigger act with filepath
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_email_summarize_attachment_triggers_act_with_filepath(initialized_cm):
    """
    Email with attachment + request to summarize -> act should include filepath.

    Natural scenario: Boss receives a document via email and asks the assistant
    to summarize it. The assistant should call `act` with the filepath of the
    auto-downloaded attachment so the Actor can access and process the file.

    The rendered email shows: "Attachments: report.pdf (auto-downloaded to Downloads/report.pdf)"
    so the LLM should know the file location and include it in the act query.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        EmailReceived(
            contact=BOSS,
            subject="Q3 Report",
            body="Please summarize this PDF for me.",
            email_id="test_summarize_attachment",
            attachments=["quarterly_report.pdf"],
        ),
    )

    # First verify that act was triggered
    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Summarize attachment request should trigger act",
        cm=cm,
    )

    # Now verify the act query includes the filepath
    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, "Expected at least one ActorHandleStarted event"

    # The query sent to act should include the download path
    act_query = actor_events[0].query.lower()
    assert "downloads" in act_query and "quarterly_report.pdf" in act_query, (
        f"Expected act query to include filepath 'Downloads/quarterly_report.pdf', "
        f"got query: {actor_events[0].query}"
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_summarize_attachment_triggers_act_with_filepath(
    initialized_cm,
):
    """
    Unify message with attachment + request to summarize -> act should include filepath.

    Natural scenario: Boss sends a document via Unify console and asks the assistant
    to summarize it. The assistant should call `act` with the filepath of the
    auto-downloaded attachment so the Actor can access and process the file.

    The rendered message shows: "[Attachments: report.pdf (auto-downloaded to Downloads/report.pdf)]"
    so the LLM should know the file location and include it in the act query.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content="Please summarize this PDF for me.",
            attachments=[{"id": "att-1", "filename": "quarterly_report.pdf"}],
        ),
    )

    # First verify that act was triggered
    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Summarize attachment request should trigger act",
        cm=cm,
    )

    # Now verify the act query includes the filepath
    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, "Expected at least one ActorHandleStarted event"

    # The query sent to act should include the download path
    act_query = actor_events[0].query.lower()
    assert "downloads" in act_query and "quarterly_report.pdf" in act_query, (
        f"Expected act query to include filepath 'Downloads/quarterly_report.pdf', "
        f"got query: {actor_events[0].query}"
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)
