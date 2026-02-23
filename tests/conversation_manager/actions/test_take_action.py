"""
tests/conversation_manager/test_take_action.py
===================================================

Tests that verify ConversationManager correctly delegates to ``act`` for
requests that require the general-purpose Actor (knowledge, tasks, web
search, guidance, files, combined/research).

Contact-specific routing (``ask_about_contacts``, ``update_contacts``) and
transcript-specific routing (``query_past_transcripts``) are tested in their
own dedicated modules under this directory.
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
