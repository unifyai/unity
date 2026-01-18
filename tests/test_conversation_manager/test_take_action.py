"""
tests/test_conversation_manager/test_take_action.py
===================================================

Tests that verify ConversationManager correctly delegates to `act` for various
types of requests that require access to knowledge, resources, or the world.

These tests use the same categories of requests as tests/test_actor/test_state_managers
but phrased as natural conversational scenarios. At this level we don't verify
which inner state manager is reached - we simply verify that the request lands
on the `act` method with a SimulatedActor under the hood.

Each scenario presents a natural conversational request from the boss that should
trigger the assistant to call `act`.
"""

import pytest

from tests.helpers import _handle_project
from tests.test_conversation_manager.conftest import TEST_CONTACTS
from unity.conversation_manager.events import (
    SMSReceived,
    ActorHandleStarted,
)

pytestmark = pytest.mark.eval

# Convenience references to test contacts
BOSS = TEST_CONTACTS[1]  # contact_id 1 - the main user

# Maximum LLM steps for efficient tool calling
# - Ideal: 2 steps (action + acknowledge concurrent, then wait)
# - Acceptable: 3 steps (action, acknowledge, wait - or action + query + wait)
MAX_EFFICIENT_STEPS = 3


def _only(events, typ):
    return [e for e in events if isinstance(e, typ)]


def _assert_efficient(result, context: str = ""):
    """Assert that the LLM completed efficiently (concurrent tool calls + wait)."""
    assert result.llm_step_count <= MAX_EFFICIENT_STEPS, (
        f"Expected efficient concurrent tool calling (<= {MAX_EFFICIENT_STEPS} steps), "
        f"but took {result.llm_step_count} steps. "
        f"LLM should call tools concurrently in one step, then wait. {context}"
    )


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

    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called for contact preference lookup, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
    _assert_efficient(result)


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

    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called for location-based contact search, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
    _assert_efficient(result)


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

    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called for contact creation, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
    _assert_efficient(result)


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

    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called for knowledge query, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
    _assert_efficient(result)


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

    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called for product knowledge query, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
    _assert_efficient(result)


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

    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called for storing knowledge, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
    _assert_efficient(result)


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

    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called for task query, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
    _assert_efficient(result)


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

    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called for task creation, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
    _assert_efficient(result)


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

    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called for priority task query, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
    _assert_efficient(result)


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

    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called for transcript search, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
    _assert_efficient(result)


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

    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called for recent messages search, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
    _assert_efficient(result)


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

    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called for topic-based message search, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
    _assert_efficient(result)


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

    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called for weather query, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
    _assert_efficient(result)


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

    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called for news query, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
    _assert_efficient(result)


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

    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called for current events query, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
    _assert_efficient(result)


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

    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called for guidance query, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
    _assert_efficient(result)


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

    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called for find-and-action request, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
    _assert_efficient(result)


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

    actor_events = _only(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, (
        f"Expected act to be called for research request, "
        f"got events: {[type(e).__name__ for e in result.output_events]}"
    )
    _assert_efficient(result)
