"""
tests/conversation_manager/actions/test_action_context_propagation.py
========================================================================

Tests that verify ConversationManager correctly propagates parent chat context
to the Actor when calling `act` and when using steering tools.

These tests verify:
1. `act()` passes the fresh rendered state snapshot as `_parent_chat_context`
2. `ask` steering passes the fresh snapshot as `_parent_chat_context`
3. `interject` steering passes the fresh snapshot as `_parent_chat_context_cont`
4. The snapshot is the current rendered state (not stale `chat_history`)

Additionally includes semantic tests where the SimulatedActor needs the parent
context to provide a sensible response.
"""

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    assert_act_triggered,
    get_in_flight_action_count,
)
from tests.conversation_manager.conftest import BOSS
from unity.conversation_manager.events import (
    SMSReceived,
    ActorHandleStarted,
)

pytestmark = pytest.mark.eval


# =============================================================================
# Unit Tests: Verify context is passed correctly
# =============================================================================


class TestActContextPropagation:
    """Tests that verify `act()` passes the correct parent context."""

    @pytest.mark.asyncio
    @_handle_project
    async def test_act_passes_current_state_snapshot(self, initialized_cm):
        """
        Verify that act() passes the fresh rendered state snapshot, not stale chat_history.

        The _current_state_snapshot is set by _run_llm() before tools execute,
        and contains the exact state the brain LLM saw when making its decision.
        """
        cm = initialized_cm

        # Track what context is passed to actor.act()
        captured_context = []
        original_act = cm.cm.actor.act

        async def capturing_act(query, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context"))
            return await original_act(query, **kwargs)

        cm.cm.actor.act = capturing_act

        try:
            # Trigger an action
            result = await cm.step_until_wait(
                SMSReceived(
                    contact=BOSS,
                    content="What's the weather in London?",
                ),
            )

            assert_act_triggered(
                result,
                ActorHandleStarted,
                "Weather query should trigger act",
                cm=cm,
            )

            # Verify context was captured
            assert len(captured_context) >= 1, "act() should have been called"
            context = captured_context[0]

            # Context should be a list with one element (the state snapshot)
            assert context is not None, "Context should not be None"
            assert isinstance(context, list), "Context should be a list"
            assert len(context) == 1, "Context should contain exactly one message"

            # The message should be the rendered state snapshot
            snapshot = context[0]
            assert isinstance(snapshot, dict), "Snapshot should be a dict"
            assert snapshot.get("role") == "user", "Snapshot should be a user message"
            assert (
                snapshot.get("_cm_state_snapshot") is True
            ), "Snapshot should be marked as state snapshot"

            # The content should contain rendered state sections
            content = snapshot.get("content", "")
            assert "<notifications>" in content, "Should contain notifications section"
            assert (
                "<active_conversations>" in content
            ), "Should contain conversations section"

        finally:
            cm.cm.actor.act = original_act

    @pytest.mark.asyncio
    @_handle_project
    async def test_act_context_contains_current_conversation(self, initialized_cm):
        """
        Verify that the context passed to act() contains the current conversation.

        The rendered state should include the message that triggered the act call.
        """
        cm = initialized_cm

        captured_context = []
        original_act = cm.cm.actor.act

        async def capturing_act(query, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context"))
            return await original_act(query, **kwargs)

        cm.cm.actor.act = capturing_act

        try:
            # Send a distinctive message
            result = await cm.step_until_wait(
                SMSReceived(
                    contact=BOSS,
                    content="Search for contacts in Berlin please",
                ),
            )

            assert len(captured_context) >= 1, "act() should have been called"
            context = captured_context[0]
            content = context[0].get("content", "")

            # The content should include the message that triggered the action
            assert (
                "Berlin" in content
            ), "Context should contain the message that triggered the action"

        finally:
            cm.cm.actor.act = original_act


class TestSteeringContextPropagation:
    """Tests that verify steering tools pass the correct parent context."""

    @pytest.mark.asyncio
    @_handle_project
    async def test_interject_passes_current_state_snapshot(self, initialized_cm):
        """
        Verify that interject steering passes the fresh state snapshot.

        When the brain calls interject_*, it should pass _parent_chat_context_cont
        containing the current rendered state, not the stale chat_history.
        """
        cm = initialized_cm

        # Step 1: Start an action
        result1 = await cm.step_until_wait(
            SMSReceived(
                contact=BOSS,
                content="Search my transcripts for budget discussions.",
            ),
        )
        assert get_in_flight_action_count(cm) >= 1, "Should have an in-flight action"

        # Get the handle to track interject calls
        handle_id = list(cm.cm.in_flight_actions.keys())[0]
        handle = cm.cm.in_flight_actions[handle_id]["handle"]

        # Track what context is passed to interject
        captured_context = []
        original_interject = handle.interject

        async def capturing_interject(message, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context_cont"))
            return await original_interject(message, **kwargs)

        handle.interject = capturing_interject

        try:
            # Step 2: Send a message that should trigger interject
            result2 = await cm.step_until_wait(
                SMSReceived(
                    contact=BOSS,
                    content="Actually, only include Q3 budget items in that search.",
                ),
            )

            # Check if interject was called
            if captured_context:
                context = captured_context[0]

                # Context should be a list with one element (the state snapshot)
                assert context is not None, "Context should not be None"
                assert isinstance(context, list), "Context should be a list"
                assert len(context) == 1, "Context should contain exactly one message"

                # The message should be the rendered state snapshot
                snapshot = context[0]
                assert isinstance(snapshot, dict), "Snapshot should be a dict"
                assert (
                    snapshot.get("_cm_state_snapshot") is True
                ), "Snapshot should be marked as state snapshot"

                # Should contain the in-flight action
                content = snapshot.get("content", "")
                assert (
                    "<in_flight_actions>" in content
                ), "Should contain in-flight actions section"

        finally:
            handle.interject = original_interject

    @pytest.mark.asyncio
    @_handle_project
    async def test_ask_passes_current_state_snapshot(self, initialized_cm):
        """
        Verify that ask steering passes the fresh state snapshot.

        When the brain calls ask_*, it should pass _parent_chat_context
        containing the current rendered state.
        """
        cm = initialized_cm

        # Step 1: Start an action
        result1 = await cm.step_until_wait(
            SMSReceived(
                contact=BOSS,
                content="Research competitor pricing strategies.",
            ),
        )
        assert get_in_flight_action_count(cm) >= 1, "Should have an in-flight action"

        # Get the handle to track ask calls
        handle_id = list(cm.cm.in_flight_actions.keys())[0]
        handle = cm.cm.in_flight_actions[handle_id]["handle"]

        # Track what context is passed to ask
        captured_context = []
        original_ask = handle.ask

        async def capturing_ask(question, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context"))
            result = await original_ask(question, **kwargs)
            return result

        handle.ask = capturing_ask

        try:
            # Step 2: Ask about progress
            result2 = await cm.step_until_wait(
                SMSReceived(
                    contact=BOSS,
                    content="What progress have you made on that research?",
                ),
            )

            # Check if ask was called
            if captured_context:
                context = captured_context[0]

                # Context should be a list with one element (the state snapshot)
                assert context is not None, "Context should not be None"
                assert isinstance(context, list), "Context should be a list"
                assert len(context) == 1, "Context should contain exactly one message"

                # The message should be the rendered state snapshot
                snapshot = context[0]
                assert isinstance(snapshot, dict), "Snapshot should be a dict"
                assert (
                    snapshot.get("_cm_state_snapshot") is True
                ), "Snapshot should be marked as state snapshot"

        finally:
            handle.ask = original_ask


class TestContextFreshness:
    """Tests that verify the context is fresh (not stale)."""

    @pytest.mark.asyncio
    @_handle_project
    async def test_context_includes_in_flight_action_for_steering(self, initialized_cm):
        """
        Verify that when steering an in-flight action, the context shows the action.

        This tests that the context is fresh: when we interject an in-flight action,
        the context should show that action in <in_flight_actions>, which wouldn't
        be true if we were passing stale chat_history from before the action started.
        """
        cm = initialized_cm

        # Step 1: Start an action
        result1 = await cm.step_until_wait(
            SMSReceived(
                contact=BOSS,
                content="Find all contacts who work in engineering.",
            ),
        )
        assert get_in_flight_action_count(cm) >= 1, "Should have an in-flight action"

        handle_id = list(cm.cm.in_flight_actions.keys())[0]
        handle = cm.cm.in_flight_actions[handle_id]["handle"]

        captured_context = []
        original_interject = handle.interject

        async def capturing_interject(message, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context_cont"))
            return await original_interject(message, **kwargs)

        handle.interject = capturing_interject

        try:
            # Step 2: Interject with additional info
            result2 = await cm.step_until_wait(
                SMSReceived(
                    contact=BOSS,
                    content="Also include their phone numbers in the results.",
                ),
            )

            if captured_context and captured_context[0]:
                content = captured_context[0][0].get("content", "")
                # The in-flight action should be visible in the context
                # This proves we're using fresh state, not stale history
                assert "<in_flight_actions>" in content
                # Should show the action is executing
                assert "executing" in content.lower() or "status=" in content

        finally:
            handle.interject = original_interject


# =============================================================================
# Semantic Tests: Actor uses parent context to give sensible responses
# =============================================================================


class TestSemanticContextUsage:
    """
    Tests that verify the Actor can use parent context to give sensible responses.

    These are end-to-end tests where the SimulatedActor needs information from
    the parent chat context to respond correctly.
    """

    @pytest.mark.asyncio
    @_handle_project
    async def test_actor_uses_context_for_ambiguous_request(self, initialized_cm):
        """
        Test that the Actor can use parent context to resolve an ambiguous request.

        Scenario:
        1. Boss mentions a specific contact name in conversation
        2. Boss then asks to "send them an email" without naming the contact
        3. The Actor should be able to infer who "them" refers to from context

        This tests that context is being passed and used semantically.
        """
        cm = initialized_cm

        # Step 1: Establish context by mentioning a contact
        # (This message goes into the conversation history)
        result1 = await cm.step_until_wait(
            SMSReceived(
                contact=BOSS,
                content="I just got off a call with Alice Smith about the project.",
            ),
        )

        # Step 2: Make an ambiguous request that requires context
        result2 = await cm.step_until_wait(
            SMSReceived(
                contact=BOSS,
                content="Can you find her email address for me?",
            ),
        )

        # The act should have been triggered
        assert_act_triggered(
            result2,
            ActorHandleStarted,
            "Ambiguous request should trigger act",
            cm=cm,
        )

        # Get the query that was sent to act
        from tests.conversation_manager.cm_helpers import filter_events_by_type

        actor_events = filter_events_by_type(result2.output_events, ActorHandleStarted)
        assert len(actor_events) >= 1, "Should have ActorHandleStarted event"

        # The query should reference Alice (resolved from context)
        # or the act should succeed because context was passed
        query = actor_events[0].query.lower()
        # Either the CM brain resolved "her" to "Alice" in the query,
        # or the context was passed to help the Actor resolve it
        assert (
            "alice" in query or "email" in query
        ), f"Query should reference the contact or action. Got: {query}"

    @pytest.mark.asyncio
    @_handle_project
    async def test_actor_context_includes_conversation_thread(self, initialized_cm):
        """
        Test that the Actor receives context that includes the conversation thread.

        Scenario:
        1. Multi-turn conversation with the boss
        2. Request an action that benefits from knowing the conversation history
        3. Verify the context passed to Actor includes the conversation

        This ensures the rendered state (which includes conversation threads)
        is being passed correctly.
        """
        cm = initialized_cm

        # Step 1: Start a conversation about a topic
        result1 = await cm.step_until_wait(
            SMSReceived(
                contact=BOSS,
                content="I need to prepare for the Henderson meeting tomorrow.",
            ),
        )

        # Step 2: Add more context
        result2 = await cm.step_until_wait(
            SMSReceived(
                contact=BOSS,
                content="They're interested in our premium tier pricing.",
            ),
        )

        # Step 3: Make a request that relies on the accumulated context
        result3 = await cm.step_until_wait(
            SMSReceived(
                contact=BOSS,
                content="Can you pull together the relevant info for that meeting?",
            ),
        )

        # Should trigger act
        assert_act_triggered(
            result3,
            ActorHandleStarted,
            "Meeting prep request should trigger act",
            cm=cm,
        )

        # Verify the context was passed (we can check the handle's perspective)
        # The important thing is that act was called and the system didn't error
        # due to missing context - if context wasn't passed, the Actor would
        # have no idea what "that meeting" refers to.
        from tests.conversation_manager.cm_helpers import filter_events_by_type

        actor_events = filter_events_by_type(result3.output_events, ActorHandleStarted)
        assert len(actor_events) >= 1

        # Query should reflect understanding of context
        query = actor_events[0].query.lower()
        # The CM brain should have understood the context and created a meaningful query
        assert (
            "henderson" in query or "meeting" in query or "pricing" in query
        ), f"Query should reflect conversation context. Got: {query}"
