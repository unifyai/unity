"""
tests/conversation_manager/actions/test_action_context_propagation.py
========================================================================

Tests that verify ConversationManager correctly propagates parent chat context
to the Actor when calling `act` and when using steering tools.

These tests verify:
1. `act()` passes the fresh rendered state snapshot as `_parent_chat_context`
2. `ask` steering passes the fresh snapshot as `_parent_chat_context`
3. `interject` steering passes the fresh snapshot as `_parent_chat_context_cont`
4. The snapshot content is correct and would be legible to an LLM

Test Design:
- Symbolic tests: Directly call tools with fixed inputs, capture arguments
- Content tests: Verify the captured context contains expected information

All tests bypass the CM brain LLM to ensure determinism.
"""

import pytest

from tests.helpers import _handle_project
from unity.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
)

pytestmark = pytest.mark.eval


# =============================================================================
# Symbolic Tests: Verify parameters are passed correctly
# =============================================================================


class TestActContextPropagation:
    """Tests that verify `act()` passes the correct parent context."""

    @pytest.mark.asyncio
    @_handle_project
    async def test_act_passes_current_state_snapshot(self, initialized_cm):
        """
        Verify that act() passes [_current_state_snapshot] as _parent_chat_context.

        Setup:
        - Set _current_state_snapshot to a known message
        - Call act() directly via brain_action_tools
        - Capture the _parent_chat_context argument passed to actor.act()

        Assert:
        - _parent_chat_context is a list with one element
        - That element is exactly _current_state_snapshot
        """
        cm = initialized_cm.cm

        # Create a known state snapshot
        test_snapshot = {
            "role": "user",
            "content": "<notifications></notifications>\n<test>unique_test_content_12345</test>",
            "_cm_state_snapshot": True,
        }
        cm._current_state_snapshot = test_snapshot

        # Capture what's passed to actor.act()
        captured_context = []
        original_act = cm.actor.act

        async def capturing_act(query, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context"))
            # Return a mock handle that won't block
            handle = await original_act(query, **kwargs)
            return handle

        cm.actor.act = capturing_act

        try:
            # Create brain action tools and call act directly
            brain_tools = ConversationManagerBrainActionTools(cm)
            await brain_tools.act(query="test query")

            # Verify context was captured
            assert len(captured_context) == 1, "act() should have been called once"
            context = captured_context[0]

            # Verify structure
            assert context is not None, "_parent_chat_context should not be None"
            assert isinstance(context, list), "_parent_chat_context should be a list"
            assert (
                len(context) == 1
            ), "_parent_chat_context should have exactly one element"

            # Verify it's exactly our snapshot
            assert (
                context[0] is test_snapshot
            ), "_parent_chat_context[0] should be the exact _current_state_snapshot object"

        finally:
            cm.actor.act = original_act
            cm._current_state_snapshot = None

    @pytest.mark.asyncio
    @_handle_project
    async def test_act_passes_none_when_no_snapshot(self, initialized_cm):
        """
        Verify that act() passes None when _current_state_snapshot is not set.
        """
        cm = initialized_cm.cm

        # Ensure no snapshot is set
        cm._current_state_snapshot = None

        captured_context = []
        original_act = cm.actor.act

        async def capturing_act(query, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context"))
            handle = await original_act(query, **kwargs)
            return handle

        cm.actor.act = capturing_act

        try:
            brain_tools = ConversationManagerBrainActionTools(cm)
            await brain_tools.act(query="test query")

            assert len(captured_context) == 1
            assert (
                captured_context[0] is None
            ), "_parent_chat_context should be None when no snapshot is set"

        finally:
            cm.actor.act = original_act


class TestSteeringContextPropagation:
    """Tests that verify steering tools pass the correct parent context."""

    @pytest.mark.asyncio
    @_handle_project
    async def test_interject_passes_incremental_diff_when_snapshot_state_available(
        self,
        initialized_cm,
    ):
        """
        Verify that interject steering passes an incremental diff as
        _parent_chat_context_cont when SnapshotState tracking is available.

        This tests the incremental context propagation feature: when an action
        is started, its initial snapshot state is stored. When interjecting,
        only the DIFF between the initial and current snapshot is sent.

        Setup:
        - Create an in-flight action with a mock handle and initial_snapshot_state
        - Set _current_snapshot_state with some new elements (a new message)
        - Call the interject steering tool directly
        - Capture the _parent_chat_context_cont argument

        Assert:
        - _parent_chat_context_cont contains only the diff (new elements)
        - _parent_chat_context_cont is marked with _cm_context_diff: True
        """
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from unity.conversation_manager.domains.renderer import (
            SnapshotState,
            MessageElement,
        )

        cm = initialized_cm.cm

        # Create a mock handle that captures interject calls
        captured_context = []

        async def capturing_interject(message, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context_cont"))

        mock_handle = MagicMock()
        mock_handle.interject = capturing_interject

        # Create initial snapshot state (at time of act())
        initial_ts = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        initial_snapshot_state = SnapshotState(
            full_render="<initial>state</initial>",
            messages=[
                MessageElement(
                    contact_id=1,
                    thread_name="global",
                    index_in_thread=0,
                    timestamp=initial_ts,
                    rendered="[User @ ...]: Hello",
                ),
            ],
        )

        # Register the mock handle as an in-flight action WITH initial snapshot state
        cm.in_flight_actions[0] = {
            "handle": mock_handle,
            "query": "search for test data",
            "handle_actions": [],
            "initial_snapshot_state": initial_snapshot_state,
        }

        # Create current snapshot state with a NEW message (simulating state change)
        new_ts = datetime(2025, 6, 13, 12, 5, 0, tzinfo=timezone.utc)
        current_snapshot_state = SnapshotState(
            full_render="<current>state with new message</current>",
            messages=[
                MessageElement(
                    contact_id=1,
                    thread_name="global",
                    index_in_thread=0,
                    timestamp=initial_ts,
                    rendered="[User @ ...]: Hello",
                ),
                MessageElement(
                    contact_id=1,
                    thread_name="global",
                    index_in_thread=1,
                    timestamp=new_ts,
                    rendered="[User @ ...]: Please also check the calendar",
                ),
            ],
        )
        cm._current_snapshot_state = current_snapshot_state

        try:
            # Build steering tools and find the interject tool
            brain_tools = ConversationManagerBrainActionTools(cm)
            steering_tools = brain_tools.build_action_steering_tools()

            # Find the interject tool (name starts with "interject_")
            interject_tool = None
            for name, tool in steering_tools.items():
                if name.startswith("interject_"):
                    interject_tool = tool
                    break

            assert interject_tool is not None, "Should have an interject tool"

            # Call the interject tool
            await interject_tool(message="additional instruction")

            # Verify context was captured
            assert len(captured_context) == 1, "interject should have been called once"
            context = captured_context[0]

            assert context is not None, "_parent_chat_context_cont should not be None"
            assert isinstance(
                context,
                list,
            ), "_parent_chat_context_cont should be a list"
            assert (
                len(context) == 1
            ), "_parent_chat_context_cont should have one element"

            # Verify it's marked as a diff
            assert (
                context[0].get("_cm_context_diff") is True
            ), "Context should be marked as a diff"

            # Verify the diff contains only the NEW message, not the old one
            diff_content = context[0].get("content", "")
            assert (
                "Please also check the calendar" in diff_content
            ), "Diff should contain the new message"
            assert (
                "<new_messages>" in diff_content
            ), "Diff should have <new_messages> section"
            # The original message should NOT be in the diff
            assert (
                "Hello" not in diff_content
            ), "Diff should NOT contain the original message"

        finally:
            cm.in_flight_actions.clear()
            cm._current_state_snapshot = None
            cm._current_snapshot_state = None

    @pytest.mark.asyncio
    @_handle_project
    async def test_interject_returns_empty_diff_when_nothing_changed(
        self,
        initialized_cm,
    ):
        """
        Verify that interject passes None when there's no diff (nothing changed).

        When the initial snapshot and current snapshot are identical, there's
        no incremental update to send.
        """
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from unity.conversation_manager.domains.renderer import (
            SnapshotState,
            MessageElement,
        )

        cm = initialized_cm.cm

        captured_context = []

        async def capturing_interject(message, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context_cont"))

        mock_handle = MagicMock()
        mock_handle.interject = capturing_interject

        # Create identical initial and current snapshots
        ts = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        snapshot_state = SnapshotState(
            full_render="<state>unchanged</state>",
            messages=[
                MessageElement(
                    contact_id=1,
                    thread_name="global",
                    index_in_thread=0,
                    timestamp=ts,
                    rendered="[User @ ...]: Hello",
                ),
            ],
        )

        cm.in_flight_actions[0] = {
            "handle": mock_handle,
            "query": "search for test data",
            "handle_actions": [],
            "initial_snapshot_state": snapshot_state,
        }
        cm._current_snapshot_state = snapshot_state

        try:
            brain_tools = ConversationManagerBrainActionTools(cm)
            steering_tools = brain_tools.build_action_steering_tools()

            interject_tool = None
            for name, tool in steering_tools.items():
                if name.startswith("interject_"):
                    interject_tool = tool
                    break

            await interject_tool(message="additional instruction")

            assert len(captured_context) == 1
            # When nothing changed, context should be None (empty diff)
            assert (
                captured_context[0] is None
            ), "When nothing changed, _parent_chat_context_cont should be None"

        finally:
            cm.in_flight_actions.clear()
            cm._current_snapshot_state = None

    @pytest.mark.asyncio
    @_handle_project
    async def test_interject_falls_back_to_full_snapshot_when_no_tracking(
        self,
        initialized_cm,
    ):
        """
        Verify backward compatibility: when no snapshot tracking is available,
        interject falls back to using the full _current_state_snapshot.
        """
        from unittest.mock import MagicMock

        cm = initialized_cm.cm

        captured_context = []

        async def capturing_interject(message, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context_cont"))

        mock_handle = MagicMock()
        mock_handle.interject = capturing_interject

        # Register WITHOUT initial_snapshot_state (no tracking)
        cm.in_flight_actions[0] = {
            "handle": mock_handle,
            "query": "search for test data",
            "handle_actions": [],
            # No initial_snapshot_state
        }

        # Set full snapshot (old behavior)
        test_snapshot = {
            "role": "user",
            "content": "<full_state>test content</full_state>",
            "_cm_state_snapshot": True,
        }
        cm._current_state_snapshot = test_snapshot
        cm._current_snapshot_state = None  # No tracking

        try:
            brain_tools = ConversationManagerBrainActionTools(cm)
            steering_tools = brain_tools.build_action_steering_tools()

            interject_tool = None
            for name, tool in steering_tools.items():
                if name.startswith("interject_"):
                    interject_tool = tool
                    break

            await interject_tool(message="additional instruction")

            assert len(captured_context) == 1
            context = captured_context[0]

            # Should fall back to full snapshot
            assert context is not None
            assert len(context) == 1
            assert context[0] is test_snapshot

        finally:
            cm.in_flight_actions.clear()
            cm._current_state_snapshot = None

    @pytest.mark.asyncio
    @_handle_project
    async def test_ask_passes_current_state_snapshot(self, initialized_cm):
        """
        Verify that ask steering passes [_current_state_snapshot] as
        _parent_chat_context.

        Setup:
        - Create an in-flight action with a mock handle
        - Set _current_state_snapshot to a known message
        - Call the ask steering tool directly
        - Capture the _parent_chat_context argument

        Assert:
        - _parent_chat_context is [_current_state_snapshot]
        """
        import asyncio
        from unittest.mock import MagicMock

        cm = initialized_cm.cm

        # Create a mock handle that captures ask calls
        captured_context = []

        class MockAskHandle:
            async def result(self):
                return "mock answer"

        async def capturing_ask(question, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context"))
            return MockAskHandle()

        mock_handle = MagicMock()
        mock_handle.ask = capturing_ask

        # Register the mock handle as an in-flight action
        cm.in_flight_actions[0] = {
            "handle": mock_handle,
            "query": "research competitor pricing",
            "handle_actions": [],
        }

        # Set a known state snapshot
        test_snapshot = {
            "role": "user",
            "content": "<active_conversations>test conversation</active_conversations>",
            "_cm_state_snapshot": True,
        }
        cm._current_state_snapshot = test_snapshot

        try:
            brain_tools = ConversationManagerBrainActionTools(cm)
            steering_tools = brain_tools.build_action_steering_tools()

            # Find the ask tool
            ask_tool = None
            for name, tool in steering_tools.items():
                if name.startswith("ask_"):
                    ask_tool = tool
                    break

            assert ask_tool is not None, "Should have an ask tool"

            # Call the ask tool (don't await the background task result)
            await ask_tool(question="what's the status?")

            # Give the background task a moment to start
            await asyncio.sleep(0.1)

            # Verify context was captured
            assert len(captured_context) == 1, "ask should have been called once"
            context = captured_context[0]

            assert context is not None, "_parent_chat_context should not be None"
            assert isinstance(context, list), "_parent_chat_context should be a list"
            assert len(context) == 1, "_parent_chat_context should have one element"
            assert (
                context[0] is test_snapshot
            ), "_parent_chat_context[0] should be the exact snapshot"

        finally:
            cm.in_flight_actions.clear()
            cm._current_state_snapshot = None


# =============================================================================
# Content Tests: Verify context content is correct
# =============================================================================


class TestContextContent:
    """Tests that verify the context content is correct and legible."""

    @pytest.mark.asyncio
    @_handle_project
    async def test_context_contains_conversation_from_message(self, initialized_cm):
        """
        Verify that when a message triggers act(), the context passed to the Actor
        contains that message's content.

        This tests that context is fresh and includes the current turn.
        """
        from tests.conversation_manager.conftest import BOSS
        from unity.conversation_manager.events import SMSReceived

        cm_driver = initialized_cm
        cm = cm_driver.cm

        # Track what context is passed to actor.act()
        captured_context = []
        original_act = cm.actor.act

        async def capturing_act(query, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context"))
            return await original_act(query, **kwargs)

        cm.actor.act = capturing_act

        try:
            # Send a message with distinctive content
            result = await cm_driver.step_until_wait(
                SMSReceived(
                    contact=BOSS,
                    content="Please find the XYZZY123 document for me.",
                ),
            )

            # Check if act was called (it might not be depending on LLM decision)
            if captured_context and captured_context[0]:
                context = captured_context[0]
                content = context[0].get("content", "")

                # The context should contain the message that triggered the action
                assert "XYZZY123" in content, (
                    "Context should contain the distinctive message content. "
                    f"Got content: {content[:500]}..."
                )

                # Should be marked as a state snapshot
                assert (
                    context[0].get("_cm_state_snapshot") is True
                ), "Context should be marked as a state snapshot"

        finally:
            cm.actor.act = original_act

    @pytest.mark.asyncio
    @_handle_project
    async def test_context_has_expected_xml_sections(self, initialized_cm):
        """
        Verify that the rendered state snapshot has the expected XML structure.

        This ensures the context is "legible" to an LLM - it has proper structure.
        """
        cm = initialized_cm.cm

        # Create a snapshot via the normal rendering path
        from unity.conversation_manager.domains.brain import build_brain_spec

        brain_spec = build_brain_spec(cm)
        state_content = brain_spec.state_prompt

        # Verify expected sections exist
        assert "<notifications>" in state_content, "Should have notifications section"
        assert (
            "<in_flight_actions>" in state_content
        ), "Should have in_flight_actions section"
        assert (
            "<active_conversations>" in state_content
        ), "Should have active_conversations section"

    @pytest.mark.asyncio
    @_handle_project
    async def test_context_shows_in_flight_action_when_steering(self, initialized_cm):
        """
        Verify that when steering an in-flight action, the context shows that action.

        With incremental context propagation, the interject receives a diff that may
        include <action_updates> (when action state changed) or the action may be
        part of the initial snapshot (sent with the original act() call).

        This tests that the context is fresh and includes relevant action information.
        """
        from tests.conversation_manager.conftest import BOSS
        from unity.conversation_manager.events import SMSReceived, ActorHandleStarted
        from tests.conversation_manager.cm_helpers import filter_events_by_type

        cm_driver = initialized_cm
        cm = cm_driver.cm

        # Step 1: Start an action
        result1 = await cm_driver.step_until_wait(
            SMSReceived(
                contact=BOSS,
                content="Search for all engineering contacts.",
            ),
        )

        # Check if an action was started
        actor_events = filter_events_by_type(result1.output_events, ActorHandleStarted)
        if not actor_events:
            pytest.skip("LLM did not trigger act - cannot test steering context")

        assert len(cm.in_flight_actions) >= 1, "Should have an in-flight action"

        # Step 2: Capture context when steering
        captured_context = []
        handle_id = list(cm.in_flight_actions.keys())[0]
        handle = cm.in_flight_actions[handle_id]["handle"]

        original_interject = handle.interject

        async def capturing_interject(message, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context_cont"))
            return await original_interject(message, **kwargs)

        handle.interject = capturing_interject

        try:
            # Send a message that should trigger interject
            result2 = await cm_driver.step_until_wait(
                SMSReceived(
                    contact=BOSS,
                    content="Also include their phone numbers.",
                ),
            )

            # If interject was called, verify the context shows relevant information
            if captured_context and captured_context[0]:
                content = captured_context[0][0].get("content", "")

                # With incremental diffs, the action may appear in <action_updates>
                # (if its state changed) or the full snapshot (backward compat mode).
                # Either format is acceptable as long as action info is present.
                has_action_info = (
                    "<action_updates>" in content
                    or "<in_flight_actions>" in content
                    or "action id='0'" in content
                )
                assert (
                    has_action_info
                ), f"Should have action information in context. Got: {content[:500]}..."

                # The action should be visible as executing
                assert (
                    "executing" in content.lower() or "search" in content.lower()
                ), "Context should show the in-flight action"

        finally:
            handle.interject = original_interject
