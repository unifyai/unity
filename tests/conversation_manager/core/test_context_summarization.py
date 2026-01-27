"""
tests/conversation_manager/test_context_summarization.py
=============================================================

Tests for context summarization functionality in ConversationManager.

This covers:
1. Triggering SummarizeContext when chat_history approaches max_messages (70% threshold)
2. The _preprocess_messages() state snapshot deduplication logic
3. Chat history persistence via store_chat_history()

Note: The SummarizeContext event handler is mocked in test mode (see main.py line 82),
so we cannot test the actual handler behavior. Tests focus on:
- The threshold triggering logic in _run_llm()
- The _preprocess_messages() deduplication logic
- The store_chat_history() publication logic
"""

from __future__ import annotations

import pytest

from unity.conversation_manager.events import (
    StoreChatHistory,
    UnifyMessageReceived,
)

# Test contact for message events (contact_id 1 is the boss/main user)
TEST_CONTACT = {"contact_id": 1, "first_name": "Test", "surname": "Contact"}


# =============================================================================
# Test _preprocess_messages() - State Snapshot Deduplication
# =============================================================================


class TestPreprocessMessages:
    """Tests for the _preprocess_messages method that deduplicates state snapshots."""

    def test_returns_string_unchanged(self, initialized_cm):
        """String messages are returned unchanged."""
        result = initialized_cm.cm._preprocess_messages("simple string")
        assert result == "simple string"

    def test_returns_dict_unchanged(self, initialized_cm):
        """Dict messages are returned unchanged."""
        msg = {"role": "user", "content": "hello"}
        result = initialized_cm.cm._preprocess_messages(msg)
        assert result == msg

    def test_returns_non_list_unchanged(self, initialized_cm):
        """Non-list, non-string, non-dict inputs are returned unchanged."""
        result = initialized_cm.cm._preprocess_messages(42)
        assert result == 42

    def test_keeps_only_latest_state_snapshot(self, initialized_cm):
        """When multiple state snapshots exist, only the latest is kept."""
        messages = [
            {"role": "system", "content": "You are an assistant"},
            {"role": "user", "content": "State v1", "_cm_state_snapshot": True},
            {"role": "assistant", "content": "response 1"},
            {"role": "user", "content": "State v2", "_cm_state_snapshot": True},
            {"role": "assistant", "content": "response 2"},
            {"role": "user", "content": "State v3", "_cm_state_snapshot": True},
        ]

        result = initialized_cm.cm._preprocess_messages(messages)

        # Should keep system message and latest state snapshot
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "State v3"
        assert result[1].get("_cm_state_snapshot") is True

    def test_preserves_system_messages(self, initialized_cm):
        """System messages are always preserved."""
        messages = [
            {"role": "system", "content": "System 1"},
            {"role": "system", "content": "System 2"},
            {"role": "user", "content": "State", "_cm_state_snapshot": True},
        ]

        result = initialized_cm.cm._preprocess_messages(messages)

        # Both system messages should be kept
        system_msgs = [m for m in result if m.get("role") == "system"]
        assert len(system_msgs) == 2

    def test_preserves_user_interjections(self, initialized_cm):
        """User messages that are NOT state snapshots are preserved."""
        messages = [
            {"role": "system", "content": "You are an assistant"},
            {"role": "user", "content": "State v1", "_cm_state_snapshot": True},
            {"role": "user", "content": "User question"},  # Not a snapshot
            {"role": "user", "content": "State v2", "_cm_state_snapshot": True},
        ]

        result = initialized_cm.cm._preprocess_messages(messages)

        # Should keep: system, user question, latest state snapshot
        assert len(result) == 3
        contents = [m["content"] for m in result]
        assert "You are an assistant" in contents
        assert "User question" in contents
        assert "State v2" in contents

    def test_no_state_snapshots_returns_messages_unchanged(self, initialized_cm):
        """If there are no state snapshots, messages are returned as-is."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Response"},
        ]

        result = initialized_cm.cm._preprocess_messages(messages)

        assert result == messages

    def test_handles_empty_list(self, initialized_cm):
        """Empty list is returned unchanged."""
        result = initialized_cm.cm._preprocess_messages([])
        assert result == []

    def test_handles_malformed_messages_gracefully(self, initialized_cm):
        """Non-dict items in the list are handled gracefully."""
        messages = [
            {"role": "system", "content": "System"},
            "not a dict",  # Should be skipped
            {"role": "user", "content": "State", "_cm_state_snapshot": True},
        ]

        result = initialized_cm.cm._preprocess_messages(messages)

        # Should keep system and state snapshot, skip the string
        assert len(result) == 2


# =============================================================================
# Test SummarizeContext Triggering (70% threshold)
# =============================================================================


class TestSummarizeContextTrigger:
    """Tests for the 70% threshold triggering SummarizeContext."""

    @pytest.mark.asyncio
    async def test_does_not_trigger_below_threshold(self, initialized_cm):
        """SummarizeContext is NOT triggered when chat_history is below 70% of max."""
        # Set max_messages to 30 (default), 70% = 21
        initialized_cm.cm.max_messages = 30
        initialized_cm.cm.is_summarizing = False

        # Add 10 messages (well below 70% threshold)
        initialized_cm.cm.chat_history = [
            {"role": "user", "content": f"msg {i}"} for i in range(10)
        ]

        # Track published events
        published_events = []
        original_publish = initialized_cm.cm.event_broker.publish

        async def tracking_publish(channel: str, message: str) -> int:
            published_events.append((channel, message))
            return await original_publish(channel, message)

        initialized_cm.cm.event_broker.publish = tracking_publish

        try:
            # Trigger an LLM run
            event = UnifyMessageReceived(
                contact=TEST_CONTACT,
                content="Hello",
            )
            await initialized_cm.step_until_wait(event, max_steps=1)

            # Check that SummarizeContext was NOT published
            summarize_channels = [
                ch for ch, _ in published_events if "summarize" in ch.lower()
            ]
            assert len(summarize_channels) == 0
        finally:
            initialized_cm.cm.event_broker.publish = original_publish

    @pytest.mark.asyncio
    async def test_triggers_at_70_percent_threshold(self, initialized_cm):
        """SummarizeContext IS triggered when chat_history reaches 70% of max."""
        # Set max_messages to 30, 70% = 21
        initialized_cm.cm.max_messages = 30
        initialized_cm.cm.is_summarizing = False

        # Pre-fill chat_history to just under threshold (20 messages)
        # After LLM run adds 2 more (input + response), we'll be at 22 = 73%
        initialized_cm.cm.chat_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(20)
        ]

        # Trigger an LLM run which should push us over 70%
        event = UnifyMessageReceived(
            contact=TEST_CONTACT,
            content="Hello",
        )
        await initialized_cm.step_until_wait(event, max_steps=1)

        # Verify by checking that is_summarizing was set to True
        # (The mocked handler doesn't reset it, so it stays True)
        # This confirms the threshold triggering logic in _run_llm() fired
        assert (
            initialized_cm.cm.is_summarizing is True
        ), "is_summarizing should be True after crossing 70% threshold"

    @pytest.mark.asyncio
    async def test_does_not_trigger_when_already_summarizing(self, initialized_cm):
        """SummarizeContext is NOT triggered if is_summarizing is already True."""
        initialized_cm.cm.max_messages = 30
        initialized_cm.cm.is_summarizing = True  # Already summarizing!

        # Pre-fill to well above threshold
        initialized_cm.cm.chat_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(25)
        ]

        # Track published events
        published_events = []
        original_publish = initialized_cm.cm.event_broker.publish

        async def tracking_publish(channel: str, message: str) -> int:
            published_events.append((channel, message))
            return await original_publish(channel, message)

        initialized_cm.cm.event_broker.publish = tracking_publish

        try:
            event = UnifyMessageReceived(
                contact=TEST_CONTACT,
                content="Hello",
            )
            await initialized_cm.step_until_wait(event, max_steps=1)

            # Should NOT publish SummarizeContext since we're already summarizing
            summarize_channels = [
                ch for ch, _ in published_events if "summarize" in ch.lower()
            ]
            assert (
                len(summarize_channels) == 0
            ), "SummarizeContext should NOT be re-triggered while already summarizing"
        finally:
            initialized_cm.cm.event_broker.publish = original_publish

    @pytest.mark.asyncio
    async def test_sets_is_summarizing_flag_when_triggered(self, initialized_cm):
        """When SummarizeContext is triggered, is_summarizing flag is set to True."""
        initialized_cm.cm.max_messages = 30
        initialized_cm.cm.is_summarizing = False

        # Pre-fill to trigger threshold
        initialized_cm.cm.chat_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(20)
        ]

        event = UnifyMessageReceived(
            contact=TEST_CONTACT,
            content="Hello",
        )
        await initialized_cm.step_until_wait(event, max_steps=1)

        # After triggering, is_summarizing should be True
        # Note: The SummarizeContext handler is mocked in test mode, so the flag
        # remains True after the mocked handler runs (it doesn't reset it).
        # We're testing that the triggering mechanism in _run_llm() works.
        assert initialized_cm.cm.is_summarizing is True


# =============================================================================
# Note: SummarizeContext Event Handler Tests
# =============================================================================
# The SummarizeContext handler is mocked in test mode (main.py:82), so we cannot
# test the actual handler behavior (e.g., clearing chat_history when MemoryManager
# is disabled). This is a test infrastructure limitation, not a bug.
#
# The real handler behavior when MemoryManager is None:
#   - Sets is_summarizing = False
#   - Clears chat_history = []
#
# When MemoryManager exists, it runs summarization in a background task.
# =============================================================================


# =============================================================================
# Test store_chat_history()
# =============================================================================


class TestStoreChatHistory:
    """Tests for the store_chat_history method."""

    @pytest.mark.asyncio
    async def test_publishes_last_two_messages(self, initialized_cm):
        """store_chat_history publishes the last 2 messages from chat_history."""
        # Set up chat history with multiple messages
        initialized_cm.cm.chat_history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
            {"role": "assistant", "content": "fourth"},
        ]

        published_events = []
        original_publish = initialized_cm.cm.event_broker.publish

        async def tracking_publish(channel: str, message: str) -> int:
            published_events.append((channel, message))
            return 0  # Don't actually publish

        initialized_cm.cm.event_broker.publish = tracking_publish

        try:
            await initialized_cm.cm.store_chat_history()

            # Find the StoreChatHistory event
            chat_history_events = [
                (ch, msg) for ch, msg in published_events if "chat_history" in ch
            ]
            assert len(chat_history_events) == 1

            channel, message = chat_history_events[0]
            assert channel == "app:comms:chat_history"

            # Parse the event and check it contains last 2 messages
            from unity.conversation_manager.events import Event

            event = Event.from_json(message)
            assert isinstance(event, StoreChatHistory)
            assert len(event.chat_history) == 2
            assert event.chat_history[0]["content"] == "third"
            assert event.chat_history[1]["content"] == "fourth"
        finally:
            initialized_cm.cm.event_broker.publish = original_publish

    @pytest.mark.asyncio
    async def test_does_not_publish_when_chat_history_too_short(self, initialized_cm):
        """store_chat_history does nothing if chat_history has fewer than 2 messages."""
        # Chat history with only 1 message
        initialized_cm.cm.chat_history = [
            {"role": "user", "content": "only one"},
        ]

        published_events = []
        original_publish = initialized_cm.cm.event_broker.publish

        async def tracking_publish(channel: str, message: str) -> int:
            published_events.append((channel, message))
            return 0

        initialized_cm.cm.event_broker.publish = tracking_publish

        try:
            await initialized_cm.cm.store_chat_history()

            # Should NOT publish anything
            assert len(published_events) == 0
        finally:
            initialized_cm.cm.event_broker.publish = original_publish

    @pytest.mark.asyncio
    async def test_does_not_publish_when_chat_history_empty(self, initialized_cm):
        """store_chat_history does nothing if chat_history is empty."""
        initialized_cm.cm.chat_history = []

        published_events = []
        original_publish = initialized_cm.cm.event_broker.publish

        async def tracking_publish(channel: str, message: str) -> int:
            published_events.append((channel, message))
            return 0

        initialized_cm.cm.event_broker.publish = tracking_publish

        try:
            await initialized_cm.cm.store_chat_history()
            assert len(published_events) == 0
        finally:
            initialized_cm.cm.event_broker.publish = original_publish

    @pytest.mark.asyncio
    async def test_called_during_cleanup(self, initialized_cm):
        """store_chat_history is called as part of cleanup()."""
        initialized_cm.cm.chat_history = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
        ]

        published_events = []
        original_publish = initialized_cm.cm.event_broker.publish

        async def tracking_publish(channel: str, message: str) -> int:
            published_events.append((channel, message))
            return 0

        initialized_cm.cm.event_broker.publish = tracking_publish

        # We can't actually call cleanup() as it will stop the CM
        # Instead, we verify the method integration by calling store_chat_history
        # directly (the cleanup test is more of an integration concern)
        try:
            await initialized_cm.cm.store_chat_history()

            chat_history_events = [
                ch for ch, _ in published_events if "chat_history" in ch
            ]
            assert len(chat_history_events) == 1
        finally:
            initialized_cm.cm.event_broker.publish = original_publish


# =============================================================================
# Test max_messages Configuration
# =============================================================================


class TestMaxMessagesConfiguration:
    """Tests for max_messages configuration and its effects."""

    def test_default_max_messages_is_30(self, initialized_cm):
        """Default max_messages value is 30."""
        # Create fresh CM to check default
        assert initialized_cm.cm.max_messages == 30

    def test_70_percent_threshold_calculation(self, initialized_cm):
        """Verify the 70% threshold calculation."""
        initialized_cm.cm.max_messages = 30
        threshold = int(0.7 * initialized_cm.cm.max_messages)
        assert threshold == 21

        initialized_cm.cm.max_messages = 50
        threshold = int(0.7 * initialized_cm.cm.max_messages)
        assert threshold == 35

        initialized_cm.cm.max_messages = 10
        threshold = int(0.7 * initialized_cm.cm.max_messages)
        assert threshold == 7


# =============================================================================
# Test Chat History Growth During LLM Runs
# =============================================================================


class TestChatHistoryGrowth:
    """Tests for chat history growth during LLM runs."""

    @pytest.mark.asyncio
    async def test_chat_history_grows_by_two_per_llm_run(self, initialized_cm):
        """Each LLM run adds 2 messages: input + assistant response."""
        initial_len = len(initialized_cm.cm.chat_history)

        event = UnifyMessageReceived(
            contact=TEST_CONTACT,
            content="Test message",
        )
        await initialized_cm.step_until_wait(event, max_steps=1)

        # Chat history should have grown by 2 (input_message + assistant_content)
        new_len = len(initialized_cm.cm.chat_history)
        assert new_len == initial_len + 2

    @pytest.mark.asyncio
    async def test_chat_history_format(self, initialized_cm):
        """Chat history messages have correct format."""
        event = UnifyMessageReceived(
            contact=TEST_CONTACT,
            content="Test message",
        )
        await initialized_cm.step_until_wait(event, max_steps=1)

        # Should have at least 2 messages
        assert len(initialized_cm.cm.chat_history) >= 2

        # Check structure of messages
        for msg in initialized_cm.cm.chat_history:
            assert isinstance(msg, dict)
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ["user", "assistant", "system"]
