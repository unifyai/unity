"""
tests/conversation_manager/voice/test_fast_brain_context_window.py
==================================================================

Tests for the fast brain rolling context window and history hydration.

Covers:
- trim_fast_brain_context(): system prompt preservation, rolling window
- hydrate_fast_brain_history(): backend query via unify.get_logs, event filtering, rendering
- _render_history_event(): per-event-type rendering and participant filtering

All tests are symbolic — unify.get_logs is mocked to return synthetic log rows.
"""

import json
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from livekit.agents import llm

from unity.conversation_manager.events import (
    ActorHandleStarted,
    ActorNotification,
    ActorResult,
    ActorSessionResponse,
    FastBrainNotification,
    EmailReceived,
    EmailSent,
    InboundPhoneUtterance,
    InboundUnifyMeetUtterance,
    OutboundPhoneUtterance,
    OutboundUnifyMeetUtterance,
    PhoneCallEnded,
    PhoneCallReceived,
    PhoneCallSent,
    PhoneCallStarted,
    SMSReceived,
    SMSSent,
    UnifyMeetEnded,
    UnifyMeetReceived,
    UnifyMeetStarted,
    UnifyMessageReceived,
    UnifyMessageSent,
    VoiceInterrupt,
)
from unity.conversation_manager.medium_scripts.common import (
    hydrate_fast_brain_history,
    trim_fast_brain_context,
    _render_history_event,
)

# =============================================================================
# Helpers
# =============================================================================

BASE_TIME = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
ALICE = {
    "contact_id": 2,
    "first_name": "Alice",
    "surname": "Smith",
    "phone_number": "+15555552222",
}
BOB = {
    "contact_id": 3,
    "first_name": "Bob",
    "surname": "Jones",
    "email_address": "bob@example.com",
}
ASSISTANT_NAME = "Alex"


def _make_log_rows(cm_events):
    """Convert CM events to unify.Log-shaped rows in descending order (newest first)."""
    rows = []
    for ev in cm_events:
        payload = ev.to_dict()["payload"]
        payload.pop("email_id", None)
        rows.append(
            SimpleNamespace(
                entries={
                    "type": "Comms",
                    "payload_cls": ev.__class__.__name__,
                    "payload_json": json.dumps(payload),
                },
            ),
        )
    rows.reverse()
    return rows


def _build_chat_context(
    n_system: int = 1,
    n_conversation: int = 0,
) -> llm.ChatContext:
    """Build a ChatContext with the given number of system and conversation items."""
    ctx = llm.ChatContext()
    for i in range(n_system):
        ctx.add_message(role="system", content=f"system prompt {i}")
    for i in range(n_conversation):
        role = "user" if i % 2 == 0 else "assistant"
        ctx.add_message(role=role, content=f"turn {i}")
    return ctx


# =============================================================================
# trim_fast_brain_context
# =============================================================================


class TestTrimFastBrainContext:

    def test_no_trim_when_within_window(self):
        """Returns all items unchanged when conversation fits in the window."""
        ctx = _build_chat_context(n_system=2, n_conversation=5)
        result = trim_fast_brain_context(ctx.items, window_size=10)
        assert len(result) == 7

    def test_trims_oldest_conversation_items(self):
        """Drops oldest conversation items when exceeding window size."""
        ctx = _build_chat_context(n_system=1, n_conversation=10)
        result = trim_fast_brain_context(ctx.items, window_size=4)
        assert len(result) == 5  # 1 system + 4 conversation
        assert result[0].text_content == "system prompt 0"
        assert result[1].text_content == "turn 6"
        assert result[-1].text_content == "turn 9"

    def test_preserves_all_system_messages(self):
        """All contiguous system messages at the start are preserved."""
        ctx = _build_chat_context(n_system=3, n_conversation=10)
        result = trim_fast_brain_context(ctx.items, window_size=2)
        assert len(result) == 5  # 3 system + 2 conversation
        system_items = [r for r in result if r.role == "system"]
        assert len(system_items) == 3

    def test_empty_context_returns_empty(self):
        """Empty items list returns empty list."""
        result = trim_fast_brain_context([], window_size=10)
        assert result == []

    def test_only_system_messages_no_trim(self):
        """Context with only system messages returns all unchanged."""
        ctx = _build_chat_context(n_system=5, n_conversation=0)
        result = trim_fast_brain_context(ctx.items, window_size=3)
        assert len(result) == 5

    def test_window_size_zero_keeps_only_system(self):
        """Window size of zero keeps only system prompt messages."""
        ctx = _build_chat_context(n_system=2, n_conversation=8)
        result = trim_fast_brain_context(ctx.items, window_size=0)
        assert len(result) == 2
        assert all(r.role == "system" for r in result)

    def test_exact_window_size_no_trim(self):
        """Exactly at window size, no trimming occurs."""
        ctx = _build_chat_context(n_system=1, n_conversation=5)
        result = trim_fast_brain_context(ctx.items, window_size=5)
        assert len(result) == 6  # 1 system + 5 conversation

    def test_returns_new_list(self):
        """Result is a new list, not a reference to the original."""
        ctx = _build_chat_context(n_system=1, n_conversation=3)
        result = trim_fast_brain_context(ctx.items, window_size=10)
        assert result is not ctx.items

    def test_system_followed_by_mixed_roles(self):
        """System block detection stops at first non-system message."""
        ctx = llm.ChatContext()
        ctx.add_message(role="system", content="prompt")
        ctx.add_message(role="user", content="hello")
        ctx.add_message(role="system", content="notification")  # mid-stream system
        ctx.add_message(role="assistant", content="reply")
        ctx.add_message(role="user", content="follow-up")

        result = trim_fast_brain_context(ctx.items, window_size=2)
        # Only the first system message is in the "system block"
        # Conversation = 4 items, trim to last 2
        assert len(result) == 3  # 1 system + 2 conversation
        assert result[0].text_content == "prompt"
        assert result[1].text_content == "reply"
        assert result[2].text_content == "follow-up"


# =============================================================================
# _render_history_event — per-event-type rendering
# =============================================================================


class TestRenderHistoryEvent:

    # -- Utterances --

    def test_inbound_phone_utterance_from_participant(self):
        ev = InboundPhoneUtterance(contact=ALICE, content="Hello there")
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result == "Alice Smith: Hello there"

    def test_inbound_phone_utterance_from_non_participant(self):
        ev = InboundPhoneUtterance(contact=ALICE, content="Hello there")
        result = _render_history_event(ev, {99}, False, ASSISTANT_NAME)
        assert result is None

    def test_outbound_phone_utterance_uses_assistant_name(self):
        ev = OutboundPhoneUtterance(contact=ALICE, content="Hi Alice!")
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result == "Alex: Hi Alice!"

    def test_outbound_utterance_not_filtered_by_participants(self):
        """Outbound utterances are always included (assistant spoke)."""
        ev = OutboundPhoneUtterance(contact=ALICE, content="Hi!")
        result = _render_history_event(ev, set(), False, ASSISTANT_NAME)
        assert result == "Alex: Hi!"

    def test_inbound_meet_utterance(self):
        ev = InboundUnifyMeetUtterance(contact=ALICE, content="Meeting hello")
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result == "Alice Smith: Meeting hello"

    def test_outbound_meet_utterance(self):
        ev = OutboundUnifyMeetUtterance(contact=ALICE, content="Meeting reply")
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result == "Alex: Meeting reply"

    # -- Call lifecycle --

    def test_phone_call_started_marker(self):
        ev = PhoneCallStarted(contact=ALICE, timestamp=BASE_TIME)
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result == "--- Call with Alice Smith ---"

    def test_phone_call_received_marker(self):
        ev = PhoneCallReceived(contact=ALICE, timestamp=BASE_TIME)
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result == "--- Call with Alice Smith ---"

    def test_phone_call_sent_marker(self):
        ev = PhoneCallSent(contact=ALICE, timestamp=BASE_TIME)
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result == "--- Call with Alice Smith ---"

    def test_call_ended_marker(self):
        ev = PhoneCallEnded(contact=ALICE, timestamp=BASE_TIME)
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result == "--- Call ended ---"

    def test_meet_started_marker(self):
        ev = UnifyMeetStarted(contact=ALICE, timestamp=BASE_TIME)
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result == "--- Meeting with Alice Smith ---"

    def test_meet_ended_marker(self):
        ev = UnifyMeetEnded(contact=ALICE, timestamp=BASE_TIME)
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result == "--- Call ended ---"

    def test_call_lifecycle_non_participant_skipped(self):
        ev = PhoneCallStarted(contact=ALICE, timestamp=BASE_TIME)
        result = _render_history_event(ev, {99}, False, ASSISTANT_NAME)
        assert result is None

    # -- SMS --

    def test_sms_received(self):
        ev = SMSReceived(contact=ALICE, content="Hey!")
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result == "[SMS from Alice Smith] Hey!"

    def test_sms_sent(self):
        ev = SMSSent(contact=ALICE, content="Got it")
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result == "[SMS to Alice Smith] Got it"

    def test_sms_non_participant_skipped(self):
        ev = SMSReceived(contact=BOB, content="Hey!")
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result is None

    # -- Email --

    def test_email_received(self):
        ev = EmailReceived(contact=ALICE, subject="Meeting", body="Let's meet")
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result == "[Email from Alice Smith] Meeting"

    def test_email_sent(self):
        ev = EmailSent(contact=ALICE, subject="Re: Meeting", body="Sure!")
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result == "[Email to Alice Smith] Re: Meeting"

    def test_email_no_subject(self):
        ev = EmailReceived(contact=ALICE, subject="", body="Body only")
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result == "[Email from Alice Smith] (no subject)"

    # -- Unify messages --

    def test_unify_message_received(self):
        ev = UnifyMessageReceived(contact=ALICE, content="Console msg")
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result == "[Message from Alice Smith] Console msg"

    def test_unify_message_sent(self):
        ev = UnifyMessageSent(contact=ALICE, content="Reply msg")
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result == "[Message to Alice Smith] Reply msg"

    # -- Boss-only events --

    def test_actor_notification_boss_only(self):
        ev = ActorNotification(handle_id=1, response="Searching…")
        result = _render_history_event(ev, {1}, True, ASSISTANT_NAME)
        assert result == "Action progress: Searching…"

    def test_actor_notification_non_boss_skipped(self):
        ev = ActorNotification(handle_id=1, response="Searching…")
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result is None

    def test_actor_result_success(self):
        ev = ActorResult(handle_id=1, success=True, result="Done!")
        result = _render_history_event(ev, {1}, True, ASSISTANT_NAME)
        assert result == "Action completed successfully: Done!"

    def test_actor_result_failure(self):
        ev = ActorResult(handle_id=1, success=False, error="Timeout")
        result = _render_history_event(ev, {1}, True, ASSISTANT_NAME)
        assert result == "Action failed: Timeout"

    def test_actor_handle_started(self):
        ev = ActorHandleStarted(
            action_name="web_search",
            handle_id=1,
            query="weather today",
        )
        result = _render_history_event(ev, {1}, True, ASSISTANT_NAME)
        assert result == "Action started: web_search — weather today"

    def test_actor_session_response(self):
        ev = ActorSessionResponse(handle_id=1, content="Here are the results")
        result = _render_history_event(ev, {1}, True, ASSISTANT_NAME)
        assert result == "Action update: Here are the results"

    # -- Skipped events --

    def test_call_guidance_skipped(self):
        """FastBrainNotification is handled by the dedicated guidance callback, not history."""
        ev = FastBrainNotification(contact=ALICE, content="Be polite")
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result is None

    def test_voice_interrupt_skipped(self):
        ev = VoiceInterrupt(contact=ALICE)
        result = _render_history_event(ev, {2}, False, ASSISTANT_NAME)
        assert result is None

    def test_unknown_event_skipped(self):
        """Events not in the rendering map return None."""
        ev = UnifyMeetReceived(contact=ALICE, room_name="room-1")
        # UnifyMeetReceived is handled as a meeting marker
        result = _render_history_event(ev, {99}, False, ASSISTANT_NAME)
        assert result is None


# =============================================================================
# hydrate_fast_brain_history — integration with backend mock
# =============================================================================

MOCK_GET_LOGS = "unify.get_logs"
MOCK_SESSION = "unity.conversation_manager.medium_scripts.common.SESSION_DETAILS"


class TestHydrateFastBrainHistory:

    @pytest.fixture(autouse=True)
    def _mock_session(self):
        with patch(MOCK_SESSION) as mock_sd:
            mock_sd.user_context = "test_user"
            mock_sd.assistant_context = "test_assistant"
            yield

    @pytest.mark.asyncio
    async def test_empty_backend_returns_empty_list(self):
        with patch(MOCK_GET_LOGS, return_value=[]):
            result = await hydrate_fast_brain_history({2}, False, ASSISTANT_NAME)
        assert result == []

    @pytest.mark.asyncio
    async def test_phone_call_transcript_hydrated(self):
        """A prior phone call transcript is rendered chronologically."""
        events = [
            PhoneCallStarted(contact=ALICE, timestamp=BASE_TIME),
            InboundPhoneUtterance(
                contact=ALICE,
                content="Hi there",
                timestamp=BASE_TIME + timedelta(seconds=5),
            ),
            OutboundPhoneUtterance(
                contact=ALICE,
                content="Hello Alice!",
                timestamp=BASE_TIME + timedelta(seconds=10),
            ),
            InboundPhoneUtterance(
                contact=ALICE,
                content="How are you?",
                timestamp=BASE_TIME + timedelta(seconds=15),
            ),
            PhoneCallEnded(
                contact=ALICE,
                timestamp=BASE_TIME + timedelta(seconds=20),
            ),
        ]

        with patch(MOCK_GET_LOGS, return_value=_make_log_rows(events)):
            result = await hydrate_fast_brain_history({2}, False, ASSISTANT_NAME)

        assert len(result) == 5
        assert result[0] == "--- Call with Alice Smith ---"
        assert result[1] == "Alice Smith: Hi there"
        assert result[2] == "Alex: Hello Alice!"
        assert result[3] == "Alice Smith: How are you?"
        assert result[4] == "--- Call ended ---"

    @pytest.mark.asyncio
    async def test_sms_exchange_hydrated(self):
        events = [
            SMSReceived(
                contact=ALICE,
                content="Running late",
                timestamp=BASE_TIME,
            ),
            SMSSent(
                contact=ALICE,
                content="No worries, see you soon",
                timestamp=BASE_TIME + timedelta(minutes=1),
            ),
        ]

        with patch(MOCK_GET_LOGS, return_value=_make_log_rows(events)):
            result = await hydrate_fast_brain_history({2}, False, ASSISTANT_NAME)

        assert len(result) == 2
        assert result[0] == "[SMS from Alice Smith] Running late"
        assert result[1] == "[SMS to Alice Smith] No worries, see you soon"

    @pytest.mark.asyncio
    async def test_non_participant_events_filtered_out(self):
        """Events from contacts not on the call are excluded."""
        events = [
            SMSReceived(contact=ALICE, content="from alice", timestamp=BASE_TIME),
            SMSReceived(
                contact=BOB,
                content="from bob",
                timestamp=BASE_TIME + timedelta(seconds=5),
            ),
        ]

        with patch(MOCK_GET_LOGS, return_value=_make_log_rows(events)):
            result = await hydrate_fast_brain_history({2}, False, ASSISTANT_NAME)

        assert len(result) == 1
        assert "alice" in result[0].lower()

    @pytest.mark.asyncio
    async def test_boss_call_includes_actor_events(self):
        """Boss calls include Actor notification events in history."""
        events = [
            ActorHandleStarted(
                action_name="search",
                handle_id=1,
                query="latest news",
                timestamp=BASE_TIME,
            ),
            ActorNotification(
                handle_id=1,
                response="Found 3 results",
                timestamp=BASE_TIME + timedelta(seconds=5),
            ),
        ]

        with patch(MOCK_GET_LOGS, return_value=_make_log_rows(events)):
            result = await hydrate_fast_brain_history({1}, True, ASSISTANT_NAME)

        assert len(result) == 2
        assert "Action started: search" in result[0]
        assert "Action progress: Found 3 results" in result[1]

    @pytest.mark.asyncio
    async def test_non_boss_call_excludes_actor_events(self):
        """Non-boss calls exclude Actor events."""
        events = [
            ActorNotification(
                handle_id=1,
                response="Progress update",
                timestamp=BASE_TIME,
            ),
        ]

        with patch(MOCK_GET_LOGS, return_value=_make_log_rows(events)):
            result = await hydrate_fast_brain_history({2}, False, ASSISTANT_NAME)

        assert result == []

    @pytest.mark.asyncio
    async def test_limit_passed_to_backend(self):
        """The limit parameter is forwarded to unify.get_logs."""
        with patch(MOCK_GET_LOGS, return_value=[]) as mock_get_logs:
            await hydrate_fast_brain_history({2}, False, ASSISTANT_NAME, limit=25)

        mock_get_logs.assert_called_once_with(
            context="test_user/test_assistant/Events",
            filter='type == "Comms"',
            sorting={"timestamp": "descending"},
            limit=25,
        )

    @pytest.mark.asyncio
    async def test_malformed_event_skipped_gracefully(self):
        """Log rows that fail Event.from_dict() conversion are silently skipped."""
        good = SMSReceived(contact=ALICE, content="Good msg", timestamp=BASE_TIME)
        log_rows = _make_log_rows([good])
        bad_row = SimpleNamespace(
            entries={
                "type": "Comms",
                "payload_cls": "NonExistentEvent",
                "payload_json": json.dumps({"broken": True}),
            },
        )
        log_rows.insert(0, bad_row)

        with patch(MOCK_GET_LOGS, return_value=log_rows):
            result = await hydrate_fast_brain_history({2}, False, ASSISTANT_NAME)

        assert len(result) == 1
        assert "Good msg" in result[0]

    @pytest.mark.asyncio
    async def test_mixed_mediums_chronological_order(self):
        """Events across different mediums are rendered in chronological order."""
        events = [
            SMSReceived(
                contact=ALICE,
                content="SMS first",
                timestamp=BASE_TIME,
            ),
            PhoneCallStarted(
                contact=ALICE,
                timestamp=BASE_TIME + timedelta(minutes=5),
            ),
            InboundPhoneUtterance(
                contact=ALICE,
                content="Call utterance",
                timestamp=BASE_TIME + timedelta(minutes=6),
            ),
            PhoneCallEnded(
                contact=ALICE,
                timestamp=BASE_TIME + timedelta(minutes=10),
            ),
            EmailReceived(
                contact=ALICE,
                subject="Follow up",
                body="Details here",
                timestamp=BASE_TIME + timedelta(minutes=30),
            ),
        ]

        with patch(MOCK_GET_LOGS, return_value=_make_log_rows(events)):
            result = await hydrate_fast_brain_history({2}, False, ASSISTANT_NAME)

        assert len(result) == 5
        assert "[SMS from Alice Smith]" in result[0]
        assert "--- Call with Alice Smith ---" in result[1]
        assert "Alice Smith: Call utterance" == result[2]
        assert "--- Call ended ---" == result[3]
        assert "[Email from Alice Smith] Follow up" == result[4]

    @pytest.mark.asyncio
    async def test_outbound_utterances_always_included(self):
        """Outbound utterances are included regardless of participant set."""
        events = [
            OutboundPhoneUtterance(
                contact=ALICE,
                content="Hello!",
                timestamp=BASE_TIME,
            ),
        ]

        with patch(MOCK_GET_LOGS, return_value=_make_log_rows(events)):
            result = await hydrate_fast_brain_history(set(), False, ASSISTANT_NAME)

        assert len(result) == 1
        assert result[0] == "Alex: Hello!"

    @pytest.mark.asyncio
    async def test_works_without_event_bus_initialization(self):
        """Subprocess scenario: history is returned even without unity.init().

        The voice agent subprocess never calls unity.init(), so EVENT_BUS is
        an uninitialized proxy.  Before the direct-backend change, this
        function called EVENT_BUS.search() which raised RuntimeError, causing
        every call to return [].  Now it queries the backend directly via
        unify.get_logs(), so history is available from the first turn.
        """
        events = [
            SMSReceived(contact=ALICE, content="Recent SMS", timestamp=BASE_TIME),
            OutboundPhoneUtterance(
                contact=ALICE,
                content="Got it",
                timestamp=BASE_TIME + timedelta(seconds=30),
            ),
        ]

        with patch(MOCK_GET_LOGS, return_value=_make_log_rows(events)):
            result = await hydrate_fast_brain_history({2}, False, ASSISTANT_NAME)

        assert len(result) == 2
        assert "[SMS from Alice Smith] Recent SMS" == result[0]
        assert "Alex: Got it" == result[1]

    @pytest.mark.asyncio
    async def test_backend_error_returns_empty_gracefully(self):
        """Network or auth errors when querying the backend return empty history."""
        with patch(MOCK_GET_LOGS, side_effect=ConnectionError("unreachable")):
            result = await hydrate_fast_brain_history({2}, False, ASSISTANT_NAME)
        assert result == []
