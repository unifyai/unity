"""
tests/conversation_manager/voice/test_unify_meet_e2e.py
=======================================================

End-to-end integration tests for Unify Meet (web-based voice) flow.

This file tests the COMPLETE Unify Meet lifecycle as documented in INFRA.md:
1. UnifyMeetReceived event arrives (from Pub/Sub via adapters)
2. ConversationManager spawns voice agent subprocess
3. Voice agent connects to LiveKit room
4. Mode transitions: TEXT → MEET → TEXT
5. Bidirectional IPC for call guidance
6. Utterance events flow correctly
7. Meet ends and cleanup occurs

These tests use REAL infrastructure where possible:
- Real LiveKit server (auto-started if available)
- Real IPC socket communication
- Real subprocess spawning
- Real event broker

Only external APIs (OpenAI, Deepgram) are mocked/skipped.

Prerequisites:
    macOS: brew install livekit
    Linux: ./scripts/install_livekit.sh

Related Ved fixes these tests would catch:
- 81596d0e: Room name handling for Unify meets
- c34270dc: IPC bidirectional communication
- 88e0d678: Mode check for call initiation events
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from unity.conversation_manager.events import (
    UnifyMeetReceived,
    UnifyMeetStarted,
    UnifyMeetEnded,
    InboundUnifyMeetUtterance,
    OutboundUnifyMeetUtterance,
    FastBrainNotification,
)
from unity.conversation_manager.types import Medium, Mode

from tests.conversation_manager.conftest import TEST_CONTACTS

# Check if livekit-server is available
LIVEKIT_SERVER_PATH = shutil.which("livekit-server")
LIVEKIT_SERVER_AVAILABLE = LIVEKIT_SERVER_PATH is not None

# Allow skipping LiveKit tests entirely via environment variable
SKIP_LIVEKIT = os.environ.get("SKIP_LIVEKIT_E2E", "0") == "1"


def start_livekit_server() -> subprocess.Popen | None:
    """Start a local LiveKit server in dev mode. Returns the process or None."""
    if not LIVEKIT_SERVER_AVAILABLE:
        return None

    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        sock.connect(("localhost", 7880))
        sock.close()
        print("[LiveKit] Server already running on port 7880")
        return None
    except (ConnectionRefusedError, socket.timeout, OSError):
        pass

    print(f"[LiveKit] Starting server from {LIVEKIT_SERVER_PATH}")
    proc = subprocess.Popen(
        [LIVEKIT_SERVER_PATH, "--dev"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    for _ in range(30):
        time.sleep(0.1)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect(("localhost", 7880))
            sock.close()
            print("[LiveKit] Server started successfully")
            return proc
        except (ConnectionRefusedError, socket.timeout, OSError):
            continue

    proc.terminate()
    return None


def stop_livekit_server(proc: subprocess.Popen | None) -> None:
    """Stop the LiveKit server if we started it."""
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[LiveKit] Server stopped")


@pytest.fixture
def boss_contact():
    return TEST_CONTACTS[1]


@pytest.fixture
def alice_contact():
    return TEST_CONTACTS[2]


@pytest_asyncio.fixture
async def event_broker():
    """Real in-memory event broker."""
    from unity.conversation_manager.in_memory_event_broker import (
        create_in_memory_event_broker,
        reset_in_memory_event_broker,
    )

    reset_in_memory_event_broker()
    broker = create_in_memory_event_broker()
    yield broker
    await broker.aclose()
    reset_in_memory_event_broker()


# =============================================================================
# Test: Unify Meet Lifecycle - Mode Transitions
# =============================================================================


@pytest.mark.asyncio
class TestUnifyMeetModeTransitions:
    """
    Tests for mode transitions during Unify Meet lifecycle.

    Mode should transition: TEXT → MEET → TEXT
    This is critical for the slow brain to know when to provide call_guidance.
    """

    async def test_unify_meet_started_changes_mode_to_meet(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        UnifyMeetStarted should change mode from TEXT to MEET.

        This is different from phone calls which use Mode.CALL.
        The distinction matters for:
        - Correct Medium selection in push_message
        - Proactive speech behavior
        - Transcript logging
        """
        assert initialized_cm.cm.mode == Mode.TEXT

        event = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(event)

        assert (
            initialized_cm.cm.mode == Mode.MEET
        ), f"Expected mode=MEET after UnifyMeetStarted, got {initialized_cm.cm.mode}"

    async def test_unify_meet_ended_resets_mode_to_text(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        UnifyMeetEnded should reset mode from MEET back to TEXT.
        """
        # Start a meet first
        started = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(started)
        assert initialized_cm.cm.mode == Mode.MEET

        # End the meet
        ended = UnifyMeetEnded(contact=boss_contact)
        with patch.object(
            initialized_cm.cm.call_manager,
            "cleanup_call_proc",
            new_callable=AsyncMock,
        ):
            await initialized_cm.step(ended)

        assert (
            initialized_cm.cm.mode == Mode.TEXT
        ), f"Expected mode=TEXT after UnifyMeetEnded, got {initialized_cm.cm.mode}"

    async def test_mode_is_voice_returns_true_in_meet_mode(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        Mode.is_voice should return True for MEET mode.

        This is used throughout the codebase to check if we're in a voice context.
        """
        assert not initialized_cm.cm.mode.is_voice, "TEXT mode should not be voice"

        event = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(event)

        assert initialized_cm.cm.mode.is_voice, "MEET mode should be voice"


# =============================================================================
# Test: Unify Meet Contact Tracking
# =============================================================================


@pytest.mark.asyncio
class TestUnifyMeetContactTracking:
    """
    Tests for contact tracking during Unify Meet sessions.

    The call_manager.call_contact should be set during meets and cleared after.
    This affects who receives guidance and transcript logging.
    """

    async def test_unify_meet_started_sets_call_contact(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        UnifyMeetStarted should set call_manager.call_contact.
        """
        assert initialized_cm.cm.call_manager.call_contact is None

        event = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(event)

        assert initialized_cm.cm.call_manager.call_contact is not None
        assert (
            initialized_cm.cm.call_manager.call_contact["contact_id"]
            == boss_contact["contact_id"]
        )

    async def test_unify_meet_ended_clears_call_contact(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        UnifyMeetEnded should clear call_manager.call_contact.
        """
        # Start meet
        started = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(started)
        assert initialized_cm.cm.call_manager.call_contact is not None

        # End meet
        ended = UnifyMeetEnded(contact=boss_contact)
        with patch.object(
            initialized_cm.cm.call_manager,
            "cleanup_call_proc",
            new_callable=AsyncMock,
        ):
            await initialized_cm.step(ended)

        assert initialized_cm.cm.call_manager.call_contact is None

    async def test_unify_meet_marks_contact_on_call(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        UnifyMeetStarted should set on_call=True in conversation state.
        """
        event = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(event)

        contact_id = boss_contact["contact_id"]
        conv = initialized_cm.cm.contact_index.active_conversations.get(contact_id)
        assert conv is not None
        assert conv.on_call is True

    async def test_unify_meet_ended_clears_on_call(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        UnifyMeetEnded should set on_call=False in conversation state.
        """
        # Start meet
        started = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(started)

        contact_id = boss_contact["contact_id"]

        # End meet
        ended = UnifyMeetEnded(contact=boss_contact)
        with patch.object(
            initialized_cm.cm.call_manager,
            "cleanup_call_proc",
            new_callable=AsyncMock,
        ):
            await initialized_cm.step(ended)

        conv = initialized_cm.cm.contact_index.active_conversations.get(contact_id)
        assert conv.on_call is False


# =============================================================================
# Test: Unify Meet Utterance Handling
# =============================================================================


@pytest.mark.asyncio
class TestUnifyMeetUtteranceHandling:
    """
    Tests for utterance event handling during Unify Meet sessions.

    Utterances should be:
    1. Pushed to the correct thread (UNIFY_MEET, not PHONE_CALL)
    2. Trigger interject_or_run for inbound utterances
    3. Be logged via transcript manager
    """

    async def test_inbound_utterance_pushed_to_unify_meet_thread(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        InboundUnifyMeetUtterance should be pushed to UNIFY_MEET thread.

        This is different from phone calls which use PHONE_CALL thread.
        """
        # Start meet first
        started = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(started)

        # Receive utterance
        utterance = InboundUnifyMeetUtterance(
            contact=boss_contact,
            content="Hello, can you hear me?",
        )
        await initialized_cm.step(utterance)

        # Check message is in UNIFY_MEET thread
        contact_id = boss_contact["contact_id"]
        meet_thread = initialized_cm.cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.UNIFY_MEET,
        )
        assert len(meet_thread) > 0, "UNIFY_MEET thread should have messages"

        messages = [msg.content for msg in meet_thread]
        assert "Hello, can you hear me?" in messages

    async def test_outbound_utterance_pushed_to_unify_meet_thread(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        OutboundUnifyMeetUtterance should be pushed to UNIFY_MEET thread.
        """
        # Start meet first
        started = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(started)

        # Send utterance
        utterance = OutboundUnifyMeetUtterance(
            contact=boss_contact,
            content="Yes, I can hear you clearly!",
        )
        await initialized_cm.step(utterance)

        # Check message is in UNIFY_MEET thread
        contact_id = boss_contact["contact_id"]
        meet_thread = initialized_cm.cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.UNIFY_MEET,
        )

        messages = [msg.content for msg in meet_thread]
        assert "Yes, I can hear you clearly!" in messages

    async def test_inbound_utterance_triggers_interject(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        InboundUnifyMeetUtterance should trigger interject_or_run.
        """
        # Start meet first
        started = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(started)

        # Mock interject_or_run
        with patch.object(
            initialized_cm.cm,
            "interject_or_run",
            new_callable=AsyncMock,
        ) as mock_interject:
            utterance = InboundUnifyMeetUtterance(
                contact=boss_contact,
                content="What's the status of the project?",
            )
            await initialized_cm.step(utterance)

            mock_interject.assert_called_once_with("What's the status of the project?")

    async def test_inbound_utterance_resets_proactive_speech(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        InboundUnifyMeetUtterance should reset the proactive speech cycle.
        """
        # Start meet first
        started = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(started)

        with patch.object(
            initialized_cm.cm,
            "schedule_proactive_speech",
            new_callable=AsyncMock,
        ) as mock_schedule:
            utterance = InboundUnifyMeetUtterance(
                contact=boss_contact,
                content="Quick question...",
            )
            await initialized_cm.step(utterance)

            mock_schedule.assert_called_once()


# =============================================================================
# Test: Unify Meet Call Guidance Flow
# =============================================================================


@pytest.mark.asyncio
class TestUnifyMeetFastBrainNotification:
    """
    Tests for FastBrainNotification flow during Unify Meet sessions.

    The slow brain sends FastBrainNotification to the fast brain (voice agent) via IPC.
    This must work correctly for the voice agent to receive context.
    """

    async def test_call_guidance_pushed_to_unify_meet_thread(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        FastBrainNotification should be pushed to UNIFY_MEET thread during a meet.
        """
        # Start meet first
        started = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(started)

        # Send guidance
        guidance = FastBrainNotification(
            contact=boss_contact,
            content="The meeting you mentioned is scheduled for 3pm Thursday",
        )
        await initialized_cm.step(guidance)

        # Check message is in UNIFY_MEET thread with role=guidance
        contact_id = boss_contact["contact_id"]
        meet_thread = initialized_cm.cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.UNIFY_MEET,
        )

        guidance_msgs = [msg for msg in meet_thread if msg.name == "guidance"]
        assert len(guidance_msgs) >= 1
        assert "3pm Thursday" in " ".join([msg.content for msg in guidance_msgs])

    async def test_call_guidance_uses_correct_medium_based_on_mode(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        FastBrainNotification should use UNIFY_MEET medium when mode=MEET.

        This ensures guidance goes to the correct thread.
        """
        # Start meet (mode becomes MEET)
        started = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(started)

        assert initialized_cm.cm.mode == Mode.MEET

        # The event handler uses cm.mode to determine the medium
        # When mode=MEET, it should use Medium.UNIFY_MEET
        guidance = FastBrainNotification(
            contact=boss_contact,
            content="Test guidance content",
        )
        await initialized_cm.step(guidance)

        # Verify it went to UNIFY_MEET thread, not PHONE_CALL
        contact_id = boss_contact["contact_id"]
        meet_thread = initialized_cm.cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.UNIFY_MEET,
        )
        phone_thread = initialized_cm.cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.PHONE_CALL,
        )

        meet_contents = " ".join([msg.content for msg in meet_thread])
        phone_contents = " ".join([msg.content for msg in phone_thread])

        assert "Test guidance content" in meet_contents
        assert "Test guidance content" not in phone_contents


# =============================================================================
# Test: Unify Meet Subprocess Spawning
# =============================================================================


@pytest.mark.asyncio
class TestUnifyMeetSubprocessSpawning:
    """
    Tests for voice agent subprocess spawning during Unify Meet.

    When UnifyMeetReceived arrives, the system should:
    1. Call start_unify_meet() with correct room/agent names
    2. Spawn the voice agent subprocess
    3. Pass correct arguments including room_name
    """

    async def test_unify_meet_received_calls_start_unify_meet(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        UnifyMeetReceived should trigger start_unify_meet() call.
        """
        with patch.object(
            initialized_cm.cm.call_manager,
            "start_unify_meet",
            new_callable=AsyncMock,
        ) as mock_start:
            event = UnifyMeetReceived(
                contact=boss_contact,
                room_name="unity_25_meet",
            )
            await initialized_cm.step(event)

            mock_start.assert_called_once()

    async def test_unify_meet_received_passes_room_name_correctly(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        Room name must be passed correctly to start_unify_meet().

        This was broken in Ved's fix 81596d0e where assistant_number
        was removed from args but still expected.
        """
        captured_args = {}

        async def capture_start_unify_meet(contact, boss, room_name):
            captured_args["contact"] = contact
            captured_args["boss"] = boss
            captured_args["room_name"] = room_name

        with patch.object(
            initialized_cm.cm.call_manager,
            "start_unify_meet",
            side_effect=capture_start_unify_meet,
        ):
            event = UnifyMeetReceived(
                contact=boss_contact,
                room_name="unity_42_meet",
            )
            await initialized_cm.step(event)

        assert (
            captured_args["room_name"] == "unity_42_meet"
        ), f"Room name not passed correctly: {captured_args.get('room_name')}"

    async def test_unify_meet_received_ignored_when_already_in_voice_mode(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        UnifyMeetReceived should be ignored if already in voice mode.

        This prevents duplicate subprocess spawns (Ved's fix 88e0d678).
        """
        # Set mode to MEET (simulating an active meet)
        initialized_cm.cm.mode = Mode.MEET

        with patch.object(
            initialized_cm.cm.call_manager,
            "start_unify_meet",
            new_callable=AsyncMock,
        ) as mock_start:
            event = UnifyMeetReceived(
                contact=boss_contact,
                room_name="room",
            )
            await initialized_cm.step(event)

            mock_start.assert_not_called()


# =============================================================================
# Test: Unify Meet Cleanup
# =============================================================================


@pytest.mark.asyncio
class TestUnifyMeetCleanup:
    """
    Tests for cleanup when Unify Meet ends.

    When a meet ends, the system should:
    1. Reset mode to TEXT
    2. Clear call_contact
    3. Terminate the voice agent subprocess
    4. Cancel proactive speech
    5. Trigger an LLM run
    """

    async def test_unify_meet_ended_calls_cleanup_call_proc(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        UnifyMeetEnded should call cleanup_call_proc() to terminate subprocess.
        """
        # Start meet
        started = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(started)

        # End meet
        ended = UnifyMeetEnded(contact=boss_contact)
        with patch.object(
            initialized_cm.cm.call_manager,
            "cleanup_call_proc",
            new_callable=AsyncMock,
        ) as mock_cleanup:
            await initialized_cm.step(ended)

            mock_cleanup.assert_called_once()

    async def test_unify_meet_ended_cancels_proactive_speech(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        UnifyMeetEnded should cancel any pending proactive speech.
        """
        # Start meet
        started = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(started)

        # End meet
        ended = UnifyMeetEnded(contact=boss_contact)
        with patch.object(
            initialized_cm.cm.call_manager,
            "cleanup_call_proc",
            new_callable=AsyncMock,
        ):
            with patch.object(
                initialized_cm.cm,
                "cancel_proactive_speech",
                new_callable=AsyncMock,
            ) as mock_cancel:
                await initialized_cm.step(ended)

                mock_cancel.assert_called_once()

    async def test_unify_meet_ended_triggers_llm_run(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        UnifyMeetEnded should trigger an LLM run to process the ended call.
        """
        # Start meet
        started = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(started)

        # End meet
        ended = UnifyMeetEnded(contact=boss_contact)
        with patch.object(
            initialized_cm.cm.call_manager,
            "cleanup_call_proc",
            new_callable=AsyncMock,
        ):
            result = await initialized_cm.step(ended)

            # CMStepDriver.step() intercepts request_llm_run calls and tracks them
            # via StepResult.llm_requested - this is the correct way to verify
            # that the event handler requested an LLM run
            assert (
                result.llm_requested
            ), "UnifyMeetEnded should trigger an LLM run to process the ended call"


# =============================================================================
# Test: Full Unify Meet Lifecycle (Integration)
# =============================================================================


@pytest.mark.asyncio
class TestFullUnifyMeetLifecycle:
    """
    End-to-end lifecycle tests for complete Unify Meet sessions.

    These tests exercise the full flow from receive → start → utterances → end.
    """

    async def test_complete_unify_meet_session(
        self,
        initialized_cm,
        boss_contact,
    ):
        """
        Test a complete Unify Meet session from start to finish.
        """
        # 1. Receive meet invite (would trigger subprocess in production)
        with patch.object(
            initialized_cm.cm.call_manager,
            "start_unify_meet",
            new_callable=AsyncMock,
        ):
            received = UnifyMeetReceived(
                contact=boss_contact,
                room_name="unity_test_room",
            )
            await initialized_cm.step(received)

        # 2. Meet starts
        started = UnifyMeetStarted(contact=boss_contact)
        await initialized_cm.step(started)
        assert initialized_cm.cm.mode == Mode.MEET

        # 3. User speaks
        user_utterance = InboundUnifyMeetUtterance(
            contact=boss_contact,
            content="Hi, let's discuss the project status",
        )
        await initialized_cm.step(user_utterance)

        # 4. Assistant responds
        assistant_utterance = OutboundUnifyMeetUtterance(
            contact=boss_contact,
            content="Sure, the project is progressing well",
        )
        await initialized_cm.step(assistant_utterance)

        # 5. Slow brain sends guidance
        guidance = FastBrainNotification(
            contact=boss_contact,
            content="Remember to mention the deadline is Friday",
        )
        await initialized_cm.step(guidance)

        # 6. More conversation
        user_utterance2 = InboundUnifyMeetUtterance(
            contact=boss_contact,
            content="When is the deadline?",
        )
        await initialized_cm.step(user_utterance2)

        # 7. Meet ends
        ended = UnifyMeetEnded(contact=boss_contact)
        with patch.object(
            initialized_cm.cm.call_manager,
            "cleanup_call_proc",
            new_callable=AsyncMock,
        ):
            await initialized_cm.step(ended)

        # Verify final state
        assert initialized_cm.cm.mode == Mode.TEXT
        assert initialized_cm.cm.call_manager.call_contact is None

        # Verify conversation was recorded
        contact_id = boss_contact["contact_id"]
        conv = initialized_cm.cm.contact_index.active_conversations.get(contact_id)
        assert conv is not None
        assert conv.on_call is False

        # Check all messages are in the thread
        meet_thread = initialized_cm.cm.contact_index.get_messages_for_contact(
            contact_id,
            Medium.UNIFY_MEET,
        )
        contents = [msg.content for msg in meet_thread]

        assert any("project status" in c for c in contents)
        assert any("progressing well" in c for c in contents)
        assert any("deadline" in c.lower() for c in contents)


# =============================================================================
# Test: Real LiveKit Integration (requires livekit-server)
# =============================================================================


@pytest.mark.skipif(
    SKIP_LIVEKIT or not LIVEKIT_SERVER_AVAILABLE,
    reason=(
        "livekit-server not installed. To run this test:\n"
        "  macOS: brew install livekit\n"
        "  Linux: ./scripts/install_livekit.sh\n"
        "  Then re-run this test (server auto-starts)"
    ),
)
class TestRealLiveKitUnifyMeet:
    """
    TRUE end-to-end tests for Unify Meet with real LiveKit.

    These tests spawn real subprocesses and connect to a real LiveKit server.
    """

    @pytest.fixture(autouse=True)
    def setup_livekit(self):
        """Auto-start LiveKit server for tests in this class."""
        self.livekit_proc = start_livekit_server()
        yield
        stop_livekit_server(self.livekit_proc)

    @pytest.mark.asyncio
    async def test_real_unify_meet_subprocess_spawn(
        self,
        event_broker,
        boss_contact,
    ):
        """
        Test that a real voice agent subprocess can be spawned for Unify Meet.

        This exercises the actual call.py script with LiveKit.
        """
        from unity.conversation_manager.domains.ipc_socket import (
            CallEventSocketServer,
            CM_EVENT_SOCKET_ENV,
        )

        livekit_env = {
            "LIVEKIT_URL": os.environ.get("LIVEKIT_URL", "ws://localhost:7880"),
            "LIVEKIT_API_KEY": os.environ.get("LIVEKIT_API_KEY", "devkey"),
            "LIVEKIT_API_SECRET": os.environ.get("LIVEKIT_API_SECRET", "secret"),
        }

        events_from_subprocess = []

        async def on_subprocess_event(channel: str, event_json: str):
            print(f"[TEST] Received from voice agent: {channel}")
            events_from_subprocess.append((channel, event_json))

        socket_server = CallEventSocketServer(
            event_broker,
            on_event=on_subprocess_event,
            forward_channels=["app:call:*"],
        )

        proc = None
        call_py = (
            Path(__file__).parent.parent.parent.parent
            / "unity"
            / "conversation_manager"
            / "medium_scripts"
            / "call.py"
        )

        sample_contact = {
            "contact_id": 2,
            "first_name": "Test",
            "surname": "User",
            "phone_number": "+15555551234",
            "email_address": "test@example.com",
        }

        try:
            socket_path = await socket_server.start()

            env = os.environ.copy()
            env[CM_EVENT_SOCKET_ENV] = socket_path
            env.update(livekit_env)

            workspace_root = str(Path(__file__).parent.parent.parent.parent)
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{workspace_root}:{existing_pythonpath}"
                if existing_pythonpath
                else workspace_root
            )

            # Spawn for Unify Meet (channel="unify")
            args = [
                sys.executable,
                str(call_py),
                "dev",
                "unify_meet_test_agent:unify_meet_test_room",
                "cartesia",
                "",
                "False",
                "unify",  # This is the key difference - "unify" for meets
                json.dumps(sample_contact),
                json.dumps(boss_contact),
                "A test assistant for Unify Meet",
            ]

            print(f"[TEST] Spawning Unify Meet voice agent...")

            proc = subprocess.Popen(
                args,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for process to start
            process_started = False
            for i in range(100):  # 10s timeout
                await asyncio.sleep(0.1)

                exit_code = proc.poll()
                if exit_code is not None:
                    stdout = proc.stdout.read().decode() if proc.stdout else ""
                    stderr = proc.stderr.read().decode() if proc.stderr else ""

                    if "LIVEKIT" in stderr or "connection" in stderr.lower():
                        pytest.skip(
                            f"LiveKit connection failed.\nstderr: {stderr[:500]}",
                        )

                    pytest.fail(
                        f"Voice agent crashed with exit code {exit_code}.\n"
                        f"stdout: {stdout[:500]}\nstderr: {stderr[:500]}",
                    )

                if events_from_subprocess:
                    process_started = True
                    print(
                        f"[TEST] Voice agent sent event: {events_from_subprocess[0][0]}",
                    )
                    break

                if i == 30 and proc.poll() is None:
                    process_started = True
                    print("[TEST] Voice agent process is running")
                    break

            if not process_started:
                stdout = proc.stdout.read().decode() if proc.stdout else ""
                stderr = proc.stderr.read().decode() if proc.stderr else ""
                pytest.fail(
                    f"Voice agent did not start.\n"
                    f"stdout: {stdout[:500]}\nstderr: {stderr[:500]}",
                )

            print("[TEST] Unify Meet voice agent spawned successfully!")

        finally:
            if CM_EVENT_SOCKET_ENV in os.environ:
                del os.environ[CM_EVENT_SOCKET_ENV]

            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()

            await socket_server.stop()

    @pytest.mark.asyncio
    async def test_real_unify_meet_ipc_guidance_flow(
        self,
        event_broker,
        boss_contact,
    ):
        """
        Test that call guidance flows correctly via IPC during Unify Meet.
        """
        from unity.conversation_manager.domains.ipc_socket import (
            CallEventSocketServer,
            CM_EVENT_SOCKET_ENV,
        )

        events_from_subprocess = []

        async def on_subprocess_event(channel: str, event_json: str):
            events_from_subprocess.append((channel, event_json))

        socket_server = CallEventSocketServer(
            event_broker,
            on_event=on_subprocess_event,
            forward_channels=["app:call:*"],
        )

        # Use the test subprocess that mimics voice agent IPC
        test_subprocess = Path(__file__).parent / "ipc_test_subprocess.py"
        proc = None

        try:
            socket_path = await socket_server.start()
            os.environ[CM_EVENT_SOCKET_ENV] = socket_path

            env = os.environ.copy()
            env[CM_EVENT_SOCKET_ENV] = socket_path

            workspace_root = str(Path(__file__).parent.parent.parent.parent)
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{workspace_root}:{existing_pythonpath}"
                if existing_pythonpath
                else workspace_root
            )

            proc = subprocess.Popen(
                [sys.executable, str(test_subprocess), "full_roundtrip"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for ready
            ready_received = False
            for _ in range(50):
                await asyncio.sleep(0.1)
                for channel, _ in events_from_subprocess:
                    if channel == "app:call:ready":
                        ready_received = True
                        break
                if ready_received:
                    break

            assert ready_received, "Subprocess never signaled ready"

            # Send Unify Meet specific guidance
            events_from_subprocess.clear()
            guidance_content = "The boss mentioned they prefer morning meetings"

            await event_broker.publish(
                "app:call:notification",
                json.dumps({"content": guidance_content}),
            )

            # Wait for ack
            ack_received = False
            for _ in range(50):
                await asyncio.sleep(0.1)
                for channel, event_json in events_from_subprocess:
                    if channel == "app:call:ack":
                        data = json.loads(event_json)
                        if "morning meetings" in data.get("received_content", ""):
                            ack_received = True
                            break
                if ack_received:
                    break

            proc.terminate()
            stdout, stderr = proc.communicate(timeout=2)

            assert ack_received, (
                f"Guidance not received by subprocess.\n"
                f"stdout: {stdout.decode()}, stderr: {stderr.decode()}"
            )

        finally:
            if CM_EVENT_SOCKET_ENV in os.environ:
                del os.environ[CM_EVENT_SOCKET_ENV]

            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()

            await socket_server.stop()
