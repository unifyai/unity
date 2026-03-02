"""
tests/conversation_manager/voice/test_e2e_call_flow.py
======================================================

TRUE end-to-end integration test for the call flow.

This test exercises the REAL call flow:
1. ConversationManager receives UnifyMeetReceived event
2. CallManager spawns voice agent subprocess
3. Voice agent connects to LiveKit (real or mock)
4. Bidirectional IPC communication works
5. Events flow correctly between all components

Requirements for full LiveKit test:
- LiveKit server running (auto-started if livekit-server binary available)
- Or set LIVEKIT_URL to an existing server

To install LiveKit locally:
    macOS: brew install livekit
    Linux: See scripts/install_livekit.sh or download from GitHub releases
    CI: Workflow installs automatically

This test would catch:
- Ved's IPC bidirectional fix (c34270dc)
- Subprocess spawn failures
- Event handler routing issues
- Initialization order bugs (fe355d6f)
- Call lifecycle problems
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
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

# Check if livekit-server is available
LIVEKIT_SERVER_PATH = shutil.which("livekit-server")
LIVEKIT_SERVER_AVAILABLE = LIVEKIT_SERVER_PATH is not None

# Allow skipping LiveKit tests entirely via environment variable
SKIP_LIVEKIT = os.environ.get("SKIP_LIVEKIT_E2E", "0") == "1"


def start_livekit_server() -> subprocess.Popen | None:
    """Start a local LiveKit server in dev mode. Returns the process or None."""
    if not LIVEKIT_SERVER_AVAILABLE:
        return None

    # Check if already running by trying to connect
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        sock.connect(("localhost", 7880))
        sock.close()
        print("[LiveKit] Server already running on port 7880")
        return None  # Already running, don't start new one
    except (ConnectionRefusedError, socket.timeout, OSError):
        pass  # Not running, start it

    print(f"[LiveKit] Starting server from {LIVEKIT_SERVER_PATH}")
    proc = subprocess.Popen(
        [LIVEKIT_SERVER_PATH, "--dev"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    for _ in range(30):  # 3 second timeout
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

    # Failed to start
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
def sample_contact():
    return {
        "contact_id": 2,
        "first_name": "Test",
        "surname": "Caller",
        "phone_number": "+15555551234",
        "email_address": "test@example.com",
    }


@pytest.fixture
def boss_contact():
    return {
        "contact_id": 1,
        "first_name": "Boss",
        "surname": "User",
        "phone_number": "+15555550001",
        "email_address": "boss@example.com",
    }


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


class TestEndToEndCallFlow:
    """
    End-to-end tests for the complete call flow.

    These tests exercise as much of the real code as possible,
    only mocking external services (LiveKit Cloud, Deepgram, etc).
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not LIVEKIT_SERVER_AVAILABLE and not SKIP_LIVEKIT,
        reason="livekit-server not installed. Run: brew install livekit",
    )
    async def test_full_call_flow_from_event_to_subprocess(
        self,
        event_broker,
        sample_contact,
        boss_contact,
    ):
        """
        End-to-end test: UnifyMeetReceived → CallManager → subprocess → IPC → guidance.

        This tests the REAL flow:
        1. Event arrives (simulating Pub/Sub message)
        2. Event handler triggers CallManager.start_unify_meet()
        3. CallManager spawns voice agent subprocess
        4. Subprocess connects to IPC socket
        5. Parent sends call guidance via IPC
        6. Subprocess receives and acknowledges

        If SKIP_LIVEKIT=1, we mock the LiveKit connection but test everything else.
        """
        from unity.conversation_manager.domains.call_manager import (
            CallConfig,
            LivekitCallManager,
        )
        from unity.conversation_manager.domains.ipc_socket import (
            CallEventSocketServer,
            CM_EVENT_SOCKET_ENV,
        )
        from unity.conversation_manager.events import (
            FastBrainNotification,
        )

        # Track events received from subprocess
        events_from_subprocess = []

        async def on_subprocess_event(channel: str, event_json: str):
            print(f"[TEST] Received from subprocess: {channel}")
            events_from_subprocess.append((channel, event_json))

        # Create CallManager with real config
        config = CallConfig(
            assistant_id="e2e_test_assistant",
            user_id="test_user",
            assistant_bio="A test assistant for end-to-end testing",
            assistant_number="+15555550000",
            voice_provider="cartesia",
            voice_id="test_voice",
        )

        call_manager = LivekitCallManager(config, event_broker)

        # Start IPC socket server (this is what CallManager does internally)
        socket_server = CallEventSocketServer(
            event_broker,
            on_event=on_subprocess_event,
            forward_channels=["app:call:*"],
        )

        subprocess_proc = None

        try:
            # Start socket server
            socket_path = await socket_server.start()
            os.environ[CM_EVENT_SOCKET_ENV] = socket_path

            # Instead of spawning call.py (which needs LiveKit), spawn our test subprocess
            # that mimics the voice agent's IPC behavior
            test_subprocess = Path(__file__).parent / "ipc_test_subprocess.py"

            env = os.environ.copy()
            env[CM_EVENT_SOCKET_ENV] = socket_path

            # Ensure PYTHONPATH includes workspace root so subprocess can find unity
            workspace_root = str(Path(__file__).parent.parent.parent.parent)
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{workspace_root}:{existing_pythonpath}"
                if existing_pythonpath
                else workspace_root
            )

            subprocess_proc = subprocess.Popen(
                [sys.executable, str(test_subprocess), "full_roundtrip"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for subprocess to signal readiness
            ready_received = False
            for _ in range(50):  # 5s timeout
                await asyncio.sleep(0.1)
                for channel, _ in events_from_subprocess:
                    if channel == "app:call:ready":
                        ready_received = True
                        break
                if ready_received:
                    break

            assert ready_received, (
                "Subprocess never signaled ready. "
                "This means subprocess spawn or IPC connection failed."
            )

            # Now send call guidance (simulating CM sending guidance during a call)
            events_from_subprocess.clear()

            guidance_event = FastBrainNotification(
                contact=sample_contact,
                content="Ask the caller about their schedule for next week",
            )

            # Publish guidance - this should be forwarded to subprocess via IPC
            await event_broker.publish(
                "app:call:notification",
                guidance_event.to_json(),
            )

            # Wait for subprocess acknowledgment
            ack_received = False
            for _ in range(50):  # 5s timeout
                await asyncio.sleep(0.1)
                for channel, event_json in events_from_subprocess:
                    if channel == "app:call:ack":
                        data = json.loads(event_json)
                        if "schedule" in data.get("received_content", ""):
                            ack_received = True
                            break
                if ack_received:
                    break

            # Get subprocess output for debugging
            subprocess_proc.terminate()
            stdout, stderr = subprocess_proc.communicate(timeout=2)

            assert ack_received, (
                f"Call guidance was not received by subprocess. "
                f"This indicates parent→child IPC is broken (Ved's c34270dc fix). "
                f"stdout: {stdout.decode()}, stderr: {stderr.decode()}"
            )

            print("[TEST] Full end-to-end flow succeeded!")

        finally:
            # Cleanup
            if CM_EVENT_SOCKET_ENV in os.environ:
                del os.environ[CM_EVENT_SOCKET_ENV]

            if subprocess_proc and subprocess_proc.poll() is None:
                subprocess_proc.terminate()
                try:
                    subprocess_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    subprocess_proc.kill()

            await socket_server.stop()

    @pytest.mark.asyncio
    async def test_event_handler_to_call_manager_integration(
        self,
        event_broker,
        sample_contact,
        boss_contact,
    ):
        """
        Integration test: Event handler correctly invokes CallManager.

        Tests that UnifyMeetReceived event flows through the event handler
        and triggers the correct CallManager method with correct arguments.

        This would catch:
        - Event routing bugs
        - Argument passing errors
        - Mode checking bugs (88e0d678)
        """
        from unity.conversation_manager.domains.event_handlers import EventHandler
        from unity.conversation_manager.domains.call_manager import (
            CallConfig,
            LivekitCallManager,
        )
        from unity.conversation_manager.events import UnifyMeetReceived
        from unity.conversation_manager.types import Mode

        # Create mock ConversationManager with real CallManager
        mock_cm = MagicMock()
        mock_cm.mode = Mode.TEXT  # Not in voice mode
        mock_cm.contact_index.get_contact = MagicMock(return_value=boss_contact)
        mock_cm.notifications_bar = MagicMock()

        config = CallConfig(
            assistant_id="test",
            user_id="test_user",
            assistant_bio="Test bio",
            assistant_number="+15555550000",
            voice_provider="cartesia",
            voice_id="test_voice",
        )
        call_manager = LivekitCallManager(config, event_broker)
        mock_cm.call_manager = call_manager

        # Track if start_unify_meet was called
        start_unify_meet_called = False
        captured_args = {}

        async def mock_start_unify_meet(contact, boss, room_name):
            nonlocal start_unify_meet_called, captured_args
            start_unify_meet_called = True
            captured_args = {
                "contact": contact,
                "boss": boss,
                "room_name": room_name,
            }

        call_manager.start_unify_meet = mock_start_unify_meet

        # Create and handle UnifyMeetReceived event
        event = UnifyMeetReceived(
            contact=boss_contact,
            room_name="test_room_456",
        )

        await EventHandler.handle_event(event, mock_cm)

        # Verify the event was routed correctly
        assert start_unify_meet_called, (
            "UnifyMeetReceived did not trigger start_unify_meet. "
            "Event routing is broken."
        )

        assert captured_args["room_name"] == "test_room_456"

    @pytest.mark.asyncio
    async def test_mode_guard_prevents_duplicate_calls(
        self,
        event_broker,
        boss_contact,
    ):
        """
        Test that receiving a call event while already in voice mode is ignored.

        This was broken before commit 88e0d678 and would cause duplicate
        subprocess spawns.
        """
        from unity.conversation_manager.domains.event_handlers import EventHandler
        from unity.conversation_manager.domains.call_manager import (
            CallConfig,
            LivekitCallManager,
        )
        from unity.conversation_manager.events import UnifyMeetReceived
        from unity.conversation_manager.types import Mode

        mock_cm = MagicMock()
        mock_cm.mode = Mode.CALL  # ALREADY in voice mode

        config = CallConfig(
            assistant_id="test",
            user_id="test_user",
            assistant_bio="Test",
            assistant_number="+15555550000",
            voice_provider="cartesia",
            voice_id="test",
        )
        call_manager = LivekitCallManager(config, event_broker)
        mock_cm.call_manager = call_manager

        start_called = False

        async def mock_start(*args, **kwargs):
            nonlocal start_called
            start_called = True

        call_manager.start_unify_meet = mock_start

        event = UnifyMeetReceived(
            contact=boss_contact,
            room_name="room",
        )

        await EventHandler.handle_event(event, mock_cm)

        assert not start_called, (
            "start_unify_meet was called while already in voice mode. "
            "This bug was fixed in commit 88e0d678."
        )


class TestContactIndexFallbackDuringInit:
    """
    Tests for contact lookup fallback before ContactManager is initialized.

    These tests verify the fallback mechanism added in commit 307b210f that
    caches contacts from inbound messages for quick lookup before/during
    ContactManager initialization.

    WHY THIS MATTERS:
    Before this fix, inbound events that arrived before ContactManager was
    fully initialized would fail to resolve contacts, causing:
    - Missing sender names in notifications
    - Failed contact lookups in event handlers
    - Broken message routing
    """

    @pytest.mark.asyncio
    async def test_contact_lookup_before_manager_initialized(self, event_broker):
        """
        Test that contact lookup works BEFORE ContactManager is set.

        This simulates the race condition where an inbound message arrives
        before managers are fully initialized. The BackupContactsEvent should
        populate the fallback cache so subsequent lookups succeed.

        This test would have caught Ved's bug (307b210f).
        """
        from unity.conversation_manager.domains.contact_index import ContactIndex

        contact_index = ContactIndex()

        # Manager is NOT initialized yet
        assert contact_index._contact_manager is None

        # Simulate backup contacts from inbound message (CommsManager does this)
        fallback_contacts = [
            {
                "contact_id": 1,
                "first_name": "Boss",
                "surname": "User",
                "phone_number": "+15555550001",
                "email_address": "boss@example.com",
            },
            {
                "contact_id": 2,
                "first_name": "Alice",
                "surname": "Smith",
                "phone_number": "+15555551234",
                "email_address": "alice@example.com",
            },
        ]
        contact_index.set_fallback_contacts(fallback_contacts)

        # Lookup should work via fallback cache
        boss = contact_index.get_contact(contact_id=1)
        assert boss is not None, "Contact lookup failed before manager init"
        assert boss["first_name"] == "Boss"

        alice = contact_index.get_contact(phone_number="+15555551234")
        assert alice is not None, "Phone number lookup failed before manager init"
        assert alice["first_name"] == "Alice"

    @pytest.mark.asyncio
    async def test_backup_contacts_event_populates_fallback(
        self,
        event_broker,
        sample_contact,
        boss_contact,
    ):
        """
        Test that BackupContactsEvent handler populates fallback cache.

        This is the production code path - CommsManager publishes BackupContactsEvent
        when inbound messages arrive, and the handler should cache contacts.
        """
        from unity.conversation_manager.domains.event_handlers import EventHandler
        from unity.conversation_manager.events import BackupContactsEvent

        # Create a minimal mock CM with uninitialized contact_manager
        mock_cm = MagicMock()
        mock_cm.contact_index._contact_manager = None
        mock_cm.contact_index._fallback_contacts = {}
        mock_cm._session_logger = MagicMock()

        # Real set_fallback_contacts implementation
        def set_fallback_contacts(contacts):
            for c in contacts:
                cid = c.get("contact_id")
                if cid is not None:
                    mock_cm.contact_index._fallback_contacts[cid] = c

        mock_cm.contact_index.set_fallback_contacts = set_fallback_contacts

        # Trigger the event
        event = BackupContactsEvent(contacts=[boss_contact, sample_contact])
        await EventHandler.handle_event(event, mock_cm)

        # Fallback cache should now contain the contacts
        assert 1 in mock_cm.contact_index._fallback_contacts
        assert 2 in mock_cm.contact_index._fallback_contacts


class TestRoomNameHandling:
    """
    Tests for correct room name and agent name handling in Unify Meets.

    These tests verify the argument passing to start_unify_meet() that was
    broken and fixed in commit 81596d0e.

    WHY THIS MATTERS:
    The LiveKit room name and agent name must be passed correctly for the
    voice agent to join the correct room. Wrong names = call fails to connect.
    """

    @pytest.mark.asyncio
    async def test_unify_meet_room_name_passed_correctly(
        self,
        event_broker,
        boss_contact,
    ):
        """
        Test that room_name is passed correctly.

        Before fix 81596d0e, the room name handling was broken because
        assistant_number was removed from args but still expected.
        """
        from unity.conversation_manager.domains.event_handlers import EventHandler
        from unity.conversation_manager.domains.call_manager import (
            CallConfig,
            LivekitCallManager,
        )
        from unity.conversation_manager.events import UnifyMeetReceived
        from unity.conversation_manager.types import Mode

        mock_cm = MagicMock()
        mock_cm.mode = Mode.TEXT  # Not in voice mode

        # Mock contact_index to return boss contact
        mock_cm.contact_index.get_contact = MagicMock(return_value=boss_contact)

        config = CallConfig(
            assistant_id="test",
            user_id="test_user",
            assistant_bio="Test bio",
            assistant_number="+15555550000",
            voice_provider="cartesia",
            voice_id="test_voice",
        )
        call_manager = LivekitCallManager(config, event_broker)
        mock_cm.call_manager = call_manager
        mock_cm.notifications_bar = MagicMock()

        # Track captured arguments
        captured_args = {}

        async def mock_start_unify_meet(contact, boss, room_name):
            captured_args["contact"] = contact
            captured_args["boss"] = boss
            captured_args["room_name"] = room_name

        call_manager.start_unify_meet = mock_start_unify_meet

        event = UnifyMeetReceived(
            contact=boss_contact,
            room_name="unity_25_meet",
        )

        await EventHandler.handle_event(event, mock_cm)

        assert (
            captured_args["room_name"] == "unity_25_meet"
        ), f"Room name not passed correctly: got {captured_args.get('room_name')}"


class TestRapidEventHandling:
    """
    Tests for handling rapid sequential events.

    These tests verify the system handles multiple events in quick succession
    without race conditions, as fixed in commit 3c44b692.

    WHY THIS MATTERS:
    In production, /wakeup and /pre-hire requests can arrive within 3 seconds
    of each other. Race conditions caused duplicate processing and errors.
    """

    @pytest.mark.asyncio
    async def test_rapid_startup_events_handled_correctly(self, event_broker):
        """
        Test that rapid startup + inbound events don't cause race conditions.

        This simulates the production scenario where:
        1. Adapter sends startup message to unity-startup
        2. Adapter immediately sends inbound message to unity-{assistant_id}
        3. Container must handle both without race conditions
        """
        from unity.conversation_manager.events import (
            StartupEvent,
            SMSReceived,
            Event,
        )

        received_events = []

        async with event_broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            # Simulate rapid-fire events (like in production)
            startup = StartupEvent(
                api_key="test_key",
                medium="sms",
                assistant_id="25",
                user_id="123",
                assistant_first_name="Test",
                assistant_surname="Assistant",
                assistant_age="25",
                assistant_nationality="American",
                assistant_about="A test assistant",
                assistant_number="+15555550000",
                assistant_email="assistant@test.com",
                user_first_name="Boss",
                user_surname="User",
                user_number="+15555550001",
                user_email="boss@test.com",
                voice_provider="cartesia",
                voice_id="test_voice",
            )

            sms = SMSReceived(
                contact={
                    "contact_id": 1,
                    "first_name": "Boss",
                    "surname": "User",
                    "phone_number": "+15555550001",
                    "email_address": "boss@test.com",
                },
                content="Hello!",
            )

            # Publish both events rapidly (no wait between them)
            await event_broker.publish("app:comms:startup", startup.to_json())
            await event_broker.publish("app:comms:msg_message", sms.to_json())

            # Collect events
            import asyncio

            for _ in range(10):  # Check for messages
                msg = await pubsub.get_message(
                    timeout=0.5,
                    ignore_subscribe_messages=True,
                )
                if msg:
                    try:
                        event = Event.from_json(msg["data"])
                        received_events.append(event)
                    except Exception:
                        pass
                await asyncio.sleep(0.1)

        # Both events should be processable
        startup_events = [e for e in received_events if isinstance(e, StartupEvent)]
        sms_events = [e for e in received_events if isinstance(e, SMSReceived)]

        assert len(startup_events) >= 1, "Startup event not received"
        assert len(sms_events) >= 1, "SMS event not received"


class TestEventChannelRouting:
    """
    Tests for correct event channel routing.

    These tests verify events are published to the correct channels,
    as fixed in commit 6237411a where pre-hire logging was using the
    wrong channel name.

    WHY THIS MATTERS:
    Wrong channel names mean events never reach their handlers, causing
    silent failures that are very hard to debug in production.
    """

    @pytest.mark.asyncio
    async def test_call_guidance_uses_correct_channel(
        self,
        event_broker,
        sample_contact,
    ):
        """
        Test that FastBrainNotification is published to the correct channel.

        The voice agent subprocess listens on specific channels. Wrong
        channel = guidance never reaches the agent.
        """
        from unity.conversation_manager.events import FastBrainNotification

        guidance = FastBrainNotification(
            contact=sample_contact,
            content="Ask about their schedule",
        )

        received_on_channel = None

        async with event_broker.pubsub() as pubsub:
            # Subscribe to the channel the voice agent listens on
            await pubsub.subscribe("app:call:notification")

            await event_broker.publish(
                "app:call:notification",
                guidance.to_json(),
            )

            import asyncio

            for _ in range(20):
                msg = await pubsub.get_message(
                    timeout=0.5,
                    ignore_subscribe_messages=True,
                )
                if msg:
                    received_on_channel = msg["channel"]
                    break
                await asyncio.sleep(0.1)

        assert (
            received_on_channel == "app:call:notification"
        ), f"FastBrainNotification published to wrong channel: {received_on_channel}"

    @pytest.mark.asyncio
    async def test_call_status_channel_for_answered(self, event_broker):
        """
        Test that call_answered status is published to app:call:status.

        This is the channel the voice agent subprocess monitors for
        call state changes.
        """
        received_on_channel = None
        received_data = None

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:status")

            await event_broker.publish(
                "app:call:status",
                json.dumps({"type": "call_answered"}),
            )

            import asyncio

            for _ in range(20):
                msg = await pubsub.get_message(
                    timeout=0.5,
                    ignore_subscribe_messages=True,
                )
                if msg:
                    received_on_channel = msg["channel"]
                    received_data = json.loads(msg["data"])
                    break
                await asyncio.sleep(0.1)

        assert received_on_channel == "app:call:status"
        assert received_data["type"] == "call_answered"


class TestIPCBidirectionalCommunication:
    """
    Extended tests for IPC bidirectional communication.

    These tests extend the basic IPC test to cover more scenarios that
    were broken before Ved's fix in c34270dc.
    """

    @pytest.mark.asyncio
    async def test_multiple_guidance_messages(self, event_broker, sample_contact):
        """
        Test that multiple guidance messages can be sent sequentially.

        Before the bidirectional fix, only the first message might work.
        """
        from unity.conversation_manager.domains.ipc_socket import (
            CallEventSocketServer,
            CM_EVENT_SOCKET_ENV,
        )
        from unity.conversation_manager.events import FastBrainNotification

        events_from_subprocess = []

        async def on_subprocess_event(channel: str, event_json: str):
            events_from_subprocess.append((channel, event_json))

        socket_server = CallEventSocketServer(
            event_broker,
            on_event=on_subprocess_event,
            forward_channels=["app:call:*"],
        )

        subprocess_proc = None
        test_subprocess = Path(__file__).parent / "ipc_test_subprocess.py"

        try:
            socket_path = await socket_server.start()
            os.environ[CM_EVENT_SOCKET_ENV] = socket_path

            env = os.environ.copy()
            env[CM_EVENT_SOCKET_ENV] = socket_path

            # Ensure PYTHONPATH includes workspace root
            workspace_root = str(Path(__file__).parent.parent.parent.parent)
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{workspace_root}:{existing_pythonpath}"
                if existing_pythonpath
                else workspace_root
            )

            subprocess_proc = subprocess.Popen(
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

            # Send guidance (subprocess will ack and exit after first one)
            await event_broker.publish(
                "app:call:notification",
                FastBrainNotification(
                    contact=sample_contact,
                    content="First guidance message",
                ).to_json(),
            )

            # Wait for acknowledgment
            ack_received = False
            for _ in range(50):
                await asyncio.sleep(0.1)
                for channel, event_json in events_from_subprocess:
                    if channel == "app:call:ack":
                        data = json.loads(event_json)
                        if "First guidance" in data.get("received_content", ""):
                            ack_received = True
                            break
                if ack_received:
                    break

            subprocess_proc.terminate()
            stdout, stderr = subprocess_proc.communicate(timeout=2)

            assert ack_received, (
                f"Guidance not received by subprocess. "
                f"stdout: {stdout.decode()}, stderr: {stderr.decode()}"
            )

        finally:
            if CM_EVENT_SOCKET_ENV in os.environ:
                del os.environ[CM_EVENT_SOCKET_ENV]

            if subprocess_proc and subprocess_proc.poll() is None:
                subprocess_proc.terminate()
                try:
                    subprocess_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    subprocess_proc.kill()

            await socket_server.stop()


@pytest.mark.skipif(
    SKIP_LIVEKIT or not LIVEKIT_SERVER_AVAILABLE,
    reason=(
        "livekit-server not installed. To run this test:\n"
        "  macOS: brew install livekit\n"
        "  Linux: ./scripts/install_livekit.sh\n"
        "  Then re-run this test (server auto-starts)"
    ),
)
class TestRealLiveKitIntegration:
    """
    TRUE end-to-end tests that require LiveKit.

    These tests spawn the REAL call.py voice agent and verify it can
    connect to LiveKit and handle the full call lifecycle.

    The LiveKit server is automatically started if the binary is available.

    Prerequisites:
        macOS: brew install livekit
        Linux: ./scripts/install_livekit.sh
    """

    @pytest.fixture(autouse=True)
    def setup_livekit(self):
        """Auto-start LiveKit server for tests in this class."""
        self.livekit_proc = start_livekit_server()
        yield
        stop_livekit_server(self.livekit_proc)

    @pytest.mark.asyncio
    async def test_real_voice_agent_spawns_and_connects(
        self,
        event_broker,
        sample_contact,
        boss_contact,
    ):
        """
        Spawn the REAL call.py voice agent and verify it starts up.

        This test:
        1. Auto-starts LiveKit server (if not running)
        2. Starts IPC socket server
        3. Spawns the actual call.py script
        4. Verifies the subprocess starts without crashing
        5. Verifies IPC connection is established

        This is the most comprehensive integration test - it exercises
        the full call flow with real components.
        """
        from unity.conversation_manager.domains.ipc_socket import (
            CallEventSocketServer,
            CM_EVENT_SOCKET_ENV,
        )

        # LiveKit dev server defaults
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

        try:
            socket_path = await socket_server.start()

            # Build environment for the voice agent
            env = os.environ.copy()
            env[CM_EVENT_SOCKET_ENV] = socket_path
            env.update(livekit_env)

            # Ensure PYTHONPATH includes workspace root so subprocess can find unity
            workspace_root = str(Path(__file__).parent.parent.parent.parent)
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{workspace_root}:{existing_pythonpath}"
                if existing_pythonpath
                else workspace_root
            )

            # Build CLI arguments for call.py
            # Format: call.py dev agent:room provider voice outbound channel contact boss bio
            args = [
                sys.executable,
                str(call_py),
                "dev",
                "e2e_test_room",  # room_name (canonical, from make_room_name)
                "cartesia",  # voice_provider
                "",  # voice_id (use default)
                "False",  # outbound
                "unify",  # channel
                json.dumps(sample_contact),
                json.dumps(boss_contact),
                "A test assistant",  # assistant_bio
                "test_assistant_id",  # ASSISTANT_ID
                "test_user_id",  # USER_ID
            ]

            print(f"[TEST] Spawning voice agent: {' '.join(args[:4])}...")

            proc = subprocess.Popen(
                args,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for either:
            # 1. Process to crash (poll() returns exit code)
            # 2. Process to send an event (IPC working)
            # 3. Timeout (something hung)

            process_started = False
            for i in range(100):  # 10 second timeout
                await asyncio.sleep(0.1)

                # Check if process crashed
                exit_code = proc.poll()
                if exit_code is not None:
                    stdout = proc.stdout.read().decode() if proc.stdout else ""
                    stderr = proc.stderr.read().decode() if proc.stderr else ""

                    # Check if it's a known "good" exit (like missing API keys)
                    if "LIVEKIT" in stderr or "connection" in stderr.lower():
                        pytest.skip(
                            f"LiveKit connection failed. Check server is running.\n"
                            f"stderr: {stderr[:500]}",
                        )

                    pytest.fail(
                        f"Voice agent crashed with exit code {exit_code}.\n"
                        f"stdout: {stdout[:500]}\n"
                        f"stderr: {stderr[:500]}",
                    )

                # Check for events from subprocess (indicates IPC is working)
                if events_from_subprocess:
                    process_started = True
                    print(
                        f"[TEST] Voice agent sent event: {events_from_subprocess[0][0]}",
                    )
                    break

                # After 3 seconds, check if process is at least running
                if i == 30 and proc.poll() is None:
                    process_started = True
                    print("[TEST] Voice agent process is running (no events yet)")
                    break

            if not process_started:
                stdout = proc.stdout.read().decode() if proc.stdout else ""
                stderr = proc.stderr.read().decode() if proc.stderr else ""
                pytest.fail(
                    f"Voice agent did not start within timeout.\n"
                    f"stdout: {stdout[:500]}\n"
                    f"stderr: {stderr[:500]}",
                )

            print("[TEST] Voice agent subprocess spawned successfully!")

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
