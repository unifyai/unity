"""
Tests for Unix domain socket IPC between ConversationManager and voice agent subprocess.

This module tests the socket-based bridge that allows call transcripts to flow
from the voice agent subprocess back to the parent ConversationManager process.
"""

import asyncio
import json
import os
import time as _time
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from unity.conversation_manager.domains.ipc_socket import (
    CallEventSocketServer,
    CallEventSocketClient,
    CM_EVENT_SOCKET_ENV,
    get_socket_client,
    send_event_to_parent,
)

# =============================================================================
# Helper for deterministic waiting
# =============================================================================


async def _wait_for_condition(
    predicate,
    *,
    timeout: float = 5.0,
    poll: float = 0.01,
) -> bool:
    """Poll predicate() until True or timeout. Returns whether condition was met."""
    start = _time.perf_counter()
    while _time.perf_counter() - start < timeout:
        if predicate():
            return True
        await asyncio.sleep(poll)
    return False


class TestCallEventSocketServer:
    """Tests for the socket server that receives events from voice agent subprocess."""

    @pytest.fixture
    def mock_event_broker(self):
        """Create a mock event broker."""
        broker = MagicMock()
        broker.publish = AsyncMock()
        return broker

    @pytest.mark.asyncio
    async def test_start_and_stop_lifecycle(self, mock_event_broker):
        """Server start/stop creates and removes socket file, idempotently."""
        server = CallEventSocketServer(mock_event_broker, forward_channels=[])

        # Before start
        assert server.socket_path is None

        # Start creates socket
        path1 = await server.start()
        assert path1 is not None
        assert path1.endswith(".sock")
        assert os.path.exists(path1)
        assert server.socket_path == path1

        # Start is idempotent
        path2 = await server.start()
        assert path1 == path2

        # Stop removes socket
        await server.stop()
        assert not os.path.exists(path1)
        assert server.socket_path is None

        # Stop is idempotent
        await server.stop()

    @pytest.mark.asyncio
    async def test_on_event_callback_called(self, mock_event_broker):
        """Custom on_event callback is called instead of publishing to broker."""
        received_events = []

        async def on_event(channel, event_json):
            received_events.append((channel, event_json))

        server = CallEventSocketServer(
            mock_event_broker,
            on_event=on_event,
            forward_channels=[],
        )
        socket_path = await server.start()

        # Create client and send event
        client = CallEventSocketClient(socket_path)
        await client.send_event("test:channel", '{"test": "data"}')

        # Wait for event to be processed (poll instead of fixed sleep)
        await _wait_for_condition(lambda: len(received_events) >= 1)

        assert len(received_events) == 1
        assert received_events[0] == ("test:channel", '{"test": "data"}')
        mock_event_broker.publish.assert_not_called()

        await client.close()
        await server.stop()

    @pytest.mark.asyncio
    async def test_on_event_routes_screenshot_to_callback(
        self,
        mock_event_broker,
    ):
        """on_event intercepts screenshot events and routes them to a
        dedicated callback, while forwarding other events to the broker."""
        import json

        screenshot_received = []

        def on_screenshot(event_json):
            screenshot_received.append(event_json)

        async def on_ipc_event(channel, event_json):
            if channel == "app:comms:screenshot" and on_screenshot is not None:
                on_screenshot(event_json)
            else:
                await mock_event_broker.publish(channel, event_json)

        server = CallEventSocketServer(
            mock_event_broker,
            on_event=on_ipc_event,
            forward_channels=[],
        )
        socket_path = await server.start()
        client = CallEventSocketClient(socket_path)

        # Send a screenshot event
        screenshot_payload = json.dumps(
            {
                "b64": "iVBORw0KGgoAAAANSUhEUg==",
                "utterance": "Look here",
                "timestamp": "2026-02-15T12:00:00+00:00",
                "source": "user",
                "filepath": "Screenshots/User/2026-02-15T12-00-00.000000.jpg",
            },
        )
        await client.send_event(
            "app:comms:screenshot",
            screenshot_payload,
        )
        await _wait_for_condition(lambda: len(screenshot_received) >= 1)

        # Send a regular utterance event
        utterance_payload = json.dumps({"content": "Hello there"})
        await client.send_event(
            "app:comms:unify_meet_utterance",
            utterance_payload,
        )
        await _wait_for_condition(
            lambda: mock_event_broker.publish.call_count >= 1,
        )

        # Screenshot went to the dedicated callback, not the broker
        assert len(screenshot_received) == 1
        assert screenshot_received[0] == screenshot_payload

        # Utterance went to the broker, not the callback
        mock_event_broker.publish.assert_called_once_with(
            "app:comms:unify_meet_utterance",
            utterance_payload,
        )

        await client.close()
        await server.stop()

    @pytest.mark.asyncio
    async def test_server_handles_malformed_json(self, mock_event_broker):
        """Server gracefully handles malformed JSON without crashing."""
        server = CallEventSocketServer(mock_event_broker, forward_channels=[])
        socket_path = await server.start()

        # Connect directly and send malformed data
        import socket as sock

        client_socket = sock.socket(sock.AF_UNIX, sock.SOCK_STREAM)
        client_socket.connect(socket_path)
        client_socket.sendall(b"not valid json\n")
        client_socket.close()

        # Wait for processing
        await asyncio.sleep(0.1)

        # Server should still be running
        assert server._running is True

        # Should not have published anything
        mock_event_broker.publish.assert_not_called()

        await server.stop()

    @pytest.mark.asyncio
    async def test_server_handles_incomplete_message(self, mock_event_broker):
        """Server handles messages without required fields."""
        server = CallEventSocketServer(mock_event_broker, forward_channels=[])
        socket_path = await server.start()

        # Send message missing 'event' field
        import socket as sock

        client_socket = sock.socket(sock.AF_UNIX, sock.SOCK_STREAM)
        client_socket.connect(socket_path)
        client_socket.sendall(b'{"channel": "test"}\n')  # Missing 'event'
        client_socket.close()

        await asyncio.sleep(0.1)

        # Server should still be running and not crash
        assert server._running is True
        mock_event_broker.publish.assert_not_called()

        await server.stop()

    @pytest.mark.asyncio
    async def test_disconnect_fallback_waits_for_inflight_message_processing(
        self,
        mock_event_broker,
    ):
        """A disconnect fallback must not overtake an already received message.

        This reproduces the staging failure class where the voice worker sends a
        lifecycle event and then disconnects immediately afterward. The server
        schedules inbound message processing on the main loop, and separately
        schedules ``on_client_disconnected`` when the socket closes. If the
        disconnect callback runs first, the parent can observe
        ``UnifyMeetEnded`` before the earlier ``UnifyMeetStarted``.

        The correct ordering is to finish processing messages already received
        from the client before running the disconnect fallback.
        """
        from unity.conversation_manager.events import UnifyMeetStarted

        started_processing = asyncio.Event()
        allow_started_finish = asyncio.Event()
        seen: list[str] = []

        async def on_event(channel: str, event_json: str) -> None:
            event_name = json.loads(event_json)["event_name"]
            if event_name == "UnifyMeetStarted":
                started_processing.set()
                await allow_started_finish.wait()
            seen.append(event_name)

        server = CallEventSocketServer(
            mock_event_broker,
            on_event=on_event,
            forward_channels=[],
        )
        socket_path = await server.start()

        async def on_disconnected() -> None:
            seen.append("UnifyMeetEnded")

        server.on_client_disconnected = on_disconnected

        client = CallEventSocketClient(socket_path)
        try:
            await client.send_event(
                "app:comms:unify_meet_started",
                UnifyMeetStarted(contact={"contact_id": 1}).to_json(),
            )

            started_entered = await _wait_for_condition(
                lambda: started_processing.is_set(),
                timeout=2.0,
            )
            assert started_entered, "Timed out waiting for inbound message processing"

            await client.close()

            disconnected = await _wait_for_condition(
                lambda: len(server._connected_clients) == 0,
                timeout=2.0,
            )
            assert disconnected, "Timed out waiting for socket disconnect"

            allow_started_finish.set()

            ordered = await _wait_for_condition(lambda: len(seen) >= 2, timeout=2.0)
            assert ordered, f"Timed out waiting for lifecycle events, seen={seen!r}"
            assert seen == ["UnifyMeetStarted", "UnifyMeetEnded"]
        finally:
            await client.close()
            await server.stop()


class TestCallEventSocketClient:
    """Tests for the socket client used by voice agent subprocess."""

    @pytest_asyncio.fixture
    async def running_server(self):
        """Create and start a socket server for testing."""
        broker = MagicMock()
        broker.publish = AsyncMock()
        server = CallEventSocketServer(broker, forward_channels=[])
        socket_path = await server.start()
        yield server, socket_path, broker
        await server.stop()

    @pytest.mark.asyncio
    async def test_connect_success(self, running_server):
        """Client successfully connects to running server."""
        server, socket_path, broker = running_server

        client = CallEventSocketClient(socket_path)
        result = await client.connect()

        assert result is True
        assert client._connected is True

        await client.close()

    @pytest.mark.asyncio
    async def test_connect_failure_no_server(self):
        """Client connection fails gracefully when no server is running."""
        client = CallEventSocketClient("/tmp/nonexistent_socket.sock")
        result = await client.connect()

        assert result is False
        assert client._connected is False

    @pytest.mark.asyncio
    async def test_send_event_success(self, running_server):
        """Client successfully sends event to server."""
        server, socket_path, broker = running_server

        client = CallEventSocketClient(socket_path)
        result = await client.send_event(
            "app:comms:phone_utterance",
            '{"text": "hello"}',
        )

        assert result is True

        # Wait for event to be processed (poll instead of fixed sleep)
        await _wait_for_condition(lambda: broker.publish.called)

        broker.publish.assert_called_once_with(
            "app:comms:phone_utterance",
            '{"text": "hello"}',
        )

        await client.close()

    @pytest.mark.asyncio
    async def test_send_event_auto_connects(self, running_server):
        """send_event() auto-connects if not already connected."""
        server, socket_path, broker = running_server

        client = CallEventSocketClient(socket_path)
        assert client._connected is False

        result = await client.send_event("test:channel", '{"data": 1}')

        assert result is True
        assert client._connected is True

        await client.close()

    @pytest.mark.asyncio
    async def test_send_event_failure_no_server(self):
        """send_event() returns False when no server is available."""
        client = CallEventSocketClient("/tmp/nonexistent_socket.sock")
        result = await client.send_event("test:channel", '{"data": 1}')

        assert result is False

    @pytest.mark.asyncio
    async def test_from_env_returns_client_when_set(self):
        """from_env() returns a client when environment variable is set."""
        with patch.dict(os.environ, {CM_EVENT_SOCKET_ENV: "/tmp/test.sock"}):
            client = CallEventSocketClient.from_env()
            assert client is not None
            assert client._socket_path == "/tmp/test.sock"

    def test_from_env_returns_none_when_not_set(self):
        """from_env() returns None when environment variable is not set."""
        # Ensure env var is not set
        env_backup = os.environ.pop(CM_EVENT_SOCKET_ENV, None)
        try:
            client = CallEventSocketClient.from_env()
            assert client is None
        finally:
            if env_backup:
                os.environ[CM_EVENT_SOCKET_ENV] = env_backup

    @pytest.mark.asyncio
    async def test_close_clears_state(self, running_server):
        """close() properly clears connection state."""
        server, socket_path, broker = running_server

        client = CallEventSocketClient(socket_path)
        await client.connect()
        assert client._connected is True

        await client.close()

        assert client._connected is False
        assert client._socket is None


class TestSendEventToParent:
    """Tests for the send_event_to_parent convenience function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_no_socket(self):
        """send_event_to_parent() returns False when no socket is configured."""
        # Clear any existing client
        import unity.conversation_manager.domains.ipc_socket as ipc_module

        ipc_module._socket_client = None

        # Ensure env var is not set
        env_backup = os.environ.pop(CM_EVENT_SOCKET_ENV, None)
        try:
            result = await send_event_to_parent("test:channel", '{"data": 1}')
            assert result is False
        finally:
            if env_backup:
                os.environ[CM_EVENT_SOCKET_ENV] = env_backup

    @pytest.mark.asyncio
    async def test_sends_via_socket_when_configured(self):
        """send_event_to_parent() uses socket client when configured."""
        # Create a real server
        broker = MagicMock()
        broker.publish = AsyncMock()
        server = CallEventSocketServer(broker, forward_channels=[])
        socket_path = await server.start()

        # Clear singleton and set env var
        import unity.conversation_manager.domains.ipc_socket as ipc_module

        ipc_module._socket_client = None

        with patch.dict(os.environ, {CM_EVENT_SOCKET_ENV: socket_path}):
            result = await send_event_to_parent(
                "app:comms:phone_utterance",
                '{"text": "test"}',
            )

            assert result is True

            # Wait for event to be processed (poll instead of fixed sleep)
            await _wait_for_condition(lambda: broker.publish.called)

            broker.publish.assert_called_once_with(
                "app:comms:phone_utterance",
                '{"text": "test"}',
            )

        # Cleanup
        ipc_module._socket_client = None
        await server.stop()


class TestGetSocketClient:
    """Tests for the get_socket_client singleton function."""

    def test_returns_none_when_env_not_set(self):
        """get_socket_client() returns None when env var is not set."""
        import unity.conversation_manager.domains.ipc_socket as ipc_module

        ipc_module._socket_client = None

        env_backup = os.environ.pop(CM_EVENT_SOCKET_ENV, None)
        try:
            client = get_socket_client()
            assert client is None
        finally:
            if env_backup:
                os.environ[CM_EVENT_SOCKET_ENV] = env_backup

    def test_returns_singleton_when_set(self):
        """get_socket_client() returns the same instance on multiple calls."""
        import unity.conversation_manager.domains.ipc_socket as ipc_module

        ipc_module._socket_client = None

        with patch.dict(os.environ, {CM_EVENT_SOCKET_ENV: "/tmp/test.sock"}):
            client1 = get_socket_client()
            client2 = get_socket_client()

            assert client1 is client2
            assert client1 is not None

        # Cleanup
        ipc_module._socket_client = None


class TestBidirectionalCommunication:
    """Tests for bidirectional socket communication (parent ↔ child)."""

    @pytest.fixture
    def real_event_broker(self):
        """Create a real in-memory event broker for testing forwarding."""
        from unity.conversation_manager.in_memory_event_broker import (
            InMemoryEventBroker,
        )

        return InMemoryEventBroker()

    @pytest.mark.asyncio
    async def test_server_forwards_events_to_client(self, real_event_broker):
        """Server forwards matching events (guidance, status) to connected clients."""
        received_events = []

        async def on_event(channel: str, event_json: str):
            received_events.append((channel, event_json))

        server = CallEventSocketServer(
            real_event_broker,
            forward_channels=["app:call:*"],
        )
        socket_path = await server.start()

        client = CallEventSocketClient(socket_path)
        await client.start_receive_loop(on_event)

        # Small delay for connection to fully establish (local IPC is fast)
        await asyncio.sleep(0.05)

        # Send guidance event
        await real_event_broker.publish(
            "app:call:notification",
            '{"content": "Ask about their schedule"}',
        )

        # Send status event
        await real_event_broker.publish(
            "app:call:status",
            '{"type": "call_answered"}',
        )

        # Wait for both events to propagate
        await _wait_for_condition(lambda: len(received_events) >= 2)

        assert len(received_events) == 2
        assert any("schedule" in e[1] for e in received_events)
        assert any("call_answered" in e[1] for e in received_events)

        await client.close()
        await server.stop()

    @pytest.mark.asyncio
    async def test_server_does_not_forward_non_matching_channels(
        self,
        real_event_broker,
    ):
        """Server only forwards events matching forward_channels patterns."""
        received_events = []

        async def on_event(channel: str, event_json: str):
            received_events.append((channel, event_json))

        server = CallEventSocketServer(
            real_event_broker,
            forward_channels=["app:call:*"],  # Only app:call:* channels
        )
        socket_path = await server.start()

        client = CallEventSocketClient(socket_path)
        await client.start_receive_loop(on_event)

        # Small delay for connection to fully establish
        await asyncio.sleep(0.05)

        # Parent publishes on non-matching channel
        await real_event_broker.publish(
            "app:comms:email_received",
            '{"subject": "Test"}',
        )

        # Small wait to verify event does NOT arrive (negative assertion)
        await asyncio.sleep(0.1)

        # Client should NOT receive this event
        assert len(received_events) == 0

        await client.close()
        await server.stop()

    @pytest.mark.asyncio
    async def test_bidirectional_full_round_trip(self, real_event_broker):
        """Test complete bidirectional flow: parent → child AND child → parent."""
        parent_received = []
        child_received = []

        # Custom on_event for server to track parent-side received events
        async def parent_on_event(channel: str, event_json: str):
            parent_received.append((channel, event_json))
            # Also publish to broker for completeness
            await real_event_broker.publish(channel, event_json)

        async def child_on_event(channel: str, event_json: str):
            child_received.append((channel, event_json))

        # Start server with forwarding and custom callback
        server = CallEventSocketServer(
            real_event_broker,
            on_event=parent_on_event,
            forward_channels=["app:call:*"],
        )
        socket_path = await server.start()

        # Connect client and start receive loop
        client = CallEventSocketClient(socket_path)
        await client.start_receive_loop(child_on_event)
        await asyncio.sleep(0.1)

        # OUTBOUND: Child → Parent (utterance)
        await client.send_event(
            "app:comms:phone_utterance",
            '{"text": "Hello from child"}',
        )

        # INBOUND: Parent → Child (guidance)
        await real_event_broker.publish(
            "app:call:notification",
            '{"content": "Guidance from parent"}',
        )

        await asyncio.sleep(0.3)

        # Verify OUTBOUND: Parent received child's utterance
        assert len(parent_received) == 1
        assert parent_received[0][0] == "app:comms:phone_utterance"
        assert "Hello from child" in parent_received[0][1]

        # Verify INBOUND: Child received parent's guidance
        assert len(child_received) == 1
        assert child_received[0][0] == "app:call:notification"
        assert "Guidance from parent" in child_received[0][1]

        await client.close()
        await server.stop()

    @pytest.mark.asyncio
    async def test_multiple_clients_receive_forwarded_events(self, real_event_broker):
        """Server forwards events to all connected clients."""
        client1_received = []
        client2_received = []

        async def client1_on_event(channel: str, event_json: str):
            client1_received.append((channel, event_json))

        async def client2_on_event(channel: str, event_json: str):
            client2_received.append((channel, event_json))

        server = CallEventSocketServer(
            real_event_broker,
            forward_channels=["app:call:*"],
        )
        socket_path = await server.start()

        # Connect two clients
        client1 = CallEventSocketClient(socket_path)
        await client1.start_receive_loop(client1_on_event)

        client2 = CallEventSocketClient(socket_path)
        await client2.start_receive_loop(client2_on_event)

        await asyncio.sleep(0.1)

        # Parent publishes event
        await real_event_broker.publish(
            "app:call:notification",
            '{"content": "Broadcast message"}',
        )

        await asyncio.sleep(0.2)

        # Both clients should receive the event
        assert len(client1_received) == 1
        assert len(client2_received) == 1
        assert "Broadcast message" in client1_received[0][1]
        assert "Broadcast message" in client2_received[0][1]

        await client1.close()
        await client2.close()
        await server.stop()

    @pytest.mark.asyncio
    async def test_client_attempts_reconnect_on_server_disconnect(
        self,
        real_event_broker,
    ):
        """Client attempts to reconnect when server disconnects (resilience feature).

        The client is designed to be resilient to temporary server outages by
        attempting reconnection. This test verifies that behavior.
        """
        received_events = []

        async def on_event(channel: str, event_json: str):
            received_events.append((channel, event_json))

        server = CallEventSocketServer(
            real_event_broker,
            forward_channels=["app:call:*"],
        )
        socket_path = await server.start()

        client = CallEventSocketClient(socket_path)
        await client.start_receive_loop(on_event)
        await asyncio.sleep(0.1)

        # Verify client is connected and running
        assert client._connected is True
        assert client._running is True

        # Stop server while client is connected
        await server.stop()

        # Wait for client to detect disconnect and start reconnection
        await asyncio.sleep(0.3)

        # Client should still be "running" (attempting reconnection)
        # but no longer "connected" (server is gone)
        assert client._running is True
        assert client._connected is False

        # Clean up - explicitly stop the client
        await client.close()
        assert client._running is False

    @pytest.mark.asyncio
    async def test_client_stops_after_max_reconnect_attempts(self, real_event_broker):
        """Client stops running after exhausting all reconnection attempts.

        The client attempts up to 3 reconnections with 0.5s delays. After all
        attempts fail, it should stop gracefully.
        """
        received_events = []

        async def on_event(channel: str, event_json: str):
            received_events.append((channel, event_json))

        server = CallEventSocketServer(
            real_event_broker,
            forward_channels=["app:call:*"],
        )
        socket_path = await server.start()

        client = CallEventSocketClient(socket_path)
        await client.start_receive_loop(on_event)
        await asyncio.sleep(0.1)

        # Stop server - client will attempt reconnection
        await server.stop()

        # Wait for all 3 reconnect attempts to exhaust
        # Each attempt has 0.5s delay, plus some buffer for processing
        # 3 attempts * 0.5s = 1.5s minimum, add buffer for safety
        await asyncio.sleep(2.5)

        # After max attempts, client should have stopped
        assert client._running is False

        await client.close()

    @pytest.mark.asyncio
    async def test_server_buffers_messages_before_client_connects(
        self,
        real_event_broker,
    ):
        """Server buffers forwarded events that arrive before any client connects.

        When the parent publishes a FastBrainNotification event right after spawning
        the subprocess, the subprocess hasn't connected to the socket yet.
        The server must buffer the message and flush it once the client connects.
        """
        received_events = []

        async def on_event(channel: str, event_json: str):
            received_events.append((channel, event_json))

        # Start server with forwarding (no clients connected yet)
        server = CallEventSocketServer(
            real_event_broker,
            forward_channels=["app:call:*"],
        )
        await server.start()

        # Parent publishes guidance BEFORE any client connects
        await real_event_broker.publish(
            "app:call:notification",
            '{"content": "Confirm the Thursday 3pm meeting"}',
        )

        # Give the forward loop time to pick up the message and buffer it
        await _wait_for_condition(
            lambda: len(server._pending_messages) >= 1,
            timeout=2.0,
        )
        assert (
            len(server._pending_messages) == 1
        ), "Server should buffer the message when no clients are connected"

        # Now a client connects (simulates subprocess startup)
        client = CallEventSocketClient(server.socket_path)
        await client.start_receive_loop(on_event)

        # Wait for the buffered message to be flushed to the client
        await _wait_for_condition(lambda: len(received_events) >= 1, timeout=2.0)

        assert len(received_events) == 1
        assert received_events[0][0] == "app:call:notification"
        assert "Thursday 3pm" in received_events[0][1]

        # Buffer should be cleared after flush
        assert len(server._pending_messages) == 0

        await client.close()
        await server.stop()

    @pytest.mark.asyncio
    async def test_server_buffers_multiple_messages_before_client_connects(
        self,
        real_event_broker,
    ):
        """Server buffers multiple events and flushes all when client connects."""
        received_events = []

        async def on_event(channel: str, event_json: str):
            received_events.append((channel, event_json))

        server = CallEventSocketServer(
            real_event_broker,
            forward_channels=["app:call:*"],
        )
        await server.start()

        # Publish multiple events before any client connects
        await real_event_broker.publish(
            "app:call:notification",
            '{"content": "First guidance"}',
        )
        await real_event_broker.publish(
            "app:call:status",
            '{"type": "call_answered"}',
        )

        # Give the forward loop time to buffer both messages
        await _wait_for_condition(
            lambda: len(server._pending_messages) >= 2,
            timeout=2.0,
        )
        assert len(server._pending_messages) == 2

        # Client connects — both messages should be flushed
        client = CallEventSocketClient(server.socket_path)
        await client.start_receive_loop(on_event)

        await _wait_for_condition(lambda: len(received_events) >= 2, timeout=2.0)

        assert len(received_events) == 2
        assert any("First guidance" in e[1] for e in received_events)
        assert any("call_answered" in e[1] for e in received_events)
        assert len(server._pending_messages) == 0

        await client.close()
        await server.stop()


class TestSocketAwareEventBroker:
    """Tests for the SocketAwareEventBroker wrapper in common.py."""

    @pytest.mark.asyncio
    async def test_uses_socket_when_available(self):
        """SocketAwareEventBroker uses socket when CM_EVENT_SOCKET is set."""
        broker = MagicMock()
        broker.publish = AsyncMock()
        server = CallEventSocketServer(broker, forward_channels=[])
        socket_path = await server.start()

        # Import and test the wrapper
        import unity.conversation_manager.domains.ipc_socket as ipc_module

        ipc_module._socket_client = None

        with patch.dict(os.environ, {CM_EVENT_SOCKET_ENV: socket_path}):
            from unity.conversation_manager.medium_scripts.common import (
                SocketAwareEventBroker,
            )

            wrapper = SocketAwareEventBroker()
            await wrapper.publish("test:channel", '{"data": 1}')

            await asyncio.sleep(0.1)

            # Event should have gone through the socket
            broker.publish.assert_called_once_with("test:channel", '{"data": 1}')

        ipc_module._socket_client = None
        await server.stop()

    @pytest.mark.asyncio
    async def test_falls_back_to_in_memory_broker(self):
        """SocketAwareEventBroker falls back to in-memory broker when no socket."""
        import unity.conversation_manager.domains.ipc_socket as ipc_module

        ipc_module._socket_client = None

        env_backup = os.environ.pop(CM_EVENT_SOCKET_ENV, None)
        try:
            from unity.conversation_manager.medium_scripts.common import (
                SocketAwareEventBroker,
            )

            with patch(
                "unity.conversation_manager.medium_scripts.common.get_event_broker",
            ) as mock_get_broker:
                mock_broker = MagicMock()
                mock_broker.publish = AsyncMock()
                mock_get_broker.return_value = mock_broker

                wrapper = SocketAwareEventBroker()
                await wrapper.publish("test:channel", '{"data": 1}')

                mock_broker.publish.assert_called_once_with(
                    "test:channel",
                    '{"data": 1}',
                )
        finally:
            if env_backup:
                os.environ[CM_EVENT_SOCKET_ENV] = env_backup
            ipc_module._socket_client = None

    @pytest.mark.asyncio
    async def test_start_receiving_enables_inbound_events(self):
        """SocketAwareEventBroker.start_receiving() enables receiving parent events."""
        from unity.conversation_manager.in_memory_event_broker import (
            InMemoryEventBroker,
        )

        # Create real parent broker
        parent_broker = InMemoryEventBroker()

        # Start server with forwarding
        server = CallEventSocketServer(
            parent_broker,
            forward_channels=["app:call:*"],
        )
        socket_path = await server.start()

        import unity.conversation_manager.domains.ipc_socket as ipc_module

        ipc_module._socket_client = None
        ipc_module._receive_loop_started = False

        with patch.dict(os.environ, {CM_EVENT_SOCKET_ENV: socket_path}):
            from unity.conversation_manager.medium_scripts.common import (
                SocketAwareEventBroker,
            )

            # Create wrapper
            wrapper = SocketAwareEventBroker()

            # Register callback to capture events
            received_events = []

            def on_guidance(data):
                received_events.append(data)

            wrapper.register_callback("app:call:notification", on_guidance)

            # Start receiving
            result = await wrapper.start_receiving()
            assert result is True

            # Wait for receive loop to start
            await asyncio.sleep(0.1)

            # Parent publishes event
            await parent_broker.publish(
                "app:call:notification",
                '{"content": "Test guidance"}',
            )

            # Wait for forwarding
            await asyncio.sleep(0.3)

            # Should have received the forwarded event via callback
            assert len(received_events) == 1
            assert received_events[0]["content"] == "Test guidance"

            await wrapper.stop()

        ipc_module._socket_client = None
        ipc_module._receive_loop_started = False
        await server.stop()

    @pytest.mark.asyncio
    async def test_start_receiving_returns_false_without_socket(self):
        """start_receiving() returns False when no socket is available."""
        import unity.conversation_manager.domains.ipc_socket as ipc_module

        ipc_module._socket_client = None
        ipc_module._receive_loop_started = False

        env_backup = os.environ.pop(CM_EVENT_SOCKET_ENV, None)
        try:
            from unity.conversation_manager.medium_scripts.common import (
                SocketAwareEventBroker,
            )

            wrapper = SocketAwareEventBroker()
            result = await wrapper.start_receiving()

            assert result is False

        finally:
            if env_backup:
                os.environ[CM_EVENT_SOCKET_ENV] = env_backup
            ipc_module._socket_client = None
            ipc_module._receive_loop_started = False
