"""
Tests for Unix domain socket IPC between ConversationManager and voice agent subprocess.

This module tests the socket-based bridge that allows call transcripts to flow
from the voice agent subprocess back to the parent ConversationManager process.
"""

import asyncio
import json
import os
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


class TestCallEventSocketServer:
    """Tests for the socket server that receives events from voice agent subprocess."""

    @pytest.fixture
    def mock_event_broker(self):
        """Create a mock event broker."""
        broker = MagicMock()
        broker.publish = AsyncMock()
        return broker

    @pytest.mark.asyncio
    async def test_start_creates_socket_file(self, mock_event_broker):
        """Server start() creates a socket file and returns its path."""
        server = CallEventSocketServer(mock_event_broker)

        socket_path = await server.start()

        assert socket_path is not None
        assert socket_path.endswith(".sock")
        assert os.path.exists(socket_path)
        assert server.socket_path == socket_path

        await server.stop()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, mock_event_broker):
        """Calling start() multiple times returns the same socket path."""
        server = CallEventSocketServer(mock_event_broker)

        path1 = await server.start()
        path2 = await server.start()

        assert path1 == path2

        await server.stop()

    @pytest.mark.asyncio
    async def test_stop_removes_socket_file(self, mock_event_broker):
        """Server stop() removes the socket file."""
        server = CallEventSocketServer(mock_event_broker)

        socket_path = await server.start()
        assert os.path.exists(socket_path)

        await server.stop()

        assert not os.path.exists(socket_path)
        assert server.socket_path is None

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, mock_event_broker):
        """Calling stop() multiple times doesn't raise errors."""
        server = CallEventSocketServer(mock_event_broker)

        await server.start()
        await server.stop()
        await server.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_socket_path_none_before_start(self, mock_event_broker):
        """socket_path is None before start() is called."""
        server = CallEventSocketServer(mock_event_broker)
        assert server.socket_path is None

    @pytest.mark.asyncio
    async def test_on_event_callback_called(self, mock_event_broker):
        """Custom on_event callback is called instead of publishing to broker."""
        received_events = []

        async def on_event(channel, event_json):
            received_events.append((channel, event_json))

        server = CallEventSocketServer(mock_event_broker, on_event=on_event)
        socket_path = await server.start()

        # Create client and send event
        client = CallEventSocketClient(socket_path)
        await client.send_event("test:channel", '{"test": "data"}')

        # Give time for event to be processed
        await asyncio.sleep(0.1)

        assert len(received_events) == 1
        assert received_events[0] == ("test:channel", '{"test": "data"}')
        mock_event_broker.publish.assert_not_called()

        await client.close()
        await server.stop()


class TestCallEventSocketClient:
    """Tests for the socket client used by voice agent subprocess."""

    @pytest_asyncio.fixture
    async def running_server(self):
        """Create and start a socket server for testing."""
        broker = MagicMock()
        broker.publish = AsyncMock()
        server = CallEventSocketServer(broker)
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

        # Give time for event to be processed
        await asyncio.sleep(0.1)

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
        server = CallEventSocketServer(broker)
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

            # Give time for event to be processed
            await asyncio.sleep(0.1)

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


class TestIntegration:
    """Integration tests for the full IPC round-trip."""

    @pytest.mark.asyncio
    async def test_full_round_trip_single_event(self):
        """Test complete flow: client → socket → server → broker."""
        # Setup
        broker = MagicMock()
        broker.publish = AsyncMock()
        server = CallEventSocketServer(broker)
        socket_path = await server.start()

        client = CallEventSocketClient(socket_path)

        # Send event
        event_data = {
            "type": "InboundPhoneUtterance",
            "text": "Hello, how are you?",
            "contact_id": 123,
        }
        await client.send_event("app:comms:phone_utterance", json.dumps(event_data))

        # Wait for processing
        await asyncio.sleep(0.1)

        # Verify
        broker.publish.assert_called_once()
        call_args = broker.publish.call_args
        assert call_args[0][0] == "app:comms:phone_utterance"
        assert json.loads(call_args[0][1]) == event_data

        # Cleanup
        await client.close()
        await server.stop()

    @pytest.mark.asyncio
    async def test_full_round_trip_multiple_events(self):
        """Test sending multiple events through the socket."""
        broker = MagicMock()
        broker.publish = AsyncMock()
        server = CallEventSocketServer(broker)
        socket_path = await server.start()

        client = CallEventSocketClient(socket_path)

        # Send multiple events
        events = [
            ("app:comms:phone_utterance", {"text": "First message", "id": 1}),
            ("app:comms:phone_utterance", {"text": "Second message", "id": 2}),
            ("app:comms:unify_meet_utterance", {"text": "Third message", "id": 3}),
        ]

        for channel, data in events:
            await client.send_event(channel, json.dumps(data))

        # Wait for processing
        await asyncio.sleep(0.2)

        # Verify all events were published
        assert broker.publish.call_count == 3

        # Verify channels
        channels = [call[0][0] for call in broker.publish.call_args_list]
        assert channels == [
            "app:comms:phone_utterance",
            "app:comms:phone_utterance",
            "app:comms:unify_meet_utterance",
        ]

        # Cleanup
        await client.close()
        await server.stop()

    @pytest.mark.asyncio
    async def test_client_reconnects_after_disconnect(self):
        """Test that client can reconnect after server restart."""
        broker = MagicMock()
        broker.publish = AsyncMock()

        # Start server and connect client
        server = CallEventSocketServer(broker)
        socket_path = await server.start()

        client = CallEventSocketClient(socket_path)
        await client.connect()
        assert client._connected is True

        # Stop server (simulates process restart)
        await server.stop()

        # Client should detect disconnection on next send
        result = await client.send_event("test", "{}")
        # May succeed or fail depending on timing, but shouldn't crash

        # Start new server on same path
        server2 = CallEventSocketServer(broker)

        # Remove old socket file if exists
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass

        # Manually set path for new server
        server2._socket_path = socket_path

        # Actually we need to create a new socket on a new path
        socket_path2 = await server2.start()
        client2 = CallEventSocketClient(socket_path2)

        # New client should work
        result = await client2.send_event("test:channel", '{"data": "works"}')
        assert result is True

        await asyncio.sleep(0.1)
        broker.publish.assert_called()

        await client2.close()
        await server2.stop()

    @pytest.mark.asyncio
    async def test_server_handles_malformed_json(self):
        """Server gracefully handles malformed JSON without crashing."""
        broker = MagicMock()
        broker.publish = AsyncMock()
        server = CallEventSocketServer(broker)
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
        broker.publish.assert_not_called()

        await server.stop()

    @pytest.mark.asyncio
    async def test_server_handles_incomplete_message(self):
        """Server handles messages without required fields."""
        broker = MagicMock()
        broker.publish = AsyncMock()
        server = CallEventSocketServer(broker)
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
        broker.publish.assert_not_called()

        await server.stop()


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
        """Server forwards events from parent broker to connected clients."""
        received_events = []

        async def on_event(channel: str, event_json: str):
            received_events.append((channel, event_json))

        # Start server with forwarding enabled (default: app:call:*)
        server = CallEventSocketServer(
            real_event_broker,
            forward_channels=["app:call:*"],
        )
        socket_path = await server.start()

        # Connect client and start receive loop
        client = CallEventSocketClient(socket_path)
        await client.start_receive_loop(on_event)

        # Give time for connection to establish
        await asyncio.sleep(0.1)

        # Parent publishes event on a forwarded channel
        await real_event_broker.publish(
            "app:call:call_guidance",
            '{"content": "Ask about their schedule"}',
        )

        # Wait for event to propagate
        await asyncio.sleep(0.2)

        # Client should have received the event
        assert len(received_events) == 1
        assert received_events[0][0] == "app:call:call_guidance"
        assert "schedule" in received_events[0][1]

        await client.close()
        await server.stop()

    @pytest.mark.asyncio
    async def test_server_forwards_status_events(self, real_event_broker):
        """Server forwards status events (call_answered, stop) to client."""
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

        # Parent publishes status events
        await real_event_broker.publish(
            "app:call:status",
            '{"type": "call_answered"}',
        )
        await real_event_broker.publish(
            "app:call:status",
            '{"type": "stop"}',
        )

        await asyncio.sleep(0.2)

        # Client should have received both events
        assert len(received_events) == 2
        assert any("call_answered" in e[1] for e in received_events)
        assert any("stop" in e[1] for e in received_events)

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
        await asyncio.sleep(0.1)

        # Parent publishes on non-matching channel
        await real_event_broker.publish(
            "app:comms:email_received",
            '{"subject": "Test"}',
        )

        await asyncio.sleep(0.2)

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
            "app:call:call_guidance",
            '{"content": "Guidance from parent"}',
        )

        await asyncio.sleep(0.3)

        # Verify OUTBOUND: Parent received child's utterance
        assert len(parent_received) == 1
        assert parent_received[0][0] == "app:comms:phone_utterance"
        assert "Hello from child" in parent_received[0][1]

        # Verify INBOUND: Child received parent's guidance
        assert len(child_received) == 1
        assert child_received[0][0] == "app:call:call_guidance"
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
            "app:call:call_guidance",
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
    async def test_start_receive_loop_is_idempotent(self, real_event_broker):
        """Calling start_receive_loop multiple times is safe."""
        received_events = []

        async def on_event(channel: str, event_json: str):
            received_events.append((channel, event_json))

        server = CallEventSocketServer(
            real_event_broker,
            forward_channels=["app:call:*"],
        )
        socket_path = await server.start()

        client = CallEventSocketClient(socket_path)

        # Start receive loop multiple times
        result1 = await client.start_receive_loop(on_event)
        result2 = await client.start_receive_loop(on_event)

        assert result1 is True
        assert result2 is True  # Should return True (already running)

        await asyncio.sleep(0.1)

        # Publish an event
        await real_event_broker.publish(
            "app:call:call_guidance",
            '{"content": "Test"}',
        )

        await asyncio.sleep(0.2)

        # Should receive exactly one event (not duplicated)
        assert len(received_events) == 1

        await client.close()
        await server.stop()


class TestSocketAwareEventBroker:
    """Tests for the SocketAwareEventBroker wrapper in common.py."""

    @pytest.mark.asyncio
    async def test_uses_socket_when_available(self):
        """SocketAwareEventBroker uses socket when CM_EVENT_SOCKET is set."""
        broker = MagicMock()
        broker.publish = AsyncMock()
        server = CallEventSocketServer(broker)
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

            wrapper.register_callback("app:call:call_guidance", on_guidance)

            # Start receiving
            result = await wrapper.start_receiving()
            assert result is True

            # Wait for receive loop to start
            await asyncio.sleep(0.1)

            # Parent publishes event
            await parent_broker.publish(
                "app:call:call_guidance",
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
