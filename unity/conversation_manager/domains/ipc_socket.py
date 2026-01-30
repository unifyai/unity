"""
Unix domain socket IPC for cross-process event communication.

This module provides a socket-based bridge between the voice agent subprocess
(call.py/sts_call.py) and the ConversationManager parent process.

The voice agent runs as a separate process and cannot share the in-memory
event broker with the parent. This socket bridge allows BIDIRECTIONAL event
flow between parent and child processes.

Architecture:
    ConversationManager (parent)          Voice Agent (child)
    ─────────────────────────────         ──────────────────────
    CallEventSocketServer  ◄──────────────  send_event_to_parent()
         │                                        ▲
         │     Unix Domain Socket                 │
         │        (bidirectional)                 │
         │                                        │
         └─► forward_channels ──────────────►  receive loop
                      │                           │
                      ▼                           ▼
              InMemoryEventBroker          local event broker
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Awaitable
from uuid import uuid4

if TYPE_CHECKING:
    from unity.conversation_manager.in_memory_event_broker import InMemoryEventBroker

# Environment variable name for socket path
CM_EVENT_SOCKET_ENV = "CM_EVENT_SOCKET"


class CallEventSocketServer:
    """
    Unix domain socket server for bidirectional event communication with voice agents.

    The server:
    1. Accepts connections from child processes
    2. Receives events from children and publishes to the parent's event broker
    3. Subscribes to specified channels and forwards events to connected clients

    Usage (in CallManager):
        server = CallEventSocketServer(event_broker, forward_channels=["app:call:*"])
        socket_path = await server.start()
        os.environ[CM_EVENT_SOCKET_ENV] = socket_path
        # spawn subprocess...
        # when done:
        await server.stop()
    """

    def __init__(
        self,
        event_broker: "InMemoryEventBroker",
        on_event: Callable[[str, str], Awaitable[None]] | None = None,
        forward_channels: list[str] | None = None,
    ):
        """
        Initialize the socket server.

        Args:
            event_broker: The event broker to publish received events to.
            on_event: Optional callback called with (channel, event_json) for each event.
                     If not provided, events are published directly to event_broker.
            forward_channels: List of channel patterns to forward to connected clients.
                            Supports glob patterns (e.g., "app:call:*").
                            Default: ["app:call:*"] for call guidance and status.
        """
        self._event_broker = event_broker
        self._on_event = on_event
        self._forward_channels = forward_channels or ["app:call:*"]
        self._socket_path: str | None = None
        self._server_socket: socket.socket | None = None
        self._accept_task: asyncio.Task | None = None
        self._forward_task: asyncio.Task | None = None
        self._client_tasks: list[asyncio.Task] = []
        self._connected_clients: list[socket.socket] = []
        self._clients_lock = asyncio.Lock()
        self._running = False

    @property
    def socket_path(self) -> str | None:
        """Return the socket path, or None if not started."""
        return self._socket_path

    async def start(self) -> str:
        """
        Create and bind the Unix domain socket, start accepting connections.

        Returns:
            The socket path (to be passed to subprocess via environment variable).
        """
        if self._running:
            return self._socket_path

        # Create socket in temp directory with unique name
        socket_dir = Path(tempfile.gettempdir())
        self._socket_path = str(socket_dir / f"cm_events_{uuid4().hex[:12]}.sock")

        # Remove stale socket file if exists
        try:
            os.unlink(self._socket_path)
        except FileNotFoundError:
            pass

        # Create and bind socket
        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket.bind(self._socket_path)
        self._server_socket.listen(5)
        self._server_socket.setblocking(False)

        self._running = True
        self._accept_task = asyncio.create_task(self._accept_connections())

        # Start forwarding events to clients
        if self._forward_channels:
            self._forward_task = asyncio.create_task(self._forward_events_to_clients())

        print(f"[CallEventSocketServer] Listening on {self._socket_path}")
        print(f"[CallEventSocketServer] Forwarding channels: {self._forward_channels}")
        return self._socket_path

    async def stop(self) -> None:
        """Stop the server and clean up resources."""
        self._running = False

        # Cancel accept task
        if self._accept_task and not self._accept_task.done():
            self._accept_task.cancel()
            try:
                await self._accept_task
            except asyncio.CancelledError:
                pass

        # Cancel forward task
        if self._forward_task and not self._forward_task.done():
            self._forward_task.cancel()
            try:
                await self._forward_task
            except asyncio.CancelledError:
                pass

        # Cancel all client tasks
        for task in self._client_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._client_tasks.clear()

        # Close all connected clients
        async with self._clients_lock:
            for client in self._connected_clients:
                try:
                    client.close()
                except Exception:
                    pass
            self._connected_clients.clear()

        # Close socket
        if self._server_socket:
            self._server_socket.close()
            self._server_socket = None

        # Remove socket file
        if self._socket_path:
            try:
                os.unlink(self._socket_path)
            except FileNotFoundError:
                pass
            self._socket_path = None

        print("[CallEventSocketServer] Stopped")

    async def _accept_connections(self) -> None:
        """Accept incoming connections from child processes."""
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                client_socket, _ = await asyncio.wait_for(
                    loop.sock_accept(self._server_socket),
                    timeout=0.5,
                )
                print("[CallEventSocketServer] Client connected")

                # Track connected client for forwarding
                async with self._clients_lock:
                    self._connected_clients.append(client_socket)

                task = asyncio.create_task(self._handle_client(client_socket))
                self._client_tasks.append(task)
                # Clean up completed tasks
                self._client_tasks = [t for t in self._client_tasks if not t.done()]
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    print(f"[CallEventSocketServer] Accept error: {e}")
                break

    async def _handle_client(self, client_socket: socket.socket) -> None:
        """Handle a connected client, reading and processing events."""
        loop = asyncio.get_event_loop()
        buffer = b""

        try:
            while self._running:
                try:
                    chunk = await asyncio.wait_for(
                        loop.sock_recv(client_socket, 4096),
                        timeout=1.0,
                    )
                    if not chunk:
                        # Client disconnected
                        break

                    buffer += chunk

                    # Process complete messages (newline-delimited JSON)
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if line:
                            await self._process_message(line.decode("utf-8"))

                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break

        except Exception as e:
            print(f"[CallEventSocketServer] Client handler error: {e}")
        finally:
            # Remove from connected clients list
            async with self._clients_lock:
                if client_socket in self._connected_clients:
                    self._connected_clients.remove(client_socket)
            try:
                client_socket.close()
            except Exception:
                pass
            print("[CallEventSocketServer] Client disconnected")

    async def _process_message(self, message: str) -> None:
        """Process a received message and publish to event broker."""
        try:
            data = json.loads(message)
            channel = data.get("channel", "")
            event_json = data.get("event", "")

            if not channel or not event_json:
                print(
                    f"[CallEventSocketServer] Invalid message format: {message[:100]}",
                )
                return

            print(f"[CallEventSocketServer] Received event on {channel}")

            if self._on_event:
                await self._on_event(channel, event_json)
            else:
                await self._event_broker.publish(channel, event_json)

        except json.JSONDecodeError as e:
            print(f"[CallEventSocketServer] JSON decode error: {e}")
        except Exception as e:
            print(f"[CallEventSocketServer] Error processing message: {e}")

    async def _forward_events_to_clients(self) -> None:
        """Subscribe to forward channels and send matching events to connected clients."""
        try:
            async with self._event_broker.pubsub() as pubsub:
                # Subscribe to all forward channel patterns
                await pubsub.psubscribe(*self._forward_channels)
                print(
                    f"[CallEventSocketServer] Subscribed to forward channels: "
                    f"{self._forward_channels}"
                )

                while self._running:
                    try:
                        msg = await pubsub.get_message(
                            timeout=1.0,
                            ignore_subscribe_messages=True,
                        )
                        if msg is None:
                            continue

                        channel = msg.get("channel", "")
                        data = msg.get("data", "")

                        if channel and data:
                            await self._send_to_all_clients(channel, data)

                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        if self._running:
                            print(f"[CallEventSocketServer] Forward loop error: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[CallEventSocketServer] Forward subscription error: {e}")

    async def _send_to_all_clients(self, channel: str, event_json: str) -> None:
        """Send an event to all connected clients."""
        message = (
            json.dumps(
                {
                    "channel": channel,
                    "event": event_json,
                }
            )
            + "\n"
        )
        message_bytes = message.encode("utf-8")

        loop = asyncio.get_event_loop()
        failed_clients = []

        async with self._clients_lock:
            for client in self._connected_clients:
                try:
                    await loop.sock_sendall(client, message_bytes)
                except Exception as e:
                    print(f"[CallEventSocketServer] Failed to send to client: {e}")
                    failed_clients.append(client)

            # Remove failed clients
            for client in failed_clients:
                if client in self._connected_clients:
                    self._connected_clients.remove(client)
                    try:
                        client.close()
                    except Exception:
                        pass

        if self._connected_clients:
            print(
                f"[CallEventSocketServer] Forwarded {channel} to "
                f"{len(self._connected_clients)} client(s)"
            )


class CallEventSocketClient:
    """
    Client for bidirectional event communication with parent process via Unix socket.

    Supports:
    - Sending events to the parent (e.g., utterances)
    - Receiving events from the parent (e.g., call guidance, status)

    Usage (in call.py):
        client = CallEventSocketClient.from_env()
        if client:
            # Start receiving events (will publish to on_event callback)
            await client.start_receive_loop(on_event_callback)

            # Send events to parent
            await client.send_event("app:comms:phone_utterance", event.to_json())
    """

    def __init__(self, socket_path: str):
        self._socket_path = socket_path
        self._socket: socket.socket | None = None
        self._connected = False
        self._lock = asyncio.Lock()
        self._receive_task: asyncio.Task | None = None
        self._on_event: Callable[[str, str], Awaitable[None]] | None = None
        self._running = False

    @classmethod
    def from_env(cls) -> "CallEventSocketClient | None":
        """
        Create a client from environment variable, or return None if not set.

        This allows call scripts to gracefully handle both socket and non-socket modes.
        """
        socket_path = os.environ.get(CM_EVENT_SOCKET_ENV)
        if socket_path:
            return cls(socket_path)
        return None

    async def connect(self) -> bool:
        """Connect to the socket server. Returns True on success."""
        if self._connected:
            return True

        try:
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.setblocking(False)

            loop = asyncio.get_event_loop()
            await loop.sock_connect(self._socket, self._socket_path)

            self._connected = True
            print(f"[CallEventSocketClient] Connected to {self._socket_path}")
            return True

        except Exception as e:
            print(f"[CallEventSocketClient] Connection failed: {e}")
            if self._socket:
                self._socket.close()
                self._socket = None
            return False

    async def start_receive_loop(
        self,
        on_event: Callable[[str, str], Awaitable[None]],
    ) -> bool:
        """
        Start the background receive loop for inbound events.

        Args:
            on_event: Callback called with (channel, event_json) for each received event.

        Returns:
            True if started successfully, False otherwise.
        """
        if self._receive_task is not None:
            return True  # Already running

        if not self._connected:
            if not await self.connect():
                return False

        self._on_event = on_event
        self._running = True
        self._receive_task = asyncio.create_task(self._receive_loop())
        print("[CallEventSocketClient] Receive loop started")
        return True

    async def _receive_loop(self) -> None:
        """Background loop to receive events from the server."""
        loop = asyncio.get_event_loop()
        buffer = b""

        try:
            while self._running and self._connected:
                try:
                    chunk = await asyncio.wait_for(
                        loop.sock_recv(self._socket, 4096),
                        timeout=1.0,
                    )
                    if not chunk:
                        # Server disconnected
                        print("[CallEventSocketClient] Server disconnected")
                        break

                    buffer += chunk

                    # Process complete messages (newline-delimited JSON)
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if line:
                            await self._process_received_message(line.decode("utf-8"))

                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if self._running:
                        print(f"[CallEventSocketClient] Receive error: {e}")
                    break

        except Exception as e:
            print(f"[CallEventSocketClient] Receive loop error: {e}")
        finally:
            self._running = False
            print("[CallEventSocketClient] Receive loop stopped")

    async def _process_received_message(self, message: str) -> None:
        """Process a message received from the server."""
        try:
            data = json.loads(message)
            channel = data.get("channel", "")
            event_json = data.get("event", "")

            if not channel or not event_json:
                print(
                    f"[CallEventSocketClient] Invalid message format: {message[:100]}"
                )
                return

            print(f"[CallEventSocketClient] Received event on {channel}")

            if self._on_event:
                await self._on_event(channel, event_json)

        except json.JSONDecodeError as e:
            print(f"[CallEventSocketClient] JSON decode error: {e}")
        except Exception as e:
            print(f"[CallEventSocketClient] Error processing message: {e}")

    async def send_event(self, channel: str, event_json: str) -> bool:
        """
        Send an event to the parent process.

        Args:
            channel: The event channel (e.g., "app:comms:phone_utterance")
            event_json: The JSON-encoded event

        Returns:
            True if sent successfully, False otherwise.
        """
        async with self._lock:
            if not self._connected:
                if not await self.connect():
                    return False

            try:
                message = (
                    json.dumps(
                        {
                            "channel": channel,
                            "event": event_json,
                        },
                    )
                    + "\n"
                )

                loop = asyncio.get_event_loop()
                await loop.sock_sendall(self._socket, message.encode("utf-8"))
                return True

            except Exception as e:
                print(f"[CallEventSocketClient] Send failed: {e}")
                self._connected = False
                if self._socket:
                    self._socket.close()
                    self._socket = None
                return False

    async def stop(self) -> None:
        """Stop the receive loop."""
        self._running = False
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        self._receive_task = None

    async def close(self) -> None:
        """Close the connection."""
        await self.stop()
        if self._socket:
            self._socket.close()
            self._socket = None
        self._connected = False


# Singleton client for use in call scripts
_socket_client: CallEventSocketClient | None = None
_receive_loop_started: bool = False


def get_socket_client() -> CallEventSocketClient | None:
    """Get or create the singleton socket client."""
    global _socket_client
    if _socket_client is None:
        _socket_client = CallEventSocketClient.from_env()
    return _socket_client


async def send_event_to_parent(channel: str, event_json: str) -> bool:
    """
    Convenience function to send an event to the parent process.

    This is the primary API for call scripts to use.

    Args:
        channel: The event channel
        event_json: The JSON-encoded event

    Returns:
        True if sent via socket, False if socket not available (caller should
        fall back to in-memory broker if needed).
    """
    client = get_socket_client()
    if client:
        return await client.send_event(channel, event_json)
    return False


async def start_socket_receive_loop(
    on_event: Callable[[str, str], Awaitable[None]],
) -> bool:
    """
    Start receiving events from the parent process.

    This should be called once at startup in call scripts to enable
    receiving inbound events (call guidance, status, etc.).

    Args:
        on_event: Callback called with (channel, event_json) for each event.

    Returns:
        True if started (or already started), False if socket not available.
    """
    global _receive_loop_started
    if _receive_loop_started:
        return True

    client = get_socket_client()
    if client:
        success = await client.start_receive_loop(on_event)
        if success:
            _receive_loop_started = True
        return success
    return False


async def stop_socket_client() -> None:
    """Stop the socket client and receive loop."""
    global _socket_client, _receive_loop_started
    if _socket_client:
        await _socket_client.close()
        _socket_client = None
    _receive_loop_started = False
