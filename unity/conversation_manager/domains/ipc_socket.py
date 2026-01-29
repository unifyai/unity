"""
Unix domain socket IPC for cross-process event communication.

This module provides a socket-based bridge between the voice agent subprocess
(call.py/sts_call.py) and the ConversationManager parent process.

The voice agent runs as a separate process and cannot share the in-memory
event broker with the parent. This socket bridge allows events (like call
transcripts) to flow from the child process back to the parent, where they
are published to the in-memory event broker.

Architecture:
    ConversationManager (parent)          Voice Agent (child)
    ─────────────────────────────         ──────────────────────
    CallEventSocketServer                  send_event_to_parent()
         │                                        │
         │  Unix Domain Socket                    │
         └────────────────────────────────────────┘
                      │
                      ▼
              InMemoryEventBroker
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
    Unix domain socket server for receiving events from voice agent subprocess.

    The server accepts connections from the child process and reads JSON-encoded
    events, then publishes them to the in-memory event broker.

    Usage (in CallManager):
        server = CallEventSocketServer(event_broker)
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
    ):
        """
        Initialize the socket server.

        Args:
            event_broker: The event broker to publish received events to.
            on_event: Optional callback called with (channel, event_json) for each event.
                     If not provided, events are published directly to event_broker.
        """
        self._event_broker = event_broker
        self._on_event = on_event
        self._socket_path: str | None = None
        self._server_socket: socket.socket | None = None
        self._accept_task: asyncio.Task | None = None
        self._client_tasks: list[asyncio.Task] = []
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

        print(f"[CallEventSocketServer] Listening on {self._socket_path}")
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

        # Cancel all client tasks
        for task in self._client_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._client_tasks.clear()

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
            client_socket.close()
            print("[CallEventSocketServer] Client disconnected")

    async def _process_message(self, message: str) -> None:
        """Process a received message and publish to event broker."""
        try:
            data = json.loads(message)
            channel = data.get("channel", "")
            event_json = data.get("event", "")

            if not channel or not event_json:
                print(
                    f"[CallEventSocketServer] Invalid message format: {message[:100]}"
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


class CallEventSocketClient:
    """
    Client for sending events to the parent process via Unix domain socket.

    Usage (in call.py):
        client = CallEventSocketClient.from_env()
        if client:
            await client.send_event("app:comms:phone_utterance", event.to_json())
        else:
            # Fall back to in-memory broker (won't work cross-process)
            await event_broker.publish(...)
    """

    def __init__(self, socket_path: str):
        self._socket_path = socket_path
        self._socket: socket.socket | None = None
        self._connected = False
        self._lock = asyncio.Lock()

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
                        }
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

    async def close(self) -> None:
        """Close the connection."""
        if self._socket:
            self._socket.close()
            self._socket = None
        self._connected = False


# Singleton client for use in call scripts
_socket_client: CallEventSocketClient | None = None


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
