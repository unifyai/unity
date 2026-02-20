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

    The server runs all socket I/O in a dedicated thread with its own event
    loop, so reads/writes are never blocked by work on the main loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Awaitable
from uuid import uuid4

from unity.conversation_manager.tracing import (
    monotonic_ms,
    now_utc_iso,
    payload_trace_id,
    trace_kv,
)

_log = logging.getLogger("unity")

if TYPE_CHECKING:
    from unity.conversation_manager.in_memory_event_broker import InMemoryEventBroker

CM_EVENT_SOCKET_ENV = "CM_EVENT_SOCKET"


class CallEventSocketServer:
    """
    Unix domain socket server for bidirectional event communication with voice agents.

    All socket I/O (accept, recv, send) runs in a **dedicated daemon thread**
    with its own asyncio event loop.  This guarantees that reads from child
    processes are never delayed by work on the main ConversationManager loop
    (HTTP requests, LLM inference, etc.).

    Cross-thread dispatch:
      - child → parent:  message received in I/O thread, dispatched to main
                          loop via ``run_coroutine_threadsafe``.
      - parent → child:  forward subscription runs in main loop, dispatches
                          ``sock_sendall`` to the I/O loop.
    """

    def __init__(
        self,
        event_broker: "InMemoryEventBroker",
        on_event: Callable[[str, str], Awaitable[None]] | None = None,
        forward_channels: list[str] | None = None,
    ):
        self._event_broker = event_broker
        self._on_event = on_event
        self._forward_channels = (
            forward_channels if forward_channels is not None else ["app:call:*"]
        )
        self._socket_path: str | None = None
        self._server_socket: socket.socket | None = None

        # Dedicated I/O thread and its event loop
        self._io_thread: threading.Thread | None = None
        self._io_loop: asyncio.AbstractEventLoop | None = None
        self._io_ready = threading.Event()

        # Main event loop (captured in start())
        self._main_loop: asyncio.AbstractEventLoop | None = None

        # Managed exclusively from the I/O loop (no locks needed)
        self._client_tasks: list[asyncio.Task] = []
        self._connected_clients: list[socket.socket] = []
        self._pending_messages: list[tuple[str, str]] = []

        # Forward subscription (runs in main loop)
        self._forward_task: asyncio.Task | None = None
        self._forward_ready = asyncio.Event()

        self._running = False
        self.on_client_disconnected: Callable[[], Awaitable[None]] | None = None

    @property
    def socket_path(self) -> str | None:
        """Return the socket path, or None if not started."""
        return self._socket_path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> str:
        """Create the socket, start the I/O thread and forward subscription.

        Returns the socket path (to be passed to subprocess via env var).
        """
        if self._running:
            return self._socket_path

        self._main_loop = asyncio.get_running_loop()

        # Create socket in temp directory with unique name
        socket_dir = Path(tempfile.gettempdir())
        self._socket_path = str(socket_dir / f"cm_events_{uuid4().hex[:12]}.sock")

        try:
            os.unlink(self._socket_path)
        except FileNotFoundError:
            pass

        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket.bind(self._socket_path)
        self._server_socket.listen(5)
        self._server_socket.setblocking(False)

        self._running = True

        # Start the dedicated I/O thread
        self._io_ready.clear()
        self._io_thread = threading.Thread(
            target=self._run_io_loop,
            daemon=True,
            name="ipc-socket-server",
        )
        self._io_thread.start()
        self._io_ready.wait()

        # Schedule the accept loop inside the I/O thread
        asyncio.run_coroutine_threadsafe(
            self._accept_connections(),
            self._io_loop,
        )

        # Start forwarding broker events to clients (runs in main loop)
        self._forward_ready.clear()
        if self._forward_channels:
            self._forward_task = asyncio.create_task(self._forward_events_to_clients())
            await self._forward_ready.wait()

        _log.info(
            "[CallEventSocketServer] Listening on %s, forward_channels=%s",
            self._socket_path,
            self._forward_channels,
        )
        print(f"[CallEventSocketServer] Listening on {self._socket_path}")
        print(f"[CallEventSocketServer] Forwarding channels: {self._forward_channels}")
        return self._socket_path

    async def stop(self) -> None:
        """Stop the server, I/O thread, and clean up resources."""
        self._running = False

        # 1. Cancel the forward subscription (main loop)
        if self._forward_task and not self._forward_task.done():
            self._forward_task.cancel()
            try:
                await self._forward_task
            except asyncio.CancelledError:
                pass

        # 2. Shut down the I/O thread
        if self._io_loop and self._io_loop.is_running():
            shutdown_future = asyncio.run_coroutine_threadsafe(
                self._shutdown_io(),
                self._io_loop,
            )
            try:
                await asyncio.wait_for(
                    asyncio.wrap_future(shutdown_future),
                    timeout=5.0,
                )
            except (asyncio.TimeoutError, Exception) as e:
                print(f"[CallEventSocketServer] I/O shutdown issue: {e}")
                self._io_loop.call_soon_threadsafe(self._io_loop.stop)

        if self._io_thread and self._io_thread.is_alive():
            self._io_thread.join(timeout=5)
        self._io_thread = None
        self._io_loop = None

        # 3. Remove socket file
        if self._socket_path:
            try:
                os.unlink(self._socket_path)
            except FileNotFoundError:
                pass
            self._socket_path = None

        print("[CallEventSocketServer] Stopped")

    async def set_forward_channels(self, channels: list[str]) -> None:
        """Update the forwarded channel patterns, restarting the subscription."""
        if channels == self._forward_channels:
            return
        self._forward_channels = channels
        if not self._running:
            return
        if self._forward_task and not self._forward_task.done():
            self._forward_task.cancel()
            try:
                await self._forward_task
            except asyncio.CancelledError:
                pass
        self._forward_ready.clear()
        if self._forward_channels:
            self._forward_task = asyncio.create_task(self._forward_events_to_clients())
            await self._forward_ready.wait()
        _log.info(
            "[CallEventSocketServer] Forward channels updated to %s",
            self._forward_channels,
        )
        print(
            f"[CallEventSocketServer] Forward channels updated: {self._forward_channels}",
        )

    # ------------------------------------------------------------------
    # I/O thread
    # ------------------------------------------------------------------

    def _run_io_loop(self) -> None:
        """Thread target: create and run the dedicated event loop."""
        try:
            self._io_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._io_loop)
            self._io_ready.set()
            self._io_loop.run_forever()
        except Exception as e:
            print(f"[CallEventSocketServer] I/O loop error: {e}")
        finally:
            if self._io_loop and not self._io_loop.is_closed():
                self._io_loop.close()
            self._io_ready.set()

    async def _accept_connections(self) -> None:
        """Runs in I/O loop. Accepts incoming client connections."""
        loop = asyncio.get_running_loop()

        while self._running:
            try:
                client_socket, _ = await asyncio.wait_for(
                    loop.sock_accept(self._server_socket),
                    timeout=0.5,
                )
                _log.info("[CallEventSocketServer] Client connected")
                print("[CallEventSocketServer] Client connected")

                self._connected_clients.append(client_socket)

                # Flush any messages buffered before a client connected
                if self._pending_messages:
                    print(
                        f"[CallEventSocketServer] Flushing "
                        f"{len(self._pending_messages)} buffered message(s)",
                    )
                    for channel, event_json in self._pending_messages:
                        message_id = payload_trace_id(
                            "ipc",
                            str(channel),
                            str(event_json),
                        )
                        try:
                            msg = (
                                json.dumps(
                                    {"channel": channel, "event": event_json},
                                )
                                + "\n"
                            )
                            await loop.sock_sendall(
                                client_socket,
                                msg.encode("utf-8"),
                            )
                            print(
                                trace_kv(
                                    "IPC_SERVER_FLUSH_BUFFERED",
                                    channel=channel,
                                    message_id=message_id,
                                    ts_utc=now_utc_iso(),
                                    monotonic_ms=monotonic_ms(),
                                ),
                                flush=True,
                            )
                        except Exception as e:
                            print(
                                f"[CallEventSocketServer] Failed to flush "
                                f"buffered message: {e}",
                            )
                    self._pending_messages.clear()

                task = asyncio.create_task(self._handle_client(client_socket))
                self._client_tasks.append(task)
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
        """Runs in I/O loop. Reads messages from a connected client."""
        loop = asyncio.get_running_loop()
        buffer = b""

        try:
            while self._running:
                try:
                    chunk = await asyncio.wait_for(
                        loop.sock_recv(client_socket, 4096),
                        timeout=1.0,
                    )
                    if not chunk:
                        break

                    buffer += chunk

                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if line:
                            self._dispatch_to_main_loop(line.decode("utf-8"))

                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break

        except Exception as e:
            print(f"[CallEventSocketServer] Client handler error: {e}")
        finally:
            if client_socket in self._connected_clients:
                self._connected_clients.remove(client_socket)
            no_clients_left = len(self._connected_clients) == 0
            try:
                client_socket.close()
            except Exception:
                pass
            print("[CallEventSocketServer] Client disconnected")
            if (
                no_clients_left
                and self.on_client_disconnected is not None
                and self._main_loop
                and self._main_loop.is_running()
            ):
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.on_client_disconnected(),
                        self._main_loop,
                    )
                except Exception as e:
                    print(
                        f"[CallEventSocketServer] on_client_disconnected "
                        f"dispatch error: {e}",
                    )

    def _dispatch_to_main_loop(self, message: str) -> None:
        """Schedule message processing on the main event loop (thread-safe)."""
        if self._main_loop and self._main_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._process_message(message),
                self._main_loop,
            )

    async def _send_to_all_clients(self, channel: str, event_json: str) -> None:
        """Runs in I/O loop. Sends an event to all connected clients.

        If no clients are connected, the message is buffered.
        """
        message_id = payload_trace_id("ipc", str(channel), str(event_json))
        message = json.dumps({"channel": channel, "event": event_json}) + "\n"
        message_bytes = message.encode("utf-8")

        loop = asyncio.get_running_loop()

        if not self._connected_clients:
            self._pending_messages.append((channel, event_json))
            _log.debug(
                "[CallEventSocketServer] Buffered (no clients): channel=%s "
                "message_id=%s buffered_count=%d",
                channel,
                message_id,
                len(self._pending_messages),
            )
            print(
                trace_kv(
                    "IPC_SERVER_BUFFER",
                    channel=channel,
                    message_id=message_id,
                    buffered_count=len(self._pending_messages),
                    ts_utc=now_utc_iso(),
                    monotonic_ms=monotonic_ms(),
                ),
                flush=True,
            )
            return

        for client in self._connected_clients:
            try:
                await loop.sock_sendall(client, message_bytes)
            except Exception as e:
                _log.warning(
                    "[CallEventSocketServer] sock_sendall failed " "(client kept): %s",
                    e,
                )
                print(f"[CallEventSocketServer] Failed to send to client: {e}")

        if self._connected_clients:
            _log.debug(
                "[CallEventSocketServer] Sent: channel=%s message_id=%s "
                "client_count=%d",
                channel,
                message_id,
                len(self._connected_clients),
            )
            print(
                trace_kv(
                    "IPC_SERVER_FORWARD",
                    channel=channel,
                    message_id=message_id,
                    client_count=len(self._connected_clients),
                    ts_utc=now_utc_iso(),
                    monotonic_ms=monotonic_ms(),
                ),
                flush=True,
            )

    async def _shutdown_io(self) -> None:
        """Runs in I/O loop. Cleanly shuts down all I/O resources."""
        # Close client sockets (interrupts any pending sock_recv)
        for client in list(self._connected_clients):
            try:
                client.close()
            except Exception:
                pass

        # Cancel and await all client tasks so finally blocks run
        for task in self._client_tasks:
            if not task.done():
                task.cancel()
        if self._client_tasks:
            await asyncio.gather(*self._client_tasks, return_exceptions=True)
        self._client_tasks.clear()
        self._connected_clients.clear()
        self._pending_messages.clear()

        # Close server socket
        if self._server_socket:
            self._server_socket.close()
            self._server_socket = None

        # Stop the I/O event loop (takes effect after this coroutine returns)
        self._io_loop.call_soon(self._io_loop.stop)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _process_message(self, message: str) -> None:
        """Runs in main loop. Processes a message received from a client."""
        try:
            data = json.loads(message)
            channel = data.get("channel", "")
            event_json = data.get("event", "")

            if not channel or not event_json:
                print(
                    f"[CallEventSocketServer] Invalid message format: {message[:100]}",
                )
                return

            message_id = payload_trace_id("ipc", channel, event_json)
            print(
                trace_kv(
                    "IPC_SERVER_INBOUND",
                    channel=channel,
                    message_id=message_id,
                    ts_utc=now_utc_iso(),
                    monotonic_ms=monotonic_ms(),
                ),
                flush=True,
            )

            if self._on_event:
                await self._on_event(channel, event_json)
            else:
                await self._event_broker.publish(channel, event_json)

        except json.JSONDecodeError as e:
            print(f"[CallEventSocketServer] JSON decode error: {e}")
        except Exception as e:
            print(f"[CallEventSocketServer] Error processing message: {e}")

    async def queue_for_clients(self, channel: str, event_json: str) -> None:
        """Directly queue a message for delivery to connected (or future) clients.

        Called from the main loop; dispatches the send to the I/O loop.
        """
        if self._io_loop and self._io_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self._send_to_all_clients(channel, event_json),
                self._io_loop,
            )
            await asyncio.wrap_future(future)

    async def _forward_events_to_clients(self) -> None:
        """Runs in main loop. Subscribes to broker and forwards to clients."""
        try:
            async with self._event_broker.pubsub() as pubsub:
                await pubsub.psubscribe(*self._forward_channels)
                self._forward_ready.set()
                _log.info(
                    "[CallEventSocketServer] Forward subscription active, channels=%s",
                    self._forward_channels,
                )
                print(
                    f"[CallEventSocketServer] Subscribed to forward channels: "
                    f"{self._forward_channels}",
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
                            _log.debug(
                                "[CallEventSocketServer] Forward: channel=%s "
                                "clients=%d data_len=%d",
                                channel,
                                len(self._connected_clients),
                                len(data),
                            )
                            # Dispatch the send to the I/O loop (fire-and-forget)
                            if self._io_loop and self._io_loop.is_running():
                                asyncio.run_coroutine_threadsafe(
                                    self._send_to_all_clients(channel, data),
                                    self._io_loop,
                                )

                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        if self._running:
                            _log.warning(
                                "[CallEventSocketServer] Forward loop error: %s",
                                e,
                            )
                            print(
                                f"[CallEventSocketServer] Forward loop error: {e}",
                            )

        except asyncio.CancelledError:
            self._forward_ready.set()
        except Exception as e:
            self._forward_ready.set()
            _log.error(
                "[CallEventSocketServer] Forward subscription error: %s",
                e,
            )
            print(f"[CallEventSocketServer] Forward subscription error: {e}")


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
        reconnect_attempts = 0
        max_reconnect_attempts = 3
        iteration = 0
        recv_count = 0
        timeout_count = 0

        print(
            f"[CallEventSocketClient] _receive_loop ENTERED "
            f"loop_id={id(loop)} socket_fd={self._socket.fileno() if self._socket else 'None'} "
            f"connected={self._connected} running={self._running}",
            flush=True,
        )

        try:
            while self._running:
                iteration += 1

                if not self._connected:
                    if reconnect_attempts >= max_reconnect_attempts:
                        print(
                            f"[CallEventSocketClient] Max reconnect attempts "
                            f"({max_reconnect_attempts}) reached, stopping",
                        )
                        break
                    reconnect_attempts += 1
                    print(
                        f"[CallEventSocketClient] Attempting reconnect "
                        f"({reconnect_attempts}/{max_reconnect_attempts})...",
                    )
                    await asyncio.sleep(0.5)
                    if not await self.connect():
                        continue
                    buffer = b""
                    print("[CallEventSocketClient] Reconnected successfully")
                    reconnect_attempts = 0

                try:
                    chunk = await asyncio.wait_for(
                        loop.sock_recv(self._socket, 4096),
                        timeout=1.0,
                    )
                    if not chunk:
                        print("[CallEventSocketClient] Server disconnected")
                        self._connected = False
                        continue

                    recv_count += 1
                    print(
                        f"[CallEventSocketClient] sock_recv: "
                        f"bytes={len(chunk)} iteration={iteration} "
                        f"total_recv={recv_count}",
                        flush=True,
                    )

                    buffer += chunk

                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if line:
                            await self._process_received_message(line.decode("utf-8"))

                except asyncio.TimeoutError:
                    timeout_count += 1
                    if timeout_count % 30 == 1:
                        print(
                            f"[CallEventSocketClient] heartbeat: "
                            f"iteration={iteration} timeouts={timeout_count} "
                            f"recv={recv_count} connected={self._connected} "
                            f"running={self._running} "
                            f"socket_fd={self._socket.fileno() if self._socket else 'None'}",
                            flush=True,
                        )
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if self._running:
                        print(
                            f"[CallEventSocketClient] Receive error: {e} "
                            f"iteration={iteration}",
                            flush=True,
                        )
                        self._connected = False
                        if self._socket:
                            try:
                                self._socket.close()
                            except Exception:
                                pass
                            self._socket = None
                    continue

        except Exception as e:
            print(f"[CallEventSocketClient] Receive loop error: {e}")
        finally:
            self._running = False
            print(
                f"[CallEventSocketClient] Receive loop stopped "
                f"iterations={iteration} recv={recv_count} "
                f"timeouts={timeout_count}",
                flush=True,
            )

    async def _process_received_message(self, message: str) -> None:
        """Process a message received from the server."""
        try:
            data = json.loads(message)
            channel = data.get("channel", "")
            event_json = data.get("event", "")

            if not channel or not event_json:
                print(
                    f"[CallEventSocketClient] Invalid message format: {message[:100]}",
                )
                return

            message_id = payload_trace_id("ipc", channel, event_json)
            print(
                trace_kv(
                    "IPC_CLIENT_INBOUND",
                    channel=channel,
                    message_id=message_id,
                    ts_utc=now_utc_iso(),
                    monotonic_ms=monotonic_ms(),
                ),
                flush=True,
            )

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
            return await self._send_event_impl(channel, event_json, retry=True)

    async def _send_event_impl(
        self,
        channel: str,
        event_json: str,
        retry: bool = True,
    ) -> bool:
        """Internal send implementation with optional retry."""
        if not self._connected:
            if not await self.connect():
                return False

        try:
            message_id = payload_trace_id("ipc", channel, event_json)
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
            print(
                trace_kv(
                    "IPC_CLIENT_OUTBOUND",
                    channel=channel,
                    message_id=message_id,
                    retry=retry,
                    ts_utc=now_utc_iso(),
                    monotonic_ms=monotonic_ms(),
                ),
                flush=True,
            )
            return True

        except Exception as e:
            print(f"[CallEventSocketClient] Send failed: {e}")
            self._connected = False
            if self._socket:
                self._socket.close()
                self._socket = None

            # Try to reconnect and retry once
            if retry:
                print("[CallEventSocketClient] Attempting reconnect...")
                if await self.connect():
                    # Restart receive loop if it was running
                    if self._on_event and (
                        self._receive_task is None or self._receive_task.done()
                    ):
                        self._running = True
                        self._receive_task = asyncio.create_task(self._receive_loop())
                        print("[CallEventSocketClient] Receive loop restarted")
                    return await self._send_event_impl(channel, event_json, retry=False)

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
