# unity/call_common.py

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Awaitable, Callable, Iterable, Optional

from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.events import (
    PhoneCallStarted,
    PhoneCallEnded,
    UnifyMeetEnded,
    UnifyMeetStarted,
)
from unity.conversation_manager.domains.ipc_socket import (
    get_socket_client,
    send_event_to_parent,
    start_socket_receive_loop,
    stop_socket_client,
)
from unity.session_details import SESSION_DETAILS

logger = logging.getLogger(__name__)


class SocketAwareEventBroker:
    """
    Simple event broker for cross-process communication via Unix socket.

    When running as a subprocess (detected via CM_EVENT_SOCKET env var):
    - Outbound: publish() sends events to parent via socket
    - Inbound: register_callback() handlers are invoked when events arrive

    Otherwise, falls back to in-memory broker for outbound events.
    """

    def __init__(self):
        self._socket_client = get_socket_client()
        self._fallback_broker = get_event_broker()
        self._receive_started = False
        self._callbacks: dict[str, Callable[[dict], None]] = {}

    def register_callback(self, channel: str, handler: Callable[[dict], None]) -> None:
        """
        Register a callback for events on a channel.

        The handler is invoked immediately when an event arrives on the channel.
        Handler receives the parsed JSON data (dict).
        """
        self._callbacks[channel] = handler

    async def start_receiving(self) -> bool:
        """
        Start receiving events from the parent process via socket.

        Returns:
            True if started (or already started), False if no socket available.
        """
        if self._receive_started:
            return True

        if not self._socket_client:
            print("[SocketAwareEventBroker] No socket client, receive disabled")
            return False

        async def on_event(channel: str, event_json: str) -> None:
            """Invoke registered callback when event arrives."""
            print(f"[SocketAwareEventBroker] Received: {channel}")
            if channel in self._callbacks:
                try:
                    data = json.loads(event_json)
                    self._callbacks[channel](data)
                except Exception as e:
                    print(f"[SocketAwareEventBroker] Callback error: {e}")

        success = await start_socket_receive_loop(on_event)
        if success:
            self._receive_started = True
            print("[SocketAwareEventBroker] Now receiving events from parent")
        return success

    async def stop(self) -> None:
        """Stop receiving events and close the socket."""
        await stop_socket_client()
        self._receive_started = False

    async def publish(self, channel: str, message: str) -> int:
        """Publish an event, using socket if available."""
        if self._socket_client:
            success = await send_event_to_parent(channel, message)
            if success:
                print(f"[SocketAwareEventBroker] Sent via socket: {channel}")
                return 1
            else:
                print(
                    f"[SocketAwareEventBroker] Socket send failed, using fallback: {channel}",
                )

        # Fall back to in-memory broker (won't work cross-process but useful for testing)
        return await self._fallback_broker.publish(channel, message)


# Shared event broker instance - socket-aware for cross-process communication
event_broker = SocketAwareEventBroker()


async def start_event_broker_receive() -> bool:
    """
    Start receiving events from parent process.

    Call this at the start of call scripts to enable receiving
    inbound events (call_guidance, call_status, etc.) from the parent.
    """
    return await event_broker.start_receiving()


# Default inactivity timeout used by both agents
DEFAULT_INACTIVITY_TIMEOUT = 300  # 5 minutes


# -------- Call lifecycle helpers -------- #


async def publish_call_started(contact: dict, channel: str) -> None:
    event = (
        PhoneCallStarted(contact=contact)
        if channel == "phone"
        else UnifyMeetStarted(contact=contact)
    )
    await event_broker.publish(f"app:comms:{channel}_call_started", event.to_json())


async def publish_call_ended(contact: dict, channel: str) -> None:
    event = (
        PhoneCallEnded(contact=contact)
        if channel == "phone"
        else UnifyMeetEnded(contact=contact)
    )
    await event_broker.publish(f"app:comms:{channel}_call_ended", event.to_json())


def create_end_call(
    contact: dict,
    channel: str,
    pre_shutdown_callback: Optional[Callable[[], None]] = None,
) -> Callable[[], Awaitable[None]]:
    """
    Returns an async function that:
      - calls optional pre_shutdown_callback (e.g., for usage logging)
      - publishes the call ended event
      - cancels all other asyncio tasks

    The process will be terminated by SIGTERM from the parent when cleanup is called.

    Args:
        contact: Contact dictionary for the call.
        channel: Channel type ("phone" or other).
        pre_shutdown_callback: Optional sync callback to run before shutdown.
            Useful for logging call usage/metrics before tasks are cancelled.
    """

    async def end_call() -> None:
        print("Initiating graceful shutdown...")

        # Run pre-shutdown callback (e.g., usage logging) before cleanup
        if pre_shutdown_callback is not None:
            try:
                pre_shutdown_callback()
            except Exception as e:  # noqa: BLE001
                print(f"Error in pre-shutdown callback: {e}")

        # Send end call event before cleaning tasks and closing connection
        await publish_call_ended(contact, channel)
        print("End call event sent")

        # Get all running tasks except current task
        tasks: Iterable[asyncio.Task] = [
            t for t in asyncio.all_tasks() if t is not asyncio.current_task()
        ]

        if tasks:
            print(f"Cancelling {len(tasks)} running tasks...")
            # Cancel all tasks
            for task in tasks:
                task.cancel()

            # Wait for tasks to be cancelled gracefully
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                print("All tasks cancelled successfully")
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001
                print(f"Error during task cancellation: {e}")

        print("Graceful shutdown completed")

    return end_call


def setup_participant_disconnect_handler(room, end_call: Callable[[], Awaitable[None]]):
    """
    Registers a participant_disconnected handler that triggers end_call().
    """

    def on_participant_disconnected(*args, **kwargs):  # noqa: ANN001, ANN002
        asyncio.create_task(end_call())

    room.on("participant_disconnected", on_participant_disconnected)


def setup_inactivity_timeout(
    end_call: Callable[[], Awaitable[None]],
    timeout: float = DEFAULT_INACTIVITY_TIMEOUT,
) -> Callable[[], None]:
    """
    Starts an inactivity watchdog and returns a `touch()` function.

    Call the returned function whenever there is user/assistant activity
    that should reset the inactivity timer.
    """
    loop = asyncio.get_event_loop()
    state = {"last_activity": loop.time()}

    async def check_inactivity():
        while True:
            await asyncio.sleep(10)
            current_time = loop.time()
            if current_time - state["last_activity"] > timeout:
                print("Inactivity timeout reached, shutting down agent...")
                await end_call()
                break

    asyncio.create_task(check_inactivity())

    def touch() -> None:
        state["last_activity"] = loop.time()

    return touch


# -------- CLI / env helpers -------- #


def configure_from_cli(
    extra_env: list[tuple[str, bool]],
) -> str:
    """
    Shared CLI argument handling for both call scripts.

    extra_env: list of (ENV_NAME, is_json) describing additional arguments
               after OUTBOUND that should be stuffed into SESSION_DETAILS.

    Layout (common to both scripts):
      argv[0] = script name
      argv[1] = "dev" | "connect" | "download-files"
      argv[2] = assistant_number
      argv[3] = VOICE_PROVIDER
      argv[4] = VOICE_ID
      argv[5] = OUTBOUND
      argv[6...] = extra_env[...]

    Returns the computed livekit_agent_name ("unity_<assistant_number>").
    """
    assistant_number = ""
    livekit_agent_name = ""
    room_name = ""
    print("sys.argv", sys.argv)

    # max index used = 6 + len(extra_env)
    required_len = 6 + len(extra_env)
    if len(sys.argv) > required_len:
        assistant_number = sys.argv[2]
        if ":" in assistant_number:
            # UnifyMeet: caller passes "livekit_agent_name:room_name" with prefix already applied
            livekit_agent_name, room_name = assistant_number.split(":")
        else:
            # Phone: caller passes raw assistant_number, we add the unity_ prefix
            livekit_agent_name = f"unity_{assistant_number}"
            room_name = livekit_agent_name

        # Populate SESSION_DETAILS with voice config
        SESSION_DETAILS.voice.provider = (
            sys.argv[3] if sys.argv[3] != "None" else "cartesia"
        )
        SESSION_DETAILS.voice.id = sys.argv[4] if sys.argv[4] != "None" else ""
        SESSION_DETAILS.voice_call.outbound = sys.argv[5] == "True"
        SESSION_DETAILS.voice_call.channel = sys.argv[6]

        # Parse extra args (CONTACT, BOSS, ASSISTANT_BIO)
        for idx, (env_name, is_json) in enumerate(extra_env, start=7):
            value = sys.argv[idx]

            if is_json:
                try:
                    loaded = json.loads(value)
                except json.JSONDecodeError:
                    print(f"{env_name} payload is not valid JSON")
                    sys.exit(1)
                if not loaded:
                    print(f"{env_name} payload is invalid (empty)")
                    sys.exit(1)

            # Map known extra args to SESSION_DETAILS fields
            if env_name == "CONTACT":
                SESSION_DETAILS.voice_call.contact_json = value
            elif env_name == "BOSS":
                SESSION_DETAILS.voice_call.boss_json = value
            elif env_name == "ASSISTANT_BIO":
                SESSION_DETAILS.assistant.about = value

        # Export to env for subprocess inheritance
        SESSION_DETAILS.export_to_env()

        # keep only script name and the command ("dev" / "connect" / "download-files")
        sys.argv = sys.argv[:2]
    elif len(sys.argv) > 1 and sys.argv[1] != "download-files":
        print("Not enough arguments provided")
        sys.exit(1)

    return livekit_agent_name, room_name


def should_dispatch_livekit_agent() -> bool:
    """
    True when we should actually call dispatch_livekit_agent() for this process.
    """
    return len(sys.argv) > 1 and sys.argv[1] != "download-files"


# -------- User screen share capture -------- #


class UserScreenCaptureManager:
    """Captures frames from a remote participant's screen share track in a LiveKit room.

    Registers track_subscribed/track_unsubscribed handlers on the room to
    automatically start and stop frame capture when a screen share track
    appears or disappears. Stores the latest frame as raw RGBA bytes and
    converts to base64 PNG on demand (lazy conversion to avoid per-frame cost).

    Usage::

        capture_mgr = UserScreenCaptureManager(ctx.room)
        # ... later, on user utterance ...
        b64 = capture_mgr.capture_screenshot()  # None if no active share
        # ... on cleanup ...
        await capture_mgr.close()
    """

    def __init__(self, room) -> None:
        self._latest_frame_data: tuple[bytes, int, int] | None = None
        self._capture_task: asyncio.Task | None = None
        self._stream = None

        @room.on("track_subscribed")
        def _on_track_subscribed(track, publication, participant):
            self._handle_track_subscribed(track, publication)

        @room.on("track_unsubscribed")
        def _on_track_unsubscribed(track, publication, participant):
            self._handle_track_unsubscribed(publication)

    def _handle_track_subscribed(self, track, publication) -> None:
        from livekit import rtc

        if (
            track.kind == rtc.TrackKind.KIND_VIDEO
            and publication.source == rtc.TrackSource.SOURCE_SCREENSHARE
        ):
            print("[UserScreenCapture] Screen share track subscribed, starting capture")
            stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGBA)
            self._stream = stream
            self._capture_task = asyncio.create_task(self._capture_loop(stream))

    def _handle_track_unsubscribed(self, publication) -> None:
        from livekit import rtc

        if publication.source == rtc.TrackSource.SOURCE_SCREENSHARE:
            print(
                "[UserScreenCapture] Screen share track unsubscribed, stopping capture",
            )
            self._latest_frame_data = None
            if self._capture_task and not self._capture_task.done():
                self._capture_task.cancel()
                self._capture_task = None
            self._stream = None

    async def _capture_loop(self, stream) -> None:
        """Continuously capture frames, rate-limited to 1 per second."""
        import time

        last_capture = 0.0
        try:
            async for frame_event in stream:
                now = time.monotonic()
                if now - last_capture < 1.0:
                    continue
                last_capture = now
                frame = frame_event.frame
                from livekit import rtc

                if frame.type != rtc.VideoBufferType.RGBA:
                    frame = frame.convert(rtc.VideoBufferType.RGBA)
                self._latest_frame_data = (
                    bytes(frame.data),
                    frame.width,
                    frame.height,
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[UserScreenCapture] Frame capture error: {e}")
        finally:
            try:
                await stream.aclose()
            except Exception:
                pass

    def capture_screenshot(self) -> str | None:
        """Convert the latest captured frame to a base64-encoded PNG string.

        Returns None if no screen share track is active or no frame has
        been captured yet.
        """
        if self._latest_frame_data is None:
            return None

        import base64
        import io

        from PIL import Image

        rgba_bytes, width, height = self._latest_frame_data
        img = Image.frombytes("RGBA", (width, height), rgba_bytes, "raw")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    async def close(self) -> None:
        """Cancel the capture loop and release resources."""
        if self._capture_task and not self._capture_task.done():
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
        self._capture_task = None
        self._latest_frame_data = None
        self._stream = None
