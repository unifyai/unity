# unity/call_common.py

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import sys
from typing import Awaitable, Callable, Iterable, Optional

from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.events import (
    Event,
    PhoneCallStarted,
    PhoneCallEnded,
    UnifyMeetEnded,
    UnifyMeetStarted,
    SMSReceived,
    EmailReceived,
    UnifyMessageReceived,
    ActorNotification,
    ActorResult,
    ActorHandleStarted,
    ActorSessionResponse,
    NotificationInjectedEvent,
    CallGuidance,
)
from unity.conversation_manager.domains.ipc_socket import (
    get_socket_client,
    send_event_to_parent,
    start_socket_receive_loop,
    stop_socket_client,
)
from unity.conversation_manager.tracing import (
    monotonic_ms,
    now_utc_iso,
    payload_trace_id,
    trace_kv,
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
        self._pattern_callbacks: list[tuple[str, Callable[[dict], None]]] = []

    def register_callback(self, channel: str, handler: Callable[[dict], None]) -> None:
        """
        Register a callback for events on a channel.

        Supports glob patterns (e.g., ``app:comms:*``).  Exact-match
        callbacks take precedence over patterns.
        """
        if "*" in channel or "?" in channel or "[" in channel:
            self._pattern_callbacks.append((channel, handler))
        else:
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
            message_id = payload_trace_id("ipc", channel, event_json)
            has_exact = channel in self._callbacks
            has_pattern = any(
                fnmatch.fnmatch(channel, pat) for pat, _ in self._pattern_callbacks
            )
            print(
                trace_kv(
                    "SOCKET_BROKER_INBOUND",
                    channel=channel,
                    message_id=message_id,
                    has_callback=has_exact or has_pattern,
                    ts_utc=now_utc_iso(),
                    monotonic_ms=monotonic_ms(),
                ),
                flush=True,
            )
            try:
                data = json.loads(event_json)
            except Exception as e:
                print(f"[SocketAwareEventBroker] JSON parse error: {e}")
                return
            if has_exact:
                try:
                    self._callbacks[channel](data)
                except Exception as e:
                    print(f"[SocketAwareEventBroker] Callback error: {e}")
            for pat, handler in self._pattern_callbacks:
                if fnmatch.fnmatch(channel, pat):
                    try:
                        handler(data)
                    except Exception as e:
                        print(f"[SocketAwareEventBroker] Pattern callback error: {e}")

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
        message_id = payload_trace_id("ipc", channel, message)
        if self._socket_client:
            success = await send_event_to_parent(channel, message)
            if success:
                print(
                    trace_kv(
                        "SOCKET_BROKER_OUTBOUND",
                        channel=channel,
                        message_id=message_id,
                        via_socket=True,
                        ts_utc=now_utc_iso(),
                        monotonic_ms=monotonic_ms(),
                    ),
                    flush=True,
                )
                return 1
            else:
                print(
                    f"[SocketAwareEventBroker] Socket send failed, using fallback: {channel}",
                )

        # Fall back to in-memory broker (won't work cross-process but useful for testing)
        print(
            trace_kv(
                "SOCKET_BROKER_OUTBOUND",
                channel=channel,
                message_id=message_id,
                via_socket=False,
                ts_utc=now_utc_iso(),
                monotonic_ms=monotonic_ms(),
            ),
            flush=True,
        )
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


# -------- Say-meta matching -------- #


def match_say_meta(
    meta: dict | None,
    utterance_text: str,
) -> dict | None:
    """Match a _last_say_meta dict against an utterance's text content.

    Returns the meta dict if it should be consumed for this utterance,
    or None if the utterance didn't originate from the session.say() call
    that set the meta.

    When meta includes a "text" key (set by maybe_speak_queued), the
    utterance must start with the same prefix to match — preventing a
    fast brain response from stealing metadata intended for a session.say()
    utterance. Meta dicts without a "text" key match unconditionally
    (backward compatibility).
    """
    if meta is None:
        return None
    expected_text = meta.get("text")
    if expected_text is None:
        return meta
    if not utterance_text or not expected_text:
        return None
    prefix_len = min(50, len(expected_text), len(utterance_text))
    if utterance_text[:prefix_len] == expected_text[:prefix_len]:
        return meta
    return None


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

    Returns the canonical room name passed as argv[2] (produced by
    make_room_name() in call_manager). This is used as both the LiveKit
    room name and the agent worker registration name.
    """
    room_name = ""
    print("sys.argv", sys.argv)

    # max index used = 6 + len(extra_env)
    required_len = 6 + len(extra_env)
    if len(sys.argv) > required_len:
        room_name = sys.argv[2]

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
            elif env_name == "ASSISTANT_ID":
                SESSION_DETAILS.assistant.id = value
            elif env_name == "USER_ID":
                SESSION_DETAILS.user.id = value

        # Export to env for subprocess inheritance
        SESSION_DETAILS.export_to_env()

        # keep only script name and the command ("dev" / "connect" / "download-files")
        sys.argv = sys.argv[:2]
    elif len(sys.argv) > 1 and sys.argv[1] != "download-files":
        print("Not enough arguments provided")
        sys.exit(1)

    return room_name


def should_dispatch_livekit_agent() -> bool:
    """
    True when we should actually call dispatch_livekit_agent() for this process.
    """
    return len(sys.argv) > 1 and sys.argv[1] != "download-files"


# -------- User screen share capture -------- #


class UserTrackCaptureManager:
    """Captures frames from a remote participant's video track in a LiveKit room.

    Registers track_subscribed/track_unsubscribed handlers on the room to
    automatically start and stop frame capture when a matching video track
    appears or disappears. Stores the latest frame as raw RGBA bytes and
    converts to base64 JPEG on demand (lazy conversion to avoid per-frame cost).

    The ``track_source`` parameter selects which LiveKit track source to
    capture (e.g. ``SOURCE_SCREENSHARE`` for screen share, ``SOURCE_CAMERA``
    for webcam).

    Usage::

        screen_mgr = UserTrackCaptureManager(ctx.room)  # screen share (default)
        webcam_mgr = UserTrackCaptureManager(ctx.room, track_source="camera")
        # ... later, on user utterance ...
        b64 = screen_mgr.capture_screenshot()  # None if no active share
        # ... on cleanup ...
        await screen_mgr.close()
    """

    def __init__(self, room, *, track_source: str = "screenshare") -> None:
        from livekit import rtc

        self._latest_frame_data: tuple[bytes, int, int] | None = None
        self._capture_task: asyncio.Task | None = None
        self._stream = None

        source_map = {
            "screenshare": rtc.TrackSource.SOURCE_SCREENSHARE,
            "camera": rtc.TrackSource.SOURCE_CAMERA,
        }
        self._rtc_source = source_map[track_source]
        self._label = track_source

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
            and publication.source == self._rtc_source
        ):
            print(
                f"[UserTrackCapture:{self._label}] Track subscribed, starting capture",
            )
            stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGBA)
            self._stream = stream
            self._capture_task = asyncio.create_task(self._capture_loop(stream))

    def _handle_track_unsubscribed(self, publication) -> None:
        if publication.source == self._rtc_source:
            print(
                f"[UserTrackCapture:{self._label}] Track unsubscribed, stopping capture",
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
        """Convert the latest captured frame to a base64-encoded JPEG string.

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
        rgb = img.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=85)
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


# Backward-compatible alias used by call.py / sts_call.py imports.
UserScreenCaptureManager = UserTrackCaptureManager


# -------- Screenshot history for fast brain visual context -------- #


class ScreenshotHistory:
    """Per-source screenshot history for the fast brain LLM.

    Tracks captured screenshots and builds a visual context message with the
    latest screenshot from each source (user / assistant) as an inline image
    and all older entries as filepath-only text references.
    """

    def __init__(self):
        self._entries: list[tuple["ScreenshotEntry", str]] = []

    def add(self, entry: "ScreenshotEntry", filepath: str) -> None:
        self._entries.append((entry, filepath))

    def build_visual_context_content(self) -> list:
        """Build a content list for a visual context chat message.

        Returns ``list[str | ImageContent]``: for each source the most recent
        entry gets a ``str`` label **plus** an ``ImageContent`` block; all
        older entries from that source get only a ``str`` filepath label.
        """
        from livekit.agents.llm import ImageContent

        if not self._entries:
            return []

        latest_idx_by_source: dict[str, int] = {}
        for i, (entry, _) in enumerate(self._entries):
            latest_idx_by_source[entry.source] = i

        source_labels = {
            "assistant": "Assistant's Screen",
            "user": "User's Screen",
            "webcam": "User's Webcam",
        }

        parts: list = []
        for i, (entry, filepath) in enumerate(self._entries):
            label = source_labels.get(entry.source, "Screenshot")
            text = (
                f"[{label} at {entry.timestamp.strftime('%H:%M:%S')} "
                f"-- {filepath}] "
                f'User said: "{entry.utterance}"'
            )
            parts.append(text)
            if i == latest_idx_by_source.get(entry.source):
                parts.append(
                    ImageContent(
                        image=f"data:image/jpeg;base64,{entry.b64}",
                    ),
                )

        return parts


def _resolve_agent_service_url() -> str:
    """Resolve the agent-service base URL, matching ComputerPrimitives conventions.

    Managed VMs expose the agent-service behind a reverse proxy at ``/api``,
    while local dev hits the service directly on port 3000.
    """
    from unity.session_details import SESSION_DETAILS

    desktop_url = SESSION_DETAILS.assistant.desktop_url
    if desktop_url:
        return desktop_url.rstrip("/") + "/api"
    return "http://localhost:3000"


async def capture_assistant_screenshot(utterance: str) -> "ScreenshotEntry | None":
    """Capture the assistant's desktop via HTTP POST.

    Returns a ``ScreenshotEntry`` on success, ``None`` on failure or if no
    desktop URL is configured.
    """
    import aiohttp

    from datetime import datetime, timezone
    from unity.session_details import SESSION_DETAILS
    from unity.conversation_manager.types.screenshot import ScreenshotEntry

    base_url = _resolve_agent_service_url()
    auth_key = SESSION_DETAILS.unify_key
    session_id = os.environ.get("AGENT_SERVICE_SESSION_ID", "")
    payload = {"sessionId": session_id} if session_id else {}
    try:
        headers = {"authorization": f"Bearer {auth_key}"}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/screenshot",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status >= 400:
                    print(f"[AssistantScreenshot] capture failed: HTTP {resp.status}")
                    return None
                data = await resp.json()
                b64 = data.get("screenshot")
                if b64:
                    return ScreenshotEntry(
                        b64=b64,
                        utterance=utterance,
                        timestamp=datetime.now(timezone.utc),
                        source="assistant",
                    )
    except Exception as e:
        print(f"[AssistantScreenshot] capture error: {e}")
    return None


# -------- Event rendering for boss-on-call mode -------- #


def _contact_name(contact: dict) -> str:
    first = contact.get("first_name", "")
    last = contact.get("surname", "")
    name = f"{first} {last}".strip()
    return (
        name or contact.get("phone_number") or contact.get("email_address") or "Unknown"
    )


def _event_contact_id(event: Event) -> int | None:
    """Extract the contact_id from a comms event, or None if not present."""
    contact = getattr(event, "contact", None)
    if isinstance(contact, dict):
        return contact.get("contact_id")
    return None


def render_participant_comms(event_json: str, participant_ids: set[int]) -> str | None:
    """Render a comms event as a tagged message if the sender is a call participant.

    Returns a string like ``[SMS from Marcus] Hey, running late`` for
    participant comms, or None if the event is not from a participant or
    is not a comms event worth surfacing.
    """
    try:
        event = Event.from_json(event_json)
    except Exception:
        return None

    cid = _event_contact_id(event)
    if cid is None or cid not in participant_ids:
        return None

    name = _contact_name(event.contact)

    if isinstance(event, SMSReceived):
        return f"[SMS from {name}] {event.content}"
    if isinstance(event, EmailReceived):
        subj = event.subject or "(no subject)"
        body_preview = (event.body or "")[:200].strip()
        return f"[Email from {name}] {subj}" + (
            f" — {body_preview}" if body_preview else ""
        )
    if isinstance(event, UnifyMessageReceived):
        return f"[Message from {name}] {event.content}"

    return None


def render_event_for_fast_brain(event_json: str) -> str | None:
    """Render a CM event as a ``[notification]``-style string for the fast brain.

    Used for boss-on-call mode where the fast brain sees all system events.
    Returns None for events that should be silently ignored (e.g. own
    utterances, call guidance which is handled by a dedicated callback, or
    events with no user-meaningful content).
    """
    try:
        event = Event.from_json(event_json)
    except Exception:
        return None

    if isinstance(event, CallGuidance):
        return None

    if isinstance(event, SMSReceived):
        return f"SMS from {_contact_name(event.contact)}: {event.content}"
    if isinstance(event, EmailReceived):
        subj = event.subject or "(no subject)"
        return f"Email from {_contact_name(event.contact)}: {subj}"
    if isinstance(event, UnifyMessageReceived):
        return f"Unify message from {_contact_name(event.contact)}: {event.content}"
    if isinstance(event, ActorNotification):
        return f"Action progress: {event.response}"
    if isinstance(event, ActorResult):
        status = "completed successfully" if event.success else "failed"
        detail = event.result or event.error or ""
        if isinstance(detail, dict):
            detail = detail.get("summary", str(detail))
        snippet = str(detail)[:200]
        return f"Action {status}: {snippet}" if snippet else f"Action {status}"
    if isinstance(event, ActorHandleStarted):
        return f"Action started: {event.action_name} — {event.query}"
    if isinstance(event, ActorSessionResponse):
        return f"Action update: {event.content}"
    if isinstance(event, NotificationInjectedEvent):
        return event.content

    return None
