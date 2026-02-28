# unity/call_common.py

from __future__ import annotations

import asyncio
import fnmatch
import json
import sys
from secrets import token_hex
from typing import TYPE_CHECKING, Awaitable, Callable, Iterable, Optional

if TYPE_CHECKING:
    from unity.conversation_manager.types.screenshot import ScreenshotEntry

from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.events import (
    Event,
    PhoneCallReceived,
    PhoneCallSent,
    PhoneCallStarted,
    PhoneCallEnded,
    UnifyMeetReceived,
    UnifyMeetEnded,
    UnifyMeetStarted,
    InboundPhoneUtterance,
    InboundUnifyMeetUtterance,
    OutboundPhoneUtterance,
    OutboundUnifyMeetUtterance,
    SMSReceived,
    SMSSent,
    EmailReceived,
    EmailSent,
    UnifyMessageReceived,
    UnifyMessageSent,
    ActorNotification,
    ActorResult,
    ActorHandleStarted,
    ActorSessionResponse,
    NotificationInjectedEvent,
    CallGuidance,
    UserScreenShareStarted,
    UserScreenShareStopped,
    UserWebcamStarted,
    UserWebcamStopped,
)
from unity.conversation_manager.domains.ipc_socket import (
    get_socket_client,
    send_event_to_parent,
    start_socket_receive_loop,
    stop_socket_client,
)
from unity.conversation_manager.tracing import (
    payload_trace_id,
)
from unity.session_details import SESSION_DETAILS
from unity.logger import LOGGER
from unity.common.hierarchical_logger import DEFAULT_ICON, ICONS

# ─────────────────────────────────────────────────────────────────────────────
# FastBrainLogger — mirrors the async tool loop's LoopLogger format so all
# terminal output uses a consistent  ``{emoji} [{label}] {message}``  style.
# Icons are read from the central ``ICONS`` registry in hierarchical_logger.
# ─────────────────────────────────────────────────────────────────────────────


class FastBrainLogger:
    """Lightweight logger using the same ``{emoji} [{label}] {message}`` format
    as the async tool loop's ``LoopLogger``.

    *label* is ``FastBrain({suffix})``, matching the ``LoopConfig`` label convention.
    """

    def __init__(self) -> None:
        suffix = token_hex(2)
        self._label = f"FastBrain({suffix})"

    @property
    def label(self) -> str:
        return self._label

    def _emit(self, event_type: str, msg: str) -> None:
        LOGGER.info(f"{ICONS[event_type]} [{self._label}] {msg}")

    def _emit_debug(self, event_type: str, msg: str) -> None:
        LOGGER.debug(f"{ICONS[event_type]} [{self._label}] {msg}")

    # ── typed helpers ────────────────────────────────────────────────────

    def llm_thinking(self, reason: str, **kv: object) -> None:
        self._emit(
            "llm_thinking",
            f"LLM thinking…{_kv_suffix(dict(reason=reason, **kv))}",
        )

    def llm_completed(self, generation_id: str = "", **kv: object) -> None:
        extra = _kv_suffix(kv)
        self._emit(
            "llm_completed",
            f"Generation completed{_id(generation_id)}{extra}",
        )

    def llm_cancelled(self, generation_id: str = "", **kv: object) -> None:
        extra = _kv_suffix(kv)
        self._emit(
            "llm_cancelled",
            f"Generation cancelled{_id(generation_id)}{extra}",
        )

    def llm_error(self, error: str, **kv: object) -> None:
        extra = _kv_suffix(kv)
        self._emit("llm_error", f"Generation error: {error}{extra}")

    def user_speech(self, text: str) -> None:
        self._emit("user_speech", _trunc(text))

    def user_state(self, new_state: str, **kv: object) -> None:
        extra = _kv_suffix(kv)
        self._emit("user_state", f"User state: {new_state}{extra}")

    def assistant_speech(self, text: str, **kv: object) -> None:
        extra = _kv_suffix(kv)
        self._emit("assistant_speech", f"{_trunc(text)}{extra}")

    def guidance(
        self,
        source: str,
        content: str,
        *,
        guidance_id: str = "",
        speak: bool = False,
        turn: bool = False,
    ) -> None:
        self._emit(
            "guidance_received",
            f"Guidance from {source}: {_trunc(content)}"
            f" (guidance_id={guidance_id}, speak={speak}, turn={turn})",
        )

    def guidance_buffered(self, guidance_id: str, count: int) -> None:
        self._emit(
            "guidance_buffered",
            f"Buffered guidance {guidance_id} (total={count})",
        )

    def guidance_say(self, guidance_id: str, text: str, **kv: object) -> None:
        extra = _kv_suffix(kv)
        self._emit(
            "guidance_say",
            f"Speaking guidance {guidance_id}: {_trunc(text)}{extra}",
        )

    # ── proactive speech helpers ─────────────────────────────────────────

    def proactive_debounce(self, seconds: float) -> None:
        self._emit_debug(
            "proactive_debounce",
            f"Proactive speech debounce {seconds}s",
        )

    def proactive_decision(self, should_speak: bool, delay: float) -> None:
        self._emit(
            "proactive_decision",
            f"Proactive decision{_kv_suffix(dict(should_speak=should_speak, delay=f'{delay}s'))}",
        )

    def proactive_deferred(self, reason: str) -> None:
        self._emit_debug("proactive_deferred", f"Proactive deferred: {reason}")

    def proactive_dormant(self) -> None:
        self._emit(
            "proactive_dormant",
            "Proactive dormant until next utterance",
        )

    def proactive_speaking(self, delay: float, content: str) -> None:
        self._emit(
            "proactive_speaking",
            f"Proactive speaking in {delay}s: {_trunc(content)}",
        )

    def proactive_published(self, guidance_id: str, content: str) -> None:
        self._emit(
            "proactive_published",
            f"Proactive spoke: {_trunc(content)}{_kv_suffix(dict(guidance_id=guidance_id))}",
        )

    def proactive_cancelled(self) -> None:
        self._emit_debug("proactive_cancelled", "Proactive speech cancelled")

    def proactive_error(self, error: str) -> None:
        LOGGER.error(
            f"{ICONS['proactive_error']} [{self._label}] Proactive error: {error}",
        )

    def call_status(self, event_name: str) -> None:
        self._emit_debug("call_status", event_name)

    def session_start(self, msg: str = "Session starting") -> None:
        self._emit_debug("session_start", msg)

    def session_end(self, msg: str = "Session ended") -> None:
        self._emit_debug("session_end", msg)

    def session_ready(self, msg: str = "Session ready") -> None:
        self._emit_debug("session_ready", msg)

    def participant_comms(self, text: str) -> None:
        self._emit("participant_comms", _trunc(text))

    def boss_event(self, text: str) -> None:
        self._emit("boss_event", _trunc(text))

    def ipc_inbound(self, channel: str, **kv: object) -> None:
        extra = _kv_suffix(kv)
        self._emit_debug("ipc_inbound", f"IPC recv {channel}{extra}")

    def ipc_outbound(self, channel: str, **kv: object) -> None:
        extra = _kv_suffix(kv)
        self._emit_debug("ipc_outbound", f"IPC send {channel}{extra}")

    def ipc_error(self, msg: str) -> None:
        self._emit_debug("ipc_error", msg)

    def screenshot(self, msg: str) -> None:
        self._emit("screenshot", msg)

    def screenshot_debug(self, msg: str) -> None:
        self._emit_debug("screenshot", msg)

    def config(self, msg: str) -> None:
        self._emit_debug("config", msg)

    def dispatch(self, msg: str) -> None:
        self._emit_debug("dispatch", msg)

    def info(self, msg: str) -> None:
        self._emit_debug("info", msg)

    def warning(self, msg: str) -> None:
        self._emit("warning", msg)

    def error(self, msg: str) -> None:
        self._emit("error", msg)

    def shutdown(self, msg: str) -> None:
        self._emit("shutdown", msg)


def _kv_suffix(kv: dict[str, object]) -> str:
    """Render extra key=value pairs as a parenthesised, comma-separated suffix."""
    if not kv:
        return ""
    parts = [f"{k}={v}" for k, v in kv.items() if v is not None and v != ""]
    return f" ({', '.join(parts)})" if parts else ""


def _id(val: str) -> str:
    return f" {val}" if val else ""


def _trunc(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


class SocketAwareEventBroker:
    """
    Simple event broker for cross-process communication via Unix socket.

    When running as a subprocess (detected via CM_EVENT_SOCKET env var):
    - Outbound: publish() sends events to parent via socket
    - Inbound: register_callback() handlers are invoked when events arrive

    Otherwise, falls back to in-memory broker for outbound events.
    """

    def __init__(self, fb_logger: FastBrainLogger | None = None):
        self._socket_client = get_socket_client()
        self._fallback_broker = get_event_broker()
        self._receive_started = False
        self._callbacks: dict[str, Callable[[dict], None]] = {}
        self._pattern_callbacks: list[tuple[str, Callable[[dict], None]]] = []
        self._log = fb_logger

    def set_logger(self, fb_logger: FastBrainLogger) -> None:
        self._log = fb_logger

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
            if self._log:
                self._log.info("No socket client, IPC receive disabled")
            return False

        async def on_event(channel: str, event_json: str) -> None:
            """Invoke registered callback when event arrives."""
            message_id = payload_trace_id("ipc", channel, event_json)
            has_exact = channel in self._callbacks
            has_pattern = any(
                fnmatch.fnmatch(channel, pat) for pat, _ in self._pattern_callbacks
            )
            if self._log:
                self._log.ipc_inbound(
                    channel,
                    message_id=message_id,
                    has_callback=has_exact or has_pattern,
                )
            try:
                data = json.loads(event_json)
            except Exception as e:
                if self._log:
                    self._log.ipc_error(f"JSON parse error: {e}")
                return
            if has_exact:
                try:
                    self._callbacks[channel](data)
                except Exception as e:
                    if self._log:
                        self._log.ipc_error(f"Callback error on {channel}: {e}")
            for pat, handler in self._pattern_callbacks:
                if fnmatch.fnmatch(channel, pat):
                    try:
                        handler(data)
                    except Exception as e:
                        if self._log:
                            self._log.ipc_error(f"Pattern callback error on {pat}: {e}")

        success = await start_socket_receive_loop(on_event)
        if success:
            self._receive_started = True
            if self._log:
                self._log.info("IPC receive loop started")
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
                if self._log:
                    self._log.ipc_outbound(channel, via="socket")
                return 1
            else:
                if self._log:
                    self._log.warning(f"Socket send failed, using fallback: {channel}")

        if self._log:
            self._log.ipc_outbound(channel, via="fallback")
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
    room_name: str = "",
    pre_shutdown_callback: Optional[Callable[[], None]] = None,
) -> Callable[[], Awaitable[None]]:
    """
    Returns an async function that:
      - calls optional pre_shutdown_callback (e.g., for usage logging)
      - deletes the LiveKit room (evicting agents / cancelling dispatches)
      - publishes the call ended event
      - cancels all other asyncio tasks

    The process will be terminated by SIGTERM from the parent when cleanup is called.

    Args:
        contact: Contact dictionary for the call.
        channel: Channel type ("phone" or other).
        room_name: LiveKit room name to delete on shutdown.
        pre_shutdown_callback: Optional sync callback to run before shutdown.
            Useful for logging call usage/metrics before tasks are cancelled.
    """

    async def end_call() -> None:
        LOGGER.debug(f"{ICONS['lifecycle']} Initiating graceful shutdown...")

        # Run pre-shutdown callback (e.g., usage logging) before cleanup
        if pre_shutdown_callback is not None:
            try:
                pre_shutdown_callback()
            except Exception as e:  # noqa: BLE001
                LOGGER.error(f"{DEFAULT_ICON} Error in pre-shutdown callback: {e}")

        # Delete room before notifying the parent, since the parent will
        # SIGKILL us immediately after receiving the call-ended event.
        if room_name:
            await delete_livekit_room(room_name)

        # Send end call event before cleaning tasks and closing connection
        await publish_call_ended(contact, channel)
        LOGGER.debug(f"{DEFAULT_ICON} End call event sent")

        # Get all running tasks except current task
        tasks: Iterable[asyncio.Task] = [
            t for t in asyncio.all_tasks() if t is not asyncio.current_task()
        ]

        if tasks:
            LOGGER.debug(f"{DEFAULT_ICON} Cancelling {len(tasks)} running tasks...")
            # Cancel all tasks
            for task in tasks:
                task.cancel()

            # Wait for tasks to be cancelled gracefully
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                LOGGER.debug(f"{DEFAULT_ICON} All tasks cancelled successfully")
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001
                LOGGER.error(f"{DEFAULT_ICON} Error during task cancellation: {e}")

        LOGGER.debug(f"{ICONS['lifecycle']} Graceful shutdown completed")

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
                LOGGER.info(
                    f"{ICONS['lifecycle']} Inactivity timeout reached, shutting down agent...",
                )
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
    LOGGER.debug(f"{DEFAULT_ICON} sys.argv {sys.argv}")

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
                    LOGGER.error(f"{DEFAULT_ICON} {env_name} payload is not valid JSON")
                    sys.exit(1)
                if not loaded:
                    LOGGER.error(
                        f"{DEFAULT_ICON} {env_name} payload is invalid (empty)",
                    )
                    sys.exit(1)

            # Map known extra args to SESSION_DETAILS fields
            if env_name == "CONTACT":
                SESSION_DETAILS.voice_call.contact_json = value
            elif env_name == "BOSS":
                SESSION_DETAILS.voice_call.boss_json = value
            elif env_name == "ASSISTANT_BIO":
                SESSION_DETAILS.assistant.about = value
            elif env_name == "ASSISTANT_ID":
                try:
                    SESSION_DETAILS.assistant.agent_id = int(value)
                except (ValueError, TypeError):
                    pass
            elif env_name == "USER_ID":
                SESSION_DETAILS.user.id = value

        # Export to env for subprocess inheritance
        SESSION_DETAILS.export_to_env()

        # keep only script name and the command ("dev" / "connect" / "download-files")
        sys.argv = sys.argv[:2]
    elif len(sys.argv) > 1 and sys.argv[1] != "download-files":
        LOGGER.error(f"{DEFAULT_ICON} Not enough arguments provided")
        sys.exit(1)

    return room_name


def should_dispatch_livekit_agent() -> bool:
    """
    True when we should actually call dispatch_livekit_agent() for this process.
    """
    return len(sys.argv) > 1 and sys.argv[1] != "download-files"


async def delete_livekit_room(room_name: str) -> None:
    """Delete a LiveKit room to evict lingering agents and cancel dispatches."""
    try:
        from livekit.api import LiveKitAPI, DeleteRoomRequest

        api = LiveKitAPI()
        try:
            await api.room.delete_room(DeleteRoomRequest(room=room_name))
            LOGGER.debug(f"{DEFAULT_ICON} Deleted LiveKit room '{room_name}'")
        finally:
            await api.aclose()
    except Exception as e:
        LOGGER.error(f"{DEFAULT_ICON} Failed to delete LiveKit room '{room_name}': {e}")


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

    def __init__(
        self,
        room,
        *,
        track_source: str = "screenshare",
        on_track_change: Callable[[str, bool], Awaitable[None]] | None = None,
        fb_logger: FastBrainLogger | None = None,
    ) -> None:
        from livekit import rtc

        self._latest_frame_data: tuple[bytes, int, int] | None = None
        self._capture_task: asyncio.Task | None = None
        self._stream = None
        self._on_track_change = on_track_change
        self._log = fb_logger

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
            if self._log:
                self._log.screenshot_debug(
                    f"{self._label} track subscribed, starting capture",
                )
            stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGBA)
            self._stream = stream
            self._capture_task = asyncio.create_task(self._capture_loop(stream))
            if self._on_track_change is not None:
                asyncio.create_task(self._on_track_change(self._label, True))

    def _handle_track_unsubscribed(self, publication) -> None:
        if publication.source == self._rtc_source:
            if self._log:
                self._log.screenshot_debug(
                    f"{self._label} track unsubscribed, stopping capture",
                )
            self._latest_frame_data = None
            if self._capture_task and not self._capture_task.done():
                self._capture_task.cancel()
                self._capture_task = None
            self._stream = None
            if self._on_track_change is not None:
                asyncio.create_task(self._on_track_change(self._label, False))

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
            if self._log:
                self._log.error(f"Frame capture error ({self._label}): {e}")
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


# Backward-compatible alias used by call.py imports.
UserScreenCaptureManager = UserTrackCaptureManager


_TRACK_TO_EVENT: dict[tuple[str, bool], type[Event]] = {
    ("screenshare", True): UserScreenShareStarted,
    ("screenshare", False): UserScreenShareStopped,
    ("camera", True): UserWebcamStarted,
    ("camera", False): UserWebcamStopped,
}


async def publish_meet_interaction_from_track(source: str, active: bool) -> None:
    """Publish a meet interaction event when a LiveKit video track changes.

    Called by ``UserTrackCaptureManager`` via the ``on_track_change`` callback
    so that the CM receives the same events it would get from a real frontend.
    """
    event_cls = _TRACK_TO_EVENT.get((source, active))
    if event_cls is None:
        return
    event = event_cls(reason="LiveKit track auto-detected")
    await event_broker.publish(event_cls.topic, event.to_json())


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

    def clear(self, source: str | None = None) -> None:
        """Remove entries, optionally filtered by source.

        When *source* is ``None``, all entries are removed.
        """
        if source is None:
            self._entries.clear()
        else:
            self._entries = [(e, p) for e, p in self._entries if e.source != source]

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
        from urllib.parse import urlparse

        parsed = urlparse(desktop_url)
        return f"{parsed.scheme}://{parsed.netloc}/api"
    return "http://localhost:3000"


def _ensure_jpeg(b64: str) -> str:
    """Convert a base64-encoded image to JPEG if it isn't already.

    The agent-service screenshot endpoint returns PNG (Playwright default),
    but all downstream consumers tag images as ``image/jpeg``.  Converting
    here keeps every ``ScreenshotEntry.b64`` consistently JPEG, matching
    the format produced by ``UserTrackCaptureManager.capture_screenshot``.
    """
    import base64 as b64mod
    import io

    raw = b64mod.b64decode(b64)
    if raw[:2] == b"\xff\xd8":
        return b64

    from PIL import Image

    img = Image.open(io.BytesIO(raw))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return b64mod.b64encode(buf.getvalue()).decode("ascii")


async def _screenshot_post(
    session,
    url: str,
    headers: dict,
    timeout,
) -> "tuple[int, str | None, dict | None]":
    """POST to the screenshot endpoint and return (status, body_text, json).

    Returns the status code, raw body text (on error), and parsed JSON (on
    success) so callers can handle retries without duplicating HTTP plumbing.
    """
    use_ssl = False if "vm.unify.ai" in url else None
    async with session.post(
        url,
        json={},
        headers=headers,
        timeout=timeout,
        ssl=use_ssl,
    ) as resp:
        if resp.status >= 400:
            return resp.status, await resp.text(), None
        return resp.status, None, await resp.json()


async def capture_assistant_screenshot(
    utterance: str,
    fb_logger: FastBrainLogger | None = None,
    agent_service_url: str | None = None,
    http_session=None,
) -> "ScreenshotEntry | None":
    """Capture the assistant's desktop via HTTP POST.

    Returns a ``ScreenshotEntry`` on success, ``None`` on failure or if no
    desktop URL is configured.

    When *http_session* is provided (a persistent ``aiohttp.ClientSession``),
    it is reused across calls — eliminating per-request DNS + TLS overhead
    (~300 ms).  When ``None``, a throwaway session with phase-level trace
    diagnostics is created (useful for one-off / CM-process captures).
    """
    import aiohttp
    import time as _time

    from datetime import datetime, timezone
    from unity.session_details import SESSION_DETAILS
    from unity.conversation_manager.types.screenshot import ScreenshotEntry

    base_url = agent_service_url or _resolve_agent_service_url()
    auth_key = SESSION_DETAILS.unify_key
    url = f"{base_url}/screenshot"
    headers = {"authorization": f"Bearer {auth_key}"}
    timeout = aiohttp.ClientTimeout(total=10)

    def _log(msg: str) -> None:
        if fb_logger:
            fb_logger.screenshot(msg)
        else:
            LOGGER.warning(msg)

    t_start = _time.monotonic()

    def _make_entry(b64: str) -> ScreenshotEntry:
        return ScreenshotEntry(
            b64=_ensure_jpeg(b64),
            utterance=utterance,
            timestamp=datetime.now(timezone.utc),
            source="assistant",
        )

    # -- Fast path: reuse a persistent session (no per-request DNS/TLS) ------
    if http_session is not None:
        try:
            status, err_body, data = await _screenshot_post(
                http_session,
                url,
                headers,
                timeout,
            )
            total_ms = (_time.monotonic() - t_start) * 1000
            if status >= 400:
                if err_body and "no_desktop_session" in err_body:
                    _log(
                        f"Assistant screenshot: no desktop session yet, "
                        f"retrying in 2s (url={url}, total={total_ms:.0f}ms)",
                    )
                    await asyncio.sleep(2)
                    status, err_body, data = await _screenshot_post(
                        http_session,
                        url,
                        headers,
                        timeout,
                    )
                    total_ms = (_time.monotonic() - t_start) * 1000
                    if status >= 400:
                        _log(
                            f"Assistant screenshot failed after retry: "
                            f"HTTP {status} (url={url}, total={total_ms:.0f}ms, "
                            f"body={(err_body or '')[:200]})",
                        )
                        return None
                else:
                    _log(
                        f"Assistant screenshot failed: HTTP {status} "
                        f"(url={url}, total={total_ms:.0f}ms, "
                        f"body={(err_body or '')[:200]})",
                    )
                    return None
            if data:
                b64 = data.get("screenshot")
                if b64:
                    _log(
                        f"Assistant screenshot OK"
                        f" (url={url}, total={total_ms:.0f}ms, b64_len={len(b64)})",
                    )
                    return _make_entry(b64)
        except Exception as e:
            total_ms = (_time.monotonic() - t_start) * 1000
            _log(
                f"Assistant screenshot error: {type(e).__name__}: {e} "
                f"(url={url}, total={total_ms:.0f}ms)",
            )
        return None

    # -- Diagnostic path: throwaway session with phase-level HTTP tracing ----
    phases: dict[str, float] = {}
    trace_cfg = aiohttp.TraceConfig()

    async def on_dns_start(session, ctx, params):
        phases["dns_start"] = _time.monotonic()

    async def on_dns_end(session, ctx, params):
        phases["dns_ms"] = (_time.monotonic() - phases.get("dns_start", t_start)) * 1000

    async def on_conn_start(session, ctx, params):
        phases["conn_start"] = _time.monotonic()

    async def on_conn_end(session, ctx, params):
        phases["conn_ms"] = (
            _time.monotonic() - phases.get("conn_start", t_start)
        ) * 1000

    async def on_req_start(session, ctx, params):
        phases["req_start"] = _time.monotonic()

    async def on_req_end(session, ctx, params):
        phases["first_byte_ms"] = (
            _time.monotonic() - phases.get("req_start", t_start)
        ) * 1000

    trace_cfg.on_dns_resolvehost_start.append(on_dns_start)
    trace_cfg.on_dns_resolvehost_end.append(on_dns_end)
    trace_cfg.on_connection_create_start.append(on_conn_start)
    trace_cfg.on_connection_create_end.append(on_conn_end)
    trace_cfg.on_request_start.append(on_req_start)
    trace_cfg.on_request_end.append(on_req_end)

    def _phase_str() -> str:
        return (
            f"dns={phases.get('dns_ms', 0):.0f}ms, "
            f"conn={phases.get('conn_ms', 0):.0f}ms, "
            f"first_byte={phases.get('first_byte_ms', 0):.0f}ms"
        )

    try:
        async with aiohttp.ClientSession(trace_configs=[trace_cfg]) as session:
            status, err_body, data = await _screenshot_post(
                session,
                url,
                headers,
                timeout,
            )
            total_ms = (_time.monotonic() - t_start) * 1000
            if status >= 400:
                if err_body and "no_desktop_session" in err_body:
                    _log(
                        f"Assistant screenshot: no desktop session yet, "
                        f"retrying in 2s (url={url}, total={total_ms:.0f}ms)",
                    )
                    await asyncio.sleep(2)
                    status, err_body, data = await _screenshot_post(
                        session,
                        url,
                        headers,
                        timeout,
                    )
                    total_ms = (_time.monotonic() - t_start) * 1000
                    if status >= 400:
                        _log(
                            f"Assistant screenshot failed after retry: "
                            f"HTTP {status} (url={url}, total={total_ms:.0f}ms, "
                            f"body={(err_body or '')[:200]})",
                        )
                        return None
                else:
                    _log(
                        f"Assistant screenshot failed: HTTP {status} "
                        f"(url={url}, total={total_ms:.0f}ms, {_phase_str()}, "
                        f"body={(err_body or '')[:200]})",
                    )
                    return None
            if data:
                b64 = data.get("screenshot")
                if b64:
                    _log(
                        f"Assistant screenshot OK"
                        f" (url={url}, total={total_ms:.0f}ms, {_phase_str()}, "
                        f"b64_len={len(b64)})",
                    )
                    return _make_entry(b64)
    except Exception as e:
        total_ms = (_time.monotonic() - t_start) * 1000
        _log(
            f"Assistant screenshot error: {type(e).__name__}: {e} "
            f"(url={url}, total={total_ms:.0f}ms, {_phase_str()})",
        )
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


# -------- Fast brain history hydration & context windowing -------- #


def _render_history_event(
    event: Event,
    participant_ids: set[int],
    is_boss_user: bool,
    assistant_name: str,
) -> str | None:
    """Render a single historical Comms event for fast brain context.

    Returns a human-readable line for the event, or None to skip it.
    Covers the same event types as the live forwarding rules, plus
    utterances and sent messages which are only relevant for history.
    """
    cid = _event_contact_id(event)
    name = _contact_name(getattr(event, "contact", {}) or {})

    # -- Utterances (transcript lines) --
    if isinstance(event, (InboundPhoneUtterance, InboundUnifyMeetUtterance)):
        if cid is not None and cid in participant_ids:
            return f"{name}: {event.content}"
        return None
    if isinstance(event, (OutboundPhoneUtterance, OutboundUnifyMeetUtterance)):
        return f"{assistant_name}: {event.content}"

    # -- Call lifecycle markers --
    if isinstance(event, (PhoneCallReceived, PhoneCallSent, PhoneCallStarted)):
        if cid is not None and cid in participant_ids:
            return f"--- Call with {name} ---"
        return None
    if isinstance(event, (UnifyMeetReceived, UnifyMeetStarted)):
        if cid is not None and cid in participant_ids:
            return f"--- Meeting with {name} ---"
        return None
    if isinstance(event, (PhoneCallEnded, UnifyMeetEnded)):
        if cid is not None and cid in participant_ids:
            return f"--- Call ended ---"
        return None

    # -- Text messages (received + sent, filtered to participants) --
    if isinstance(event, SMSReceived):
        if cid is not None and cid in participant_ids:
            return f"[SMS from {name}] {event.content}"
        return None
    if isinstance(event, SMSSent):
        if cid is not None and cid in participant_ids:
            return f"[SMS to {name}] {event.content}"
        return None
    if isinstance(event, EmailReceived):
        if cid is not None and cid in participant_ids:
            subj = event.subject or "(no subject)"
            return f"[Email from {name}] {subj}"
        return None
    if isinstance(event, EmailSent):
        if cid is not None and cid in participant_ids:
            subj = event.subject or "(no subject)"
            return f"[Email to {name}] {subj}"
        return None
    if isinstance(event, UnifyMessageReceived):
        if cid is not None and cid in participant_ids:
            return f"[Message from {name}] {event.content}"
        return None
    if isinstance(event, UnifyMessageSent):
        if cid is not None and cid in participant_ids:
            return f"[Message to {name}] {event.content}"
        return None

    # -- Boss-only: Actor events --
    if is_boss_user:
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

    return None


async def hydrate_fast_brain_history(
    participant_ids: set[int],
    is_boss_user: bool,
    assistant_name: str,
    limit: int = 50,
) -> list[str]:
    """Load recent Comms events from the backend and render them for the fast brain.

    Queries Orchestra directly via ``unify.get_logs()`` rather than going
    through the EventBus proxy, which is not initialised in the voice agent
    subprocess.  Returns a chronologically ordered list of rendered strings
    suitable for injecting as historical context before the current call begins.
    """
    import unify

    context = (
        f"{SESSION_DETAILS.user_context}/" f"{SESSION_DETAILS.assistant_context}/Events"
    )

    try:
        logs = await asyncio.to_thread(
            unify.get_logs,
            context=context,
            filter='type == "Comms"',
            sorting={"timestamp": "descending"},
            limit=limit,
        )
    except Exception:
        return []
    if not logs:
        return []

    # Logs arrive newest-first; reverse for chronological order
    logs.reverse()

    rendered: list[str] = []
    for log in logs:
        entries = log.entries
        payload_cls = entries.get("payload_cls", "")
        if "." in payload_cls:
            payload_cls = payload_cls.rsplit(".", 1)[-1]

        payload_json_str = entries.get("payload_json")
        if not payload_json_str:
            continue

        try:
            payload = json.loads(payload_json_str)
            cm_event = Event.from_dict({"event_name": payload_cls, "payload": payload})
        except Exception:
            continue

        text = _render_history_event(
            cm_event,
            participant_ids,
            is_boss_user,
            assistant_name,
        )
        if text:
            rendered.append(text)

    return rendered


def trim_fast_brain_context(items: list, window_size: int) -> list:
    """Return a trimmed view of ChatContext items respecting a rolling window.

    Preserves all contiguous system-role messages at the start of the list
    (the system prompt / history preamble) and keeps at most ``window_size``
    conversation items after them.
    """
    # Find where the system prompt block ends
    system_end = 0
    for i, item in enumerate(items):
        if getattr(item, "role", None) == "system":
            system_end = i + 1
        else:
            break

    conversation_items = items[system_end:]
    if len(conversation_items) <= window_size:
        return list(items)

    if window_size == 0:
        return list(items[:system_end])
    return list(items[:system_end]) + conversation_items[-window_size:]
