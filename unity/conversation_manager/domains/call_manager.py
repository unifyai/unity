from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import aiohttp
from livekit.api import CreateAgentDispatchRequest, LiveKitAPI

from unity.contact_manager.types.contact import UNASSIGNED
from unity.conversation_manager.events import *
from unity.conversation_manager.domains.ipc_socket import (
    CallEventSocketServer,
    CM_EVENT_SOCKET_ENV,
)
from unity.conversation_manager.tracing import trace_kv
from unity.logger import LOGGER
from unity.common.hierarchical_logger import DEFAULT_ICON, ICONS
from unity.helpers import (
    run_script,
    terminate_process,
)

if TYPE_CHECKING:
    from unity.conversation_manager.in_memory_event_broker import InMemoryEventBroker


def _resolve_agent_service_url() -> str:
    """Resolve agent-service base URL (same logic as common.py)."""
    from unity.session_details import SESSION_DETAILS

    desktop_url = SESSION_DETAILS.assistant.desktop_url
    if desktop_url:
        from urllib.parse import urlparse

        parsed = urlparse(desktop_url)
        return f"{parsed.scheme}://{parsed.netloc}/api"
    return "http://localhost:3000"


def make_room_name(assistant_id: str, medium: str) -> str:
    """Canonical LiveKit room name for a given assistant and medium.

    Format: unity_{assistant_id}_{medium}
    Examples: unity_25_phone, unity_25_meet, unity_25_teams
    """
    return f"unity_{assistant_id}_{medium}"


@dataclass
class CallConfig:
    assistant_id: str
    user_id: str
    assistant_bio: str
    assistant_number: str
    voice_provider: str
    voice_id: str
    assistant_name: str = ""
    job_name: str = ""


_BASE_FORWARD_CHANNELS = [
    "app:call:*",
    "app:comms:*",
]


class LivekitCallManager:
    def __init__(
        self,
        config: CallConfig,
        event_broker: "InMemoryEventBroker | None" = None,
    ):
        self.job_name: str = ""
        self.set_config(config=config)
        self.call_exchange_id = UNASSIGNED
        self.unify_meet_exchange_id = UNASSIGNED
        self.call_start_timestamp = None
        self.unify_meet_start_timestamp = None
        self.call_contact = None
        self._call_proc: subprocess.Popen | None = None
        self._worker_proc: subprocess.Popen | None = None
        self._active_job: bool = False
        self._call_pending: bool = False
        self.conference_name = ""
        self.room_name = ""
        self._event_broker = event_broker
        self._socket_server: CallEventSocketServer | None = None
        self.is_outbound: bool = False
        self.initial_notification: str = ""
        self.on_screenshot: Callable[[str], None] | None = None
        self.on_fast_brain_generating: Callable[[], None] | None = None
        self.on_pipeline_quiescent: Callable[[bool], None] | None = None
        self._call_channel: str | None = None
        self._disconnect_contact: dict | None = None
        self._boss_notification_task: asyncio.Task | None = None
        self._worker_watchdog_task: asyncio.Task | None = None
        # Google Meet state
        self._gmeet_session_id: str | None = None
        self._gmeet_joining: bool = False
        self.google_meet_start_timestamp = None
        self.google_meet_exchange_id = UNASSIGNED

    def set_config(self, config: CallConfig):
        self.assistant_id = config.assistant_id
        self.user_id = config.user_id
        self.assistant_bio = config.assistant_bio
        self.assistant_number = config.assistant_number
        self.voice_provider = config.voice_provider
        self.voice_id = config.voice_id
        self.assistant_name = config.assistant_name
        if config.job_name:
            self.job_name = config.job_name

    def set_event_broker(self, event_broker: "InMemoryEventBroker") -> None:
        """Set the event broker for socket server to publish to."""
        self._event_broker = event_broker

    @property
    def worker_agent_name(self) -> str:
        return f"unity_{self.job_name}"

    @property
    def has_active_call(self) -> bool:
        return self._call_pending or self._active_job or self._call_proc is not None

    # ------------------------------------------------------------------
    # Persistent worker lifecycle
    # ------------------------------------------------------------------

    def start_persistent_worker(self) -> None:
        """Start the persistent LiveKit agent worker subprocess.

        Called once during pod initialisation.  The worker registers with
        LiveKit and maintains a pool of pre-warmed child processes.
        Skips silently when LiveKit is not configured (e.g. in tests).
        """
        if not os.environ.get("LIVEKIT_URL"):
            return
        if self._worker_proc is not None and self._worker_proc.poll() is None:
            return

        target = Path(__file__).parent.parent.resolve() / "medium_scripts" / "worker.py"
        self._worker_proc = run_script(str(target), "dev", self.worker_agent_name)
        LOGGER.info(
            f"{ICONS['ipc']} [LivekitCallManager] Persistent worker started "
            f"(pid={self._worker_proc.pid}, agent_name={self.worker_agent_name})",
        )
        if self._worker_watchdog_task is None or self._worker_watchdog_task.done():
            self._worker_watchdog_task = asyncio.create_task(self._worker_watchdog())

    async def _worker_watchdog(self) -> None:
        """Restart the persistent worker if it exits unexpectedly,
        and emit an INFO log when the warm pool is ready."""
        ready_logged = False
        while True:
            await asyncio.sleep(2)
            if self._worker_proc is None:
                continue
            if self._worker_proc.poll() is not None:
                LOGGER.warning(
                    f"{ICONS['ipc']} [LivekitCallManager] Persistent worker exited "
                    f"(code={self._worker_proc.returncode}), restarting…",
                )
                self._worker_proc = None
                ready_logged = False
                self.start_persistent_worker()
            elif not ready_logged:
                from unity.conversation_manager.medium_scripts.worker import (
                    WORKER_READY_PATH,
                )

                if os.path.exists(WORKER_READY_PATH):
                    LOGGER.info(
                        "🎙️ [LivekitCallManager] Voice agent ready",
                    )
                    ready_logged = True

    async def _dispatch_job(
        self,
        room_name: str,
        channel: str,
        contact: dict,
        boss: dict,
        outbound: bool,
        *,
        extra_metadata: dict | None = None,
    ) -> None:
        """Dispatch a LiveKit job to the persistent worker."""
        socket_path = await self._ensure_socket_server()

        meta_dict = {
            "voice_provider": self.voice_provider,
            "voice_id": self.voice_id,
            "outbound": outbound,
            "channel": channel,
            "contact": contact,
            "boss": boss,
            "assistant_bio": self.assistant_bio,
            "assistant_id": self.assistant_id,
            "user_id": self.user_id,
            "assistant_name": self.assistant_name,
            "ipc_socket_path": socket_path or "",
        }
        if extra_metadata:
            meta_dict.update(extra_metadata)
        metadata = json.dumps(meta_dict)

        lk = LiveKitAPI(
            url=os.environ.get("LIVEKIT_URL", ""),
            api_key=os.environ.get("LIVEKIT_API_KEY", ""),
            api_secret=os.environ.get("LIVEKIT_API_SECRET", ""),
        )
        try:
            dispatch = await lk.agent_dispatch.create_dispatch(
                CreateAgentDispatchRequest(
                    agent_name=self.worker_agent_name,
                    room=room_name,
                    metadata=metadata,
                ),
            )
            self._active_job = True
            LOGGER.info(
                f"{ICONS['ipc']} [LivekitCallManager] Dispatched job "
                f"(dispatch_id={dispatch.id}, room={room_name})",
            )
        finally:
            await lk.aclose()

    async def _ensure_socket_server(self) -> str | None:
        """Start the socket server if not running, return socket path."""
        if self._event_broker is None:
            LOGGER.error(
                f"{ICONS['ipc']} [LivekitCallManager] Warning: No event broker set, socket IPC disabled",
            )
            return None

        if self._socket_server is None:

            async def _on_ipc_event(channel: str, event_json: str) -> None:
                if channel == "app:comms:screenshot" and self.on_screenshot is not None:
                    self.on_screenshot(event_json)
                elif (
                    channel == "app:comms:fast_brain_generating"
                    and self.on_fast_brain_generating is not None
                ):
                    self.on_fast_brain_generating()
                elif (
                    channel == "app:comms:pipeline_quiescent"
                    and self.on_pipeline_quiescent is not None
                ):
                    import json as _json

                    payload = _json.loads(event_json)
                    self.on_pipeline_quiescent(payload["quiescent"])
                else:
                    await self._event_broker.publish(channel, event_json)

            self._socket_server = CallEventSocketServer(
                self._event_broker,
                on_event=_on_ipc_event,
            )
            self._socket_server.on_client_disconnected = (
                self._on_ipc_client_disconnected
            )

        if self._socket_server.socket_path is None:
            socket_path = await self._socket_server.start()
            return socket_path

        return self._socket_server.socket_path

    async def start_call(
        self,
        contact: dict,
        boss: dict,
        outbound: bool = False,
        channel: str = "phone_call",
    ):
        if self.has_active_call:
            LOGGER.warning(
                f"{ICONS['ipc']} [LivekitCallManager] start_call ignored: "
                "call already active",
            )
            return

        self._call_pending = False
        self.is_outbound = outbound
        self._call_channel = channel
        self._disconnect_contact = contact

        await self._ensure_socket_server()
        if self._socket_server:
            await self._socket_server.set_forward_channels(list(_BASE_FORWARD_CHANNELS))

        if contact.get("is_system", False):
            self._start_boss_notification_rendering()

        medium = "whatsapp_call" if channel == "whatsapp_call" else "phone"
        room_name = make_room_name(self.assistant_id, medium)

        if self._worker_proc is not None and self._worker_proc.poll() is None:
            await self._dispatch_job(room_name, channel, contact, boss, outbound)
        else:
            await self._start_call_subprocess(
                room_name,
                channel,
                contact,
                boss,
                outbound,
            )

        if self.initial_notification:
            notification_event = FastBrainNotification(
                contact=contact,
                content=self.initial_notification,
                source="initial_call",
            )
            await self._socket_server.queue_for_clients(
                "app:call:notification",
                notification_event.to_json(),
            )
            await self._event_broker.publish(
                "app:comms:assistant_notification",
                notification_event.to_json(),
            )
            LOGGER.debug(
                f"{ICONS['ipc']} {trace_kv('CALL_MANAGER_INITIAL_NOTIFICATION', content_preview=self.initial_notification[:80])}",
            )
            self.initial_notification = ""

    async def start_unify_meet(
        self,
        contact: dict,
        boss: dict,
        room_name: str | None,
    ):
        if self.has_active_call:
            LOGGER.warning(
                f"{ICONS['ipc']} [LivekitCallManager] start_unify_meet ignored: "
                "call already active",
            )
            return

        self.is_outbound = False
        self._call_channel = "unify_meet"
        self._disconnect_contact = contact

        await self._ensure_socket_server()
        if self._socket_server:
            await self._socket_server.set_forward_channels(list(_BASE_FORWARD_CHANNELS))

        if contact.get("is_system", False):
            self._start_boss_notification_rendering()

        room_name = room_name or make_room_name(self.assistant_id, "meet")
        self.room_name = room_name

        if self._worker_proc is not None and self._worker_proc.poll() is None:
            await self._dispatch_job(room_name, "unify_meet", contact, boss, False)
        else:
            await self._start_call_subprocess(
                room_name,
                "unify_meet",
                contact,
                boss,
                False,
            )

    # ------------------------------------------------------------------
    # Google Meet lifecycle
    # ------------------------------------------------------------------

    @property
    def has_active_google_meet(self) -> bool:
        return self._gmeet_session_id is not None or self._gmeet_joining

    async def start_google_meet(
        self,
        meet_url: str,
        contact: dict,
        boss: dict,
        display_name: str = "",
    ) -> bool:
        """Join a Google Meet via agent-service browser and start the audio bridge.

        1. POST /googlemeet/join on agent-service to launch browser + automation.
        2. Start audio bridge (PulseAudio <-> LiveKit).
        3. Dispatch a fast brain job into the same LiveKit room.
        4. Kick off a background monitor that polls /googlemeet/state and
           publishes GoogleMeetEnded when the meeting terminates.
        """
        if self.has_active_call or self.has_active_google_meet:
            LOGGER.warning(
                f"{ICONS['ipc']} [LivekitCallManager] start_google_meet ignored: "
                "session already active",
            )
            return False

        self._gmeet_joining = True
        self._call_channel = "google_meet"
        self._disconnect_contact = contact

        display_name = display_name or self.assistant_name or "Unity Assistant"

        base_url = "http://localhost:3000"
        auth_key = os.environ.get("UNIFY_KEY", "")

        room_name = make_room_name(self.assistant_id, "gmeet")
        self.room_name = room_name

        # Pre-create the LiveKit room with a long empty_timeout.  Google Meet
        # audio flows through sounddevice/PulseAudio — no "real" LiveKit
        # participant ever joins, so the server's default empty_timeout (300s)
        # would auto-delete the room after 5 minutes.
        from livekit.api import CreateRoomRequest

        lk = LiveKitAPI(
            url=os.environ.get("LIVEKIT_URL", ""),
            api_key=os.environ.get("LIVEKIT_API_KEY", ""),
            api_secret=os.environ.get("LIVEKIT_API_SECRET", ""),
        )
        try:
            await lk.room.create_room(
                CreateRoomRequest(name=room_name, empty_timeout=10800),
            )
        finally:
            await lk.aclose()

        # Dispatch fast brain first so it initializes (models, history, greeting)
        # while the browser navigates the slow LLM-guided join flow.
        await self._ensure_socket_server()
        if self._socket_server:
            await self._socket_server.set_forward_channels(list(_BASE_FORWARD_CHANNELS))

        if contact.get("is_system", False):
            self._start_boss_notification_rendering()

        gmeet_extra = {
            "gmeet_session_id": "",
            "gmeet_meet_url": meet_url,
            "agent_service_url": "http://localhost:3000",
        }

        if self._worker_proc is not None and self._worker_proc.poll() is None:
            await self._dispatch_job(
                room_name,
                "google_meet",
                contact,
                boss,
                False,
                extra_metadata=gmeet_extra,
            )
        else:
            await self._start_call_subprocess(
                room_name,
                "google_meet",
                contact,
                boss,
                False,
                extra_env=gmeet_extra,
            )

        # Browser join runs after dispatch — fast brain initializes in parallel.
        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                f"{base_url}/googlemeet/join",
                json={"meetUrl": meet_url, "displayName": display_name},
                headers={"authorization": f"Bearer {auth_key}"},
                timeout=aiohttp.ClientTimeout(total=300),
            )
            body = await resp.json()

        if resp.status != 200:
            LOGGER.error(
                f"{ICONS['ipc']} [LivekitCallManager] Google Meet join failed: {body}",
            )
            self._gmeet_joining = False
            await self.cleanup_google_meet()
            return False

        self._gmeet_session_id = body.get("sessionId")
        self._gmeet_joining = False
        LOGGER.info(
            f"{ICONS['ipc']} [LivekitCallManager] Google Meet joined "
            f"(session={self._gmeet_session_id})",
        )

        if self._socket_server and self._gmeet_session_id:
            await self._socket_server.queue_for_clients(
                "app:call:status",
                json.dumps(
                    {"type": "gmeet_session_id", "session_id": self._gmeet_session_id},
                ),
            )

        return True

    async def cleanup_google_meet(self) -> None:
        """Leave the Google Meet session and tear down the audio bridge."""
        session_id = self._gmeet_session_id
        room_name = self.room_name
        self._gmeet_session_id = None
        self._gmeet_joining = False
        self.google_meet_start_timestamp = None
        self.google_meet_exchange_id = UNASSIGNED

        if session_id:
            base_url = "http://localhost:3000"
            auth_key = os.environ.get("UNIFY_KEY", "")
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        f"{base_url}/googlemeet/leave",
                        json={"sessionId": session_id},
                        headers={"authorization": f"Bearer {auth_key}"},
                        timeout=aiohttp.ClientTimeout(total=30),
                    )
            except Exception as exc:
                LOGGER.warning(
                    f"{ICONS['ipc']} [LivekitCallManager] "
                    f"Error leaving Google Meet: {exc}",
                )

        if room_name:
            from unity.conversation_manager.medium_scripts.common import (
                delete_livekit_room,
            )

            await delete_livekit_room(room_name)

        await self.cleanup_call_proc()

    async def _start_call_subprocess(
        self,
        room_name: str,
        channel: str,
        contact: dict,
        boss: dict,
        outbound: bool,
        *,
        extra_env: dict | None = None,
    ) -> None:
        """Legacy path: spawn a fresh subprocess per call."""
        socket_path = await self._ensure_socket_server()
        if extra_env:
            for k, v in extra_env.items():
                os.environ[k.upper()] = str(v)
        if socket_path:
            os.environ[CM_EVENT_SOCKET_ENV] = socket_path
            LOGGER.debug(
                f"{ICONS['ipc']} [LivekitCallManager] Socket server at {socket_path}",
            )
        target_path = (
            Path(__file__).parent.parent.resolve() / "medium_scripts" / "call.py"
        )
        args = [
            str(a)
            for a in [
                room_name,
                self.voice_provider,
                self.voice_id,
                outbound,
                channel,
                json.dumps(contact),
                json.dumps(boss),
                self.assistant_bio,
                self.assistant_id,
                self.user_id,
            ]
        ]
        LOGGER.debug(f"{DEFAULT_ICON} target_path: {target_path}, args: {args}")
        self._call_proc = run_script(str(target_path), "dev", *args)

    # -- IPC disconnect fallback (safety net for lost call-ended events) --
    async def _on_ipc_client_disconnected(self) -> None:
        """Called by the socket server when the last IPC client disconnects.

        If ``cleanup_call_proc`` hasn't already run (meaning the call-ended
        event was lost), wait a short grace period then publish a synthetic
        call-ended event so the normal event-handler path runs the cleanup.
        """
        if not self.has_active_call:
            return

        await asyncio.sleep(1)

        if not self.has_active_call:
            return

        contact = self._disconnect_contact or {}
        channel = self._call_channel or "phone_call"
        if channel == "whatsapp_call":
            event = WhatsAppCallEnded(contact=contact)
        elif channel == "google_meet":
            event = GoogleMeetEnded(contact=contact)
        elif channel == "phone_call":
            event = PhoneCallEnded(contact=contact)
        else:
            event = UnifyMeetEnded(contact=contact)
        LOGGER.debug(
            f"{ICONS['ipc']} [LivekitCallManager] IPC client disconnected without cleanup, "
            f"publishing fallback {event.__class__.__name__}",
        )
        if self._event_broker:
            await self._event_broker.publish(
                event.topic,
                event.to_json(),
            )

    async def cleanup_call_proc(self) -> None:
        """Stop any running voice agent job/subprocess and socket server."""
        proc = self._call_proc
        self._call_proc = None
        self._active_job = False
        self._call_pending = False

        self.is_outbound = False
        self.initial_notification = ""
        self._call_channel = None
        self._disconnect_contact = None

        if self._boss_notification_task and not self._boss_notification_task.done():
            self._boss_notification_task.cancel()
            try:
                await self._boss_notification_task
            except asyncio.CancelledError:
                pass
        self._boss_notification_task = None

        if self._socket_server:
            await self._socket_server.stop()
            self._socket_server = None

        if CM_EVENT_SOCKET_ENV in os.environ:
            del os.environ[CM_EVENT_SOCKET_ENV]

        if proc is None:
            return

        if proc.poll() is not None:
            LOGGER.debug(
                f"{ICONS['ipc']} [LivekitCallManager] Process already exited with code {proc.returncode}",
            )
            return

        LOGGER.debug(
            f"{ICONS['ipc']} [LivekitCallManager] Terminating voice agent process {proc.pid}...",
        )
        await asyncio.to_thread(terminate_process, proc, 5)
        LOGGER.debug(
            f"{ICONS['ipc']} [LivekitCallManager] Voice agent process terminated",
        )

    # ------------------------------------------------------------------
    # Symbolic event forwarding for system contact calls
    # ------------------------------------------------------------------

    def _start_boss_notification_rendering(self) -> None:
        """Start an async task that forwards actor events to the fast brain."""
        if self._boss_notification_task and not self._boss_notification_task.done():
            return
        self._boss_notification_task = asyncio.create_task(
            self._render_boss_notifications(),
        )

    async def _render_boss_notifications(self) -> None:
        """Subscribe to actor events and publish rendered notifications.

        Runs for system contact calls only. Converts raw actor lifecycle
        events into FastBrainNotification messages on
        ``app:call:notification`` so the fast brain receives them as
        immediate silent context — guaranteed delivery, zero LLM latency.
        The slow brain separately decides whether to speak via
        ``guide_voice_agent``.
        """
        from unity.conversation_manager.medium_scripts.common import (
            render_event_for_fast_brain,
        )

        try:
            async with self._event_broker.pubsub() as pubsub:
                await pubsub.psubscribe("app:actor:*")
                while True:
                    msg = await pubsub.get_message(
                        timeout=1.0,
                        ignore_subscribe_messages=True,
                    )
                    if msg is None:
                        continue
                    data = msg.get("data", "")
                    if not data:
                        continue
                    text = render_event_for_fast_brain(data)
                    if not text:
                        continue
                    notification = FastBrainNotification(
                        content=text,
                        source="system",
                        contact={},
                    )
                    await self._event_broker.publish(
                        "app:call:notification",
                        notification.to_json(),
                    )
        except asyncio.CancelledError:
            pass
