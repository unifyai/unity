from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from unity.contact_manager.types.contact import UNASSIGNED
from unity.conversation_manager.events import *
from unity.conversation_manager.domains.ipc_socket import (
    CallEventSocketServer,
    CM_EVENT_SOCKET_ENV,
)
from unity.conversation_manager.tracing import content_trace_id, trace_kv
from unity.helpers import (
    cleanup_dangling_call_processes,
    run_script,
    terminate_process,
)

if TYPE_CHECKING:
    from unity.conversation_manager.in_memory_event_broker import InMemoryEventBroker


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
    voice_mode: str


_BASE_FORWARD_CHANNELS = [
    "app:call:*",
    "app:comms:*",
]
_BOSS_EXTRA_CHANNELS = [
    "app:actor:*",
    "app:managers:output",
    "app:logging:message_logged",
]


class LivekitCallManager:
    def __init__(
        self,
        config: CallConfig,
        event_broker: "InMemoryEventBroker | None" = None,
    ):
        self.set_config(config=config)
        self.call_exchange_id = UNASSIGNED
        self.unify_meet_exchange_id = UNASSIGNED
        self.call_start_timestamp = None
        self.unify_meet_start_timestamp = None
        self.call_contact = None
        self._call_proc: subprocess.Popen | None = None
        self.conference_name = ""
        self.room_name = ""
        self._event_broker = event_broker
        self._socket_server: CallEventSocketServer | None = None
        # Track whether the current call is outbound (we initiated it)
        self.is_outbound: bool = False
        # Initial guidance for outbound calls, set by make_call tool before the
        # call is placed, published to the fast brain after the subprocess spawns.
        self.initial_call_guidance: str = ""
        # Callback for screenshots (user or assistant) received via IPC.
        # Set by the ConversationManager to route screenshots to its buffer.
        self.on_screenshot: Callable[[str], None] | None = None
        # Track the active call's channel type so the disconnect fallback
        # can publish the correct call-ended event.
        self._call_channel: str | None = None
        # Contact for the active call, used by the disconnect fallback.
        self._disconnect_contact: dict | None = None

    def set_config(self, config: CallConfig):
        self.assistant_id = config.assistant_id
        self.user_id = config.user_id
        self.assistant_bio = config.assistant_bio
        self.assistant_number = config.assistant_number
        self.voice_provider = config.voice_provider
        self.voice_id = config.voice_id
        self.uses_realtime_api = config.voice_mode == "sts"

    def set_event_broker(self, event_broker: "InMemoryEventBroker") -> None:
        """Set the event broker for socket server to publish to."""
        self._event_broker = event_broker

    async def _ensure_socket_server(self) -> str | None:
        """Start the socket server if not running, return socket path."""
        if self._event_broker is None:
            print(
                "[LivekitCallManager] Warning: No event broker set, socket IPC disabled",
            )
            return None

        if self._socket_server is None:

            async def _on_ipc_event(channel: str, event_json: str) -> None:
                if channel == "app:comms:screenshot" and self.on_screenshot is not None:
                    self.on_screenshot(event_json)
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

    async def start_call(self, contact: dict, boss: dict, outbound: bool = False):
        # Track whether this is an outbound call
        self.is_outbound = outbound
        self._call_channel = "phone"

        # Start socket server and get path
        socket_path = await self._ensure_socket_server()

        # All calls get comms events; boss calls additionally get actor/manager events.
        if self._socket_server:
            is_boss = contact.get("contact_id") == 1
            channels = (
                _BASE_FORWARD_CHANNELS + _BOSS_EXTRA_CHANNELS
                if is_boss
                else list(_BASE_FORWARD_CHANNELS)
            )
            await self._socket_server.set_forward_channels(channels)

        # Set socket path in environment for subprocess
        if socket_path:
            os.environ[CM_EVENT_SOCKET_ENV] = socket_path
            print(f"[LivekitCallManager] Socket server at {socket_path}")

        target_path = Path(__file__).parent.parent.resolve() / "medium_scripts"
        # Both TTS and Realtime modes use the fast brain architecture and need
        # boss details and assistant bio for the phone agent prompt
        args = [
            make_room_name(self.assistant_id, "phone"),
            self.voice_provider,
            self.voice_id,
            outbound,
            "phone",
            json.dumps(contact),
            json.dumps(boss),
            self.assistant_bio,
            self.assistant_id,
            self.user_id,
        ]
        if self.uses_realtime_api:
            target_path = target_path / "sts_call.py"
        else:
            target_path = target_path / "call.py"
        args = [str(arg) for arg in args]
        print(f"target_path: {target_path}, args: {args}")
        self._call_proc = run_script(str(target_path), "dev", *args)
        self._disconnect_contact = contact

        # Deliver initial guidance to the fast brain (if any was stored by
        # make_call).  We bypass the event-broker pub/sub roundtrip and push
        # directly into the socket server buffer so the message cannot be lost
        # due to the forward-subscription task not having subscribed yet.
        if self.initial_call_guidance:
            guidance_id = content_trace_id("guid", self.initial_call_guidance)
            guidance_event = CallGuidance(
                contact=contact,
                content=self.initial_call_guidance,
                source="initial_call",
            )
            # Direct socket delivery to the fast brain subprocess
            await self._socket_server.queue_for_clients(
                "app:call:call_guidance",
                guidance_event.to_json(),
            )
            # Also publish on the comms channel for the transcript / UI
            await self._event_broker.publish(
                "app:comms:assistant_call_guidance",
                guidance_event.to_json(),
            )
            print(
                trace_kv(
                    "CALL_MANAGER_INITIAL_GUIDANCE",
                    guidance_id=guidance_id,
                    content_preview=self.initial_call_guidance[:80],
                ),
                flush=True,
            )
            self.initial_call_guidance = ""

    async def start_unify_meet(
        self,
        contact: dict,
        boss: dict,
        room_name: str | None,
    ):
        # Unify Meet is always inbound (user initiates)
        self.is_outbound = False
        self._call_channel = "unify"

        # Start socket server and get path
        socket_path = await self._ensure_socket_server()

        # All calls get comms events; boss calls additionally get actor/manager events.
        if self._socket_server:
            is_boss = contact.get("contact_id") == 1
            channels = (
                _BASE_FORWARD_CHANNELS + _BOSS_EXTRA_CHANNELS
                if is_boss
                else list(_BASE_FORWARD_CHANNELS)
            )
            await self._socket_server.set_forward_channels(channels)

        # Set socket path in environment for subprocess
        if socket_path:
            os.environ[CM_EVENT_SOCKET_ENV] = socket_path
            print(f"[LivekitCallManager] Socket server at {socket_path}")

        target_path = Path(__file__).parent.parent.resolve() / "medium_scripts"
        room_name = room_name or make_room_name(self.assistant_id, "meet")
        self.room_name = room_name
        # Both TTS and Realtime modes use the fast brain architecture and need
        # boss details and assistant bio for the phone agent prompt
        args = [
            room_name,
            self.voice_provider,
            self.voice_id,
            False,
            "unify",
            json.dumps(contact),
            json.dumps(boss),
            self.assistant_bio,
            self.assistant_id,
            self.user_id,
        ]
        if self.uses_realtime_api:
            target_path = target_path / "sts_call.py"
        else:
            target_path = target_path / "call.py"
        args = [str(arg) for arg in args]
        print(f"target_path: {target_path}, args: {args}")
        self._call_proc = run_script(str(target_path), "dev", *args)
        self._disconnect_contact = contact

    # -- IPC disconnect fallback (safety net for lost call-ended events) --
    async def _on_ipc_client_disconnected(self) -> None:
        """Called by the socket server when the last IPC client disconnects.

        If ``cleanup_call_proc`` hasn't already run (meaning the call-ended
        event was lost), wait a short grace period then publish a synthetic
        call-ended event so the normal event-handler path runs the cleanup.
        """
        if self._call_proc is None:
            return  # already cleaned up

        await asyncio.sleep(1)

        if self._call_proc is None:
            return  # cleaned up during grace period

        contact = self._disconnect_contact or {}
        channel = self._call_channel or "phone"
        event = (
            PhoneCallEnded(contact=contact)
            if channel == "phone"
            else UnifyMeetEnded(contact=contact)
        )
        print(
            f"[LivekitCallManager] IPC client disconnected without cleanup, "
            f"publishing fallback {event.__class__.__name__}",
            flush=True,
        )
        if self._event_broker:
            await self._event_broker.publish(
                f"app:comms:{channel}_call_ended",
                event.to_json(),
            )

    async def cleanup_call_proc(self, *, timeout: float = 5.0) -> None:
        """
        Stop any running voice agent subprocess and socket server.

        Sends SIGTERM for graceful shutdown, then SIGKILL if needed.
        """
        # Grab the proc ref and null it out FIRST.  stop() below awaits
        # _handle_client's finally block, which fires the disconnect
        # callback -- that callback is a no-op when _call_proc is None.
        proc = self._call_proc
        self._call_proc = None

        # Reset outbound tracking
        self.is_outbound = False
        self.initial_call_guidance = ""
        self._call_channel = None
        self._disconnect_contact = None

        # Stop socket server
        if self._socket_server:
            await self._socket_server.stop()
            self._socket_server = None

        # Clean up environment variable
        if CM_EVENT_SOCKET_ENV in os.environ:
            del os.environ[CM_EVENT_SOCKET_ENV]

        if proc is None:
            return

        # Check if process is still running
        if proc.poll() is not None:
            print(
                f"[LivekitCallManager] Process already exited with code {proc.returncode}",
            )
            return

        print(f"[LivekitCallManager] Terminating voice agent process {proc.pid}...")
        if sys.platform.startswith("win"):
            await asyncio.to_thread(terminate_process, proc, timeout)
        else:
            await asyncio.to_thread(cleanup_dangling_call_processes)
        print("[LivekitCallManager] Voice agent process terminated")
