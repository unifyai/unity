from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from unity.contact_manager.types.contact import UNASSIGNED
from unity.conversation_manager.events import *
from unity.conversation_manager.domains.ipc_socket import (
    CallEventSocketServer,
    CM_EVENT_SOCKET_ENV,
)
from unity.helpers import (
    cleanup_dangling_call_processes,
    run_script,
    terminate_process,
)

if TYPE_CHECKING:
    from unity.conversation_manager.in_memory_event_broker import InMemoryEventBroker


@dataclass
class CallConfig:
    assistant_id: str
    assistant_bio: str
    assistant_number: str
    voice_provider: str
    voice_id: str
    voice_mode: str


class LivekitCallManager:
    def __init__(
        self, config: CallConfig, event_broker: "InMemoryEventBroker | None" = None
    ):
        self.set_config(config=config)
        self.call_exchange_id = UNASSIGNED
        self.unify_meet_exchange_id = UNASSIGNED
        self.call_start_timestamp = None
        self.unify_meet_start_timestamp = None
        self.call_contact = None
        self._call_proc: subprocess.Popen | None = None
        self.conference_name = ""
        self._event_broker = event_broker
        self._socket_server: CallEventSocketServer | None = None

    def set_config(self, config: CallConfig):
        self.assistant_id = config.assistant_id
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
                "[LivekitCallManager] Warning: No event broker set, socket IPC disabled"
            )
            return None

        if self._socket_server is None:
            self._socket_server = CallEventSocketServer(self._event_broker)

        if self._socket_server.socket_path is None:
            socket_path = await self._socket_server.start()
            return socket_path

        return self._socket_server.socket_path

    async def start_call(self, contact: dict, boss: dict, outbound: bool = False):
        # Start socket server and get path
        socket_path = await self._ensure_socket_server()

        # Set socket path in environment for subprocess
        if socket_path:
            os.environ[CM_EVENT_SOCKET_ENV] = socket_path
            print(f"[LivekitCallManager] Socket server at {socket_path}")

        target_path = Path(__file__).parent.parent.resolve() / "medium_scripts"
        # Both TTS and Realtime modes use the fast brain architecture and need
        # boss details and assistant bio for the phone agent prompt
        args = [
            self.assistant_number,
            self.voice_provider,
            self.voice_id,
            outbound,
            "phone",
            json.dumps(contact),
            json.dumps(boss),
            self.assistant_bio,
        ]
        if self.uses_realtime_api:
            target_path = target_path / "sts_call.py"
        else:
            target_path = target_path / "call.py"
        args = [str(arg) for arg in args]
        print(f"target_path: {target_path}, args: {args}")
        self._call_proc = run_script(str(target_path), "dev", *args)

    async def start_unify_meet(
        self,
        contact: dict,
        boss: dict,
        livekit_agent_name: str | None,
        room_name: str | None,
    ):
        # Start socket server and get path
        socket_path = await self._ensure_socket_server()

        # Set socket path in environment for subprocess
        if socket_path:
            os.environ[CM_EVENT_SOCKET_ENV] = socket_path
            print(f"[LivekitCallManager] Socket server at {socket_path}")

        target_path = Path(__file__).parent.parent.resolve() / "medium_scripts"
        livekit_agent_name = (
            livekit_agent_name
            if livekit_agent_name
            else (
                f"unity_{self.assistant_id}_web"
                if self.assistant_id
                else "unity_unify_meet_1"
            )
        )
        room_name = (
            room_name
            if room_name
            else (
                f"unity_{self.assistant_id}_web"
                if self.assistant_id
                else "unity_unify_meet_1"
            )
        )
        # Both TTS and Realtime modes use the fast brain architecture and need
        # boss details and assistant bio for the phone agent prompt
        args = [
            f"{livekit_agent_name}:{room_name}",
            self.voice_provider,
            self.voice_id,
            False,
            "unify",
            json.dumps(contact),
            json.dumps(boss),
            self.assistant_bio,
        ]
        if self.uses_realtime_api:
            target_path = target_path / "sts_call.py"
        else:
            target_path = target_path / "call.py"
        args = [str(arg) for arg in args]
        print(f"target_path: {target_path}, args: {args}")
        self._call_proc = run_script(str(target_path), "dev", *args)

    async def cleanup_call_proc(self, *, timeout: float = 5.0) -> None:
        """
        Stop any running voice agent subprocess and socket server.

        Sends SIGTERM for graceful shutdown, then SIGKILL if needed.
        """
        # Stop socket server first
        if self._socket_server:
            await self._socket_server.stop()
            self._socket_server = None

        # Clean up environment variable
        if CM_EVENT_SOCKET_ENV in os.environ:
            del os.environ[CM_EVENT_SOCKET_ENV]

        proc = self._call_proc
        self._call_proc = None
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
