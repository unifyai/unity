from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from unity.contact_manager.types.contact import UNASSIGNED
from unity.conversation_manager.events import *
from unity.helpers import (
    cleanup_dangling_call_processes,
    run_script,
    terminate_process,
)


@dataclass
class CallConfig:
    assistant_id: str
    assistant_bio: str
    assistant_number: str
    voice_provider: str
    voice_id: str
    voice_mode: str


class LivekitCallManager:
    def __init__(self, config: CallConfig):
        self.set_config(config=config)
        self.call_exchange_id = UNASSIGNED
        self.unify_meet_exchange_id = UNASSIGNED
        self.call_start_timestamp = None
        self.unify_meet_start_timestamp = None
        self.call_contact = None
        self._call_proc: subprocess.Popen | None = None
        self.conference_name = ""

    def set_config(self, config: CallConfig):
        self.assistant_id = config.assistant_id
        self.assistant_bio = config.assistant_bio
        self.assistant_number = config.assistant_number
        self.voice_provider = config.voice_provider
        self.voice_id = config.voice_id
        self.uses_realtime_api = config.voice_mode == "sts"

    def start_call(self, contact: dict, boss: dict, outbound: bool = False):
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

    def start_unify_meet(
        self,
        contact: dict,
        boss: dict,
        livekit_agent_name: str | None,
        room_name: str | None,
    ):
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
        Stop any running voice agent subprocess.

        Sends SIGTERM for graceful shutdown, then SIGKILL if needed.
        """
        proc = self._call_proc
        self._call_proc = None
        if proc is None:
            return

        # Check if process is still running
        if proc.poll() is not None:
            print(f"[LivekitCallManager] Process already exited with code {proc.returncode}")
            return

        print(f"[LivekitCallManager] Terminating voice agent process {proc.pid}...")
        if sys.platform.startswith("win"):
            await asyncio.to_thread(terminate_process, proc, timeout)
        else:
            await asyncio.to_thread(cleanup_dangling_call_processes)
        print("[LivekitCallManager] Voice agent process terminated")
