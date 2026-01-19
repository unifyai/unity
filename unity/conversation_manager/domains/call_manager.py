from __future__ import annotations

import asyncio
import runpy
import threading
from pathlib import Path

from unity.contact_manager.types.contact import UNASSIGNED
from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.events import *

# Preload LiveKit OpenAI plugin on the main thread.
# LiveKit requires plugins to be registered on the main thread, but the voice
# agent script runs in a background thread. Importing here ensures the plugin
# registration happens before the thread is spawned.
try:
    from livekit.plugins.openai import (
        realtime as _openai_realtime_preload,
    )  # noqa: F401
except ImportError:
    # livekit-plugins-openai is optional; STS mode will fail at runtime if missing
    pass


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
        self._call_thread: threading.Thread | None = None
        self.conference_name = ""

    def set_config(self, config: CallConfig):
        self.assistant_id = config.assistant_id
        self.assistant_bio = config.assistant_bio
        self.assistant_number = config.assistant_number
        self.voice_provider = config.voice_provider
        self.voice_id = config.voice_id
        self.uses_realtime_api = config.voice_mode == "sts"

    def _start_script_thread(self, *, script_path: Path, argv: list[str]) -> None:
        """
        Run the LiveKit voice agent script *in-process* on a background thread.

        This replaces the previous subprocess-based containment so the voice
        agent can share the same in-memory event broker.
        """

        def _runner() -> None:
            import sys as _sys

            old_argv = list(_sys.argv)
            try:
                _sys.argv = [str(script_path), *argv]
                runpy.run_path(str(script_path), run_name="__main__")
            except SystemExit:
                # Click-based CLIs use SystemExit for normal termination.
                pass
            except Exception as e:
                print(f"[LivekitCallManager] Voice agent crashed: {e}")
            finally:
                _sys.argv = old_argv

        # Best-effort: stop any previously running agent thread.
        # (In normal operation there should only be one active call.)
        if self._call_thread and self._call_thread.is_alive():
            print("[LivekitCallManager] Warning: call thread already running")

        t = threading.Thread(
            target=_runner,
            name="LivekitVoiceAgent",
            daemon=True,
        )
        self._call_thread = t
        t.start()

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
        self._start_script_thread(script_path=target_path, argv=["dev", *args])

    def start_unify_meet(
        self,
        contact: dict,
        boss: dict,
        agent_name: str | None,
        room_name: str | None,
    ):
        target_path = Path(__file__).parent.parent.resolve() / "medium_scripts"
        agent_name = (
            agent_name
            if agent_name
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
            f"{agent_name}:{room_name}",
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
        self._start_script_thread(script_path=target_path, argv=["dev", *args])

    async def cleanup_call_proc(self, *, timeout: float = 10.0) -> None:
        """
        Stop any running in-process voice agent thread.

        We signal the agent via the shared event broker (app:call:status) and
        then join the thread with a timeout.
        """
        t = self._call_thread
        self._call_thread = None
        if t is None:
            return

        try:
            # Notify the voice agent to stop (handled by both TTS and STS scripts).
            await get_event_broker().publish(
                "app:call:status",
                json.dumps({"type": "stop"}),
            )
        except Exception:
            pass

        if t.is_alive():
            try:
                await asyncio.to_thread(t.join, timeout)
            except Exception:
                pass
            if t.is_alive():
                print(
                    f"[LivekitCallManager] Warning: voice agent thread did not exit within {timeout}s",
                )
