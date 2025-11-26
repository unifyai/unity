from pathlib import Path
import sys

from unity.transcript_manager.types.message import UNASSIGNED
from unity.helpers import cleanup_dangling_call_processes, run_script, terminate_process
from unity.conversation_manager.new_events import *


class LivekitCallManager:
    def __init__(
        self,
        assistant_id=None,
        assistant_bio=None,
        assistant_number=None,
        voice_provider=None,
        voice_id=None,
        voice_mode=None,
    ):
        self.assistant_id = assistant_id
        self.assistant_bio = assistant_bio
        self.assistant_number = assistant_number
        self.voice_provider = voice_provider
        self.voice_id = voice_id
        self.realtime = voice_mode == "sts"
        self.call_proc = None

        self.call_exchange_id = UNASSIGNED
        self.call_start_timestamp = None
        self.conference_name = ""

    # TODO: support unify calls and clean up boss data passage
    def start_call(self, contact: dict, boss: dict, outbound: bool = False):
        target_path = Path(__file__).parent.parent.resolve() / "medium_scripts"
        args = [
            self.assistant_number,
            self.voice_provider,
            self.voice_id,
            outbound,
            json.dumps(contact),
        ]
        if self.realtime:
            args += [json.dumps(boss), self.assistant_bio]
            target_path = target_path / "realtime_call.py"
        else:
            target_path = target_path / "call.py"
        args = [str(arg) for arg in args]
        print(f"target_path: {target_path}, args: {args}")
        self.call_proc = run_script(str(target_path), "dev", *args)

    def start_unify_call(
        self,
        agent_name: str | None,
        room_name: str | None,
        contact: dict,
    ):
        target_path = (
            Path(__file__).parent.parent.resolve() / "medium_scripts" / "unify_call.py"
        )
        agent_name = (
            agent_name
            if agent_name
            else (
                f"unity_{self.assistant_id}_web"
                if self.assistant_id
                else "unity_unify_call_1"
            )
        )
        room_name = (
            room_name
            if room_name
            else (
                f"unity_{self.assistant_id}_web"
                if self.assistant_id
                else "unity_unify_call_1"
            )
        )
        args = [
            agent_name,
            room_name,
            self.voice_provider,
            self.voice_id,
            json.dumps(contact),
        ]
        args = [str(arg) for arg in args]
        print(f"target_path: {target_path}, args: {args}")
        self.call_proc = run_script(str(target_path), "dev", *args)

    def cleanup_call_proc(self):
        print(f"Terminating call process")
        try:
            if sys.platform.startswith("win"):
                terminate_process(self.call_proc, timeout=0.1)
            else:
                cleanup_dangling_call_processes()
            self.call_proc = None
            print(f"Call process terminated")
        except Exception as e:
            print(f"Error terminating call process: {e}")
