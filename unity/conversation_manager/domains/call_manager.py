from pathlib import Path
import sys

from unity.contact_manager.types.contact import UNASSIGNED
from unity.helpers import cleanup_dangling_call_processes, run_script, terminate_process
from unity.conversation_manager.events import *


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
        self.call_proc = None
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
        self.call_proc = run_script(str(target_path), "dev", *args)

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
