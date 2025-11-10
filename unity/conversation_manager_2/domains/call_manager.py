from pathlib import Path

from unity.transcript_manager.types.message import UNASSIGNED
from unity.helpers import run_script, terminate_process


class LivekitCallManager:
    def __init__(self, assistant_number=None, voice_provider=None, voice_id=None, realtime: bool = False):
        self.assistant_number = assistant_number
        self.voice_provider = voice_provider
        self.voice_id = voice_id
        self.realtime = realtime
        self.call_proc = None

        self.call_exchange_id = UNASSIGNED
        self.call_start_timestamp = None
        self.conference_name = ""

    # TODO: support unify calls and clean up boss data passage
    def start_call(self, user_contact):
        target_path = Path(__file__).parent.resolve() / "medium_scripts"
        if self.realtime:
            self.call_proc = run_script(
                str(target_path),
                "dev",
                user_contact.phone_number,
                self.assistant_number,
                str(False),
                # contact details
                str(user_contact.is_boss),
                str(user_contact.first_name),
                str(user_contact.surname),
                str(user_contact.email_address),
                # boss user details (env vars are enough actually?)
                # str(boss.first_name),
                # str(boss.surname),
                # str(boss.phone_number),
                # str(boss.email_address),
            )
        else:
            target_path = target_path / "call.py"
            args = [
                user_contact.phone_number,
                self.assistant_number,
                self.voice_provider,
                self.voice_id if self.voice_id else "None",
                "None",
                str(False),
            ]
            print(f"target_path: {target_path}, args: {args}")
        self.call_proc = run_script(str(target_path), "dev", *args)
    
    def cleanup_call_proc(self):
        ...