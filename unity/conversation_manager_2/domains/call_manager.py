from unity.transcript_manager.types.message import UNASSIGNED
from unity.helpers import run_script, terminate_process


class LivekitCallManager:
    def __init__(self, realtime: bool = False):
        self.realtime = realtime
        self.call_proc = None

        self.call_exchange_id = UNASSIGNED
        self.call_start_timestamp = None
        self.conference_name = ""

    def start_call(self):
        ...
    
    def cleanup_call_proc(self):
        ...