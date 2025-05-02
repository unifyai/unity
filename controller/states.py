# actions.py
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Deque, Dict, List, Optional

HISTORY_LEN = 100  # keep the last 100 low‑level commands


@dataclass
class ActionRecord:
    timestamp: float
    command: str


@dataclass
class BrowserState:
    url: str = ""
    title: str = ""
    auto_scroll: str | None = None  # "up" | "down" | None
    in_textbox: bool = False
    scroll_y: int = 0
    dom_sha: Optional[str] = None
    focused_xpath: Optional[str] = None


class ActionHistory:
    def __init__(self, capacity: int = HISTORY_LEN) -> None:
        self._dq: Deque[ActionRecord] = deque(maxlen=capacity)

    def add(self, cmd: str) -> None:
        self._dq.append(ActionRecord(datetime.now().timestamp(), cmd))

    def dump(self) -> List[Dict]:
        return [asdict(r) for r in self._dq]
