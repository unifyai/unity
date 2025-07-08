# actions.py
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import Deque, Dict, List

HISTORY_LEN = 20


@dataclass
class RichActionRecord:
    """A structured record of an executed command, including before/after screenshots."""

    timestamp: float
    command: str
    before_screenshot_b64: str
    after_screenshot_b64: str


@dataclass
class BrowserState:
    url: str = ""
    title: str = ""
    auto_scroll: str | None = None  # "up" | "down" | None
    in_textbox: bool = False
    scroll_y: int = 0
    scroll_speed: int = 250  # auto-scroll speed in pixels per second (default)
    # ────────────────────────────── Dialog & Pop-ups (NEW) ──────────────────────────────
    dialog_open: bool = False  # any JS dialog currently shown?
    dialog_type: str | None = None  # alert | confirm | prompt | beforeunload
    dialog_msg: str = ""  # the message text of the dialog
    popups: List[str] = field(default_factory=list)  # titles of open popup pages
    captcha_pending: bool = False  # Anti-Captcha busy flag
    can_go_back: bool = False
    can_go_forward: bool = False


class ActionHistory:
    def __init__(self, capacity: int = HISTORY_LEN) -> None:
        self._dq: Deque[RichActionRecord] = deque(maxlen=capacity)

    def add_record(self, record_data: Dict) -> None:
        """Creates and appends a RichActionRecord from a dictionary."""
        self._dq.append(RichActionRecord(**record_data))

    def dump(self) -> List[Dict]:
        """Dumps the history of rich action records as a list of dictionaries."""
        return [asdict(r) for r in self._dq]
