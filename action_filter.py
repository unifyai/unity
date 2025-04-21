"""
Compute the set of *currently* valid low‑level primitives
given the live BrowserState.  No literals appear here – we
import every action name or pattern from constants.py.
"""

from actions import BrowserState
from typing import Union
from constants import *


def get_valid_actions(state: Union[BrowserState, dict]) -> set[str]:
    """Return a wildcard‑aware set of command strings."""

    # Accept either a BrowserState OR the raw dict we send across Tk queues
    if not isinstance(state, BrowserState):
        state = BrowserState(**state)

    valid: set[str] = set()

    # ── text entry & key‑presses ─────────────────────────────────────────
    if state.in_textbox:
        valid.update(TEXTBOX_COMMANDS)
        return valid

    # ── scrolling ──────────────────────────────────────────
    if state.auto_scroll is None:
        valid.update({CMD_SCROLL_DOWN, CMD_START_SCROLL_DOWN})
        if state.scroll_y > 0:
            valid.update({CMD_SCROLL_UP, CMD_START_SCROLL_UP})
    elif state.auto_scroll == "up":
        valid.update({CMD_STOP_SCROLLING, CMD_START_SCROLL_DOWN})
        return valid
    elif state.auto_scroll == "down":
        valid.update({CMD_STOP_SCROLLING, CMD_START_SCROLL_UP})
        return valid
    else:
        raise Exception("Invalid auto_scroll state.")

    # ── page navigation & search ──────────────────────
    valid.update(NAV_COMMANDS)

    # ── dynamic tab & button placeholders ──────────────
    valid.update(BUTTON_PATTERNS)

    underscore_aliases = {v.replace(" ", "_") for v in valid}
    return valid | underscore_aliases
