"""
Compute the set of *currently* valid low‑level primitives
given the live BrowserState.  No literals appear here – we
import every action name or pattern from constants.py.
"""

from controller.states import BrowserState
from typing import Union
from controller.commands import *


def get_valid_actions(state: Union[BrowserState, dict], mode="both") -> set[str]:
    """Return a wildcard‑aware set of command strings."""

    # Accept either a BrowserState OR the raw dict we send across Tk queues
    if not isinstance(state, BrowserState):
        state = BrowserState(**state)

    # ── Blocker: JavaScript dialog visible ─────────────────────────
    if state.dialog_open:
        dlg_cmds = {CMD_ACCEPT_DIALOG, CMD_DISMISS_DIALOG}
        if state.dialog_type == "prompt":
            dlg_cmds.add(CMD_TYPE_DIALOG)

        if mode == "schema":
            return {c.replace(" ", "_") for c in dlg_cmds}
        if mode == "actions":
            return dlg_cmds
        if mode == "both":
            return dlg_cmds | {c.replace(" ", "_") for c in dlg_cmds}

    # ── Blocker: CAPTCHA solving in progress ─────────────────────────
    if state.captcha_pending:
        # while solver runs, do not allow new actions to avoid accidental nav
        # Users can still scroll gently if desired.
        return {CMD_SCROLL_UP, CMD_SCROLL_DOWN, CMD_STOP_SCROLLING}

    valid: set[str] = set()

    # ── text entry & key-presses ─────────────────────────────────────────
    if state.in_textbox:
        # Keep textbox-specific commands, but do NOT exit early – still allow
        # scrolling, clicking buttons, etc. while the caret is inside a box.
        valid.update(TEXTBOX_COMMANDS)

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

    # ── popup window commands ──────────────────────────
    if state.popups:
        valid.update({CMD_SELECT_POPUP, CMD_CLOSE_POPUP})

    if mode == "schema":
        ret = {v.replace(" ", "_") for v in valid}
        return {
            (
                v.replace("_*", "")
                if v in ("open_url_*", "search_*", "scroll_down_*", "scroll_up_*")
                else v
            )
            for v in ret
        }
    elif mode == "actions":
        return valid
    elif mode == "both":
        return valid | {v.replace(" ", "_") for v in valid}
    else:
        raise Exception("Invalid mode")
