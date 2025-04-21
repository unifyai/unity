from actions import BrowserState
from constants import (
    CMD_SCROLL_UP,
    CMD_SCROLL_DOWN,
    CMD_START_SCROLL_UP,
    CMD_START_SCROLL_DOWN,
    CMD_STOP_SCROLLING,
    CMD_CONT_SCROLLING,
)


# ------------------------------------------------------------------
# All logic in *one* place; GUI, Agent & hot‑keys share the same list
# ------------------------------------------------------------------
def get_valid_actions(state: BrowserState) -> set[str]:
    valid: set[str] = set()

    # ── text entry & key‑presses ───────────────────────────────────
    if state.in_textbox:
        valid.update(
            {
                "click out",
                "enter text",
                "press enter",
                "press backspace",
                "press delete",
                "cursor left",
                "cursor right",
                "cursor up",
                "cursor down",
                "select all",
                "move line start",
                "move line end",
                "move word left",
                "move word right",
                "hold shift",
                "release shift",
            },
        )

    # ── page navigation & search  (always allowed) ─────────────────
    valid.update({"new tab", "search action", "search url action"})

    # ── smooth one‑off scrolls ─────────────────────────────────────
    if state.scroll_y > 0:  # at least one line above top
        valid.add(CMD_SCROLL_UP)
    valid.add(CMD_SCROLL_DOWN)  # cheap heuristic – always ok

    # ── auto‑scroll controls ───────────────────────────────────────
    if state.auto_scroll is None:  # not scrolling yet
        valid.update({CMD_START_SCROLL_UP, CMD_START_SCROLL_DOWN})
    else:  # already scrolling
        valid.update({CMD_STOP_SCROLLING, CMD_CONT_SCROLLING})

    # ── dynamic tab / button placeholders – keep wildcards ─────────
    valid.update({"click button *", "select tab *", "close tab *"})

    return valid
