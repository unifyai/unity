"""
Compute the set of *currently* valid low‑level primitives
given the live BrowserState.  No literals appear here – we
import every action name or pattern from constants.py.
"""

from actions import BrowserState
from constants import (
    TEXTBOX_COMMANDS,
    NAV_COMMANDS,
    SCROLL_PATTERNS,
    AUTOSCROLL_START,
    AUTOSCROLL_ACTIVE,
    BUTTON_PATTERNS,
)


def get_valid_actions(state: BrowserState) -> set[str]:
    """Return a wildcard‑aware set of command strings."""

    valid: set[str] = set()

    # ── text entry & key‑presses ─────────────────────────────────────────
    if state.in_textbox:
        valid.update(TEXTBOX_COMMANDS)

    # ── page navigation & search  (always allowed) ──────────────────────
    valid.update(NAV_COMMANDS)

    # ── smooth one‑off scrolls ──────────────────────────────────────────
    if state.scroll_y > 0:  # not at absolute top
        valid.add(SCROLL_PATTERNS["up"])
    valid.add(SCROLL_PATTERNS["down"])  # you can always scroll further down

    # ── auto‑scroll controls ────────────────────────────────────────────
    if state.auto_scroll is None:
        valid.update(AUTOSCROLL_START)  # allow starting auto‑scroll
    else:
        valid.update(AUTOSCROLL_ACTIVE)  # allow stop / continue

    # ── dynamic tab & button placeholders (wildcards kept) ──────────────
    valid.update(BUTTON_PATTERNS)

    return valid
