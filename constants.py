"""
Centralised command / action literals.
Edit ONLY this file when adding, renaming or deleting a command.
Every other module must import from here.
"""

# ───────────────────────────────────────────────────────────────────────────
#  SINGLE‑COMMAND CONSTANTS (keep them alphabetically sorted, please)
# ───────────────────────────────────────────────────────────────────────────
CMD_CLICK_OUT = "click out"
CMD_CONT_SCROLLING = "continue scrolling"
CMD_CURSOR_DOWN = "cursor down"
CMD_CURSOR_LEFT = "cursor left"
CMD_CURSOR_RIGHT = "cursor right"
CMD_CURSOR_UP = "cursor up"
CMD_ENTER_TEXT = "enter text *"
CMD_HOLD_SHIFT = "hold shift"
CMD_MOVE_LINE_END = "move line end"
CMD_MOVE_LINE_START = "move line start"
CMD_MOVE_WORD_LEFT = "move word left"
CMD_MOVE_WORD_RIGHT = "move word right"
CMD_NEW_TAB = "new tab"
CMD_CLOSE_THIS_TAB = "close this tab"
CMD_PRESS_BACKSPACE = "press backspace"
CMD_PRESS_DELETE = "press delete"
CMD_PRESS_ENTER = "press enter"
CMD_RELEASE_SHIFT = "release shift"
CMD_SCROLL_DOWN = "scroll down *"
CMD_SCROLL_UP = "scroll up *"
CMD_SEARCH = "search *"
CMD_OPEN_URL = "open url *"
CMD_SELECT_ALL = "select all"
CMD_START_SCROLL_DOWN = "start scrolling down"
CMD_START_SCROLL_UP = "start scrolling up"
CMD_STOP_SCROLLING = "stop scrolling"
CMD_CLICK_BUTTON = "click button *"
CMD_SELECT_TAB = "select tab *"
CMD_CLOSE_TAB = "close tab *"

# ───────────────────────────────────────────────────────────────────────────
#  WILDCARD GROUPS (sets reused by GUI / agent / filters)
# ───────────────────────────────────────────────────────────────────────────
# 1.  Buttons only valid inside a text‑box
TEXTBOX_COMMANDS: set[str] = {
    CMD_CLICK_OUT,
    CMD_ENTER_TEXT,
    CMD_PRESS_ENTER,
    CMD_PRESS_BACKSPACE,
    CMD_PRESS_DELETE,
    CMD_CURSOR_LEFT,
    CMD_CURSOR_RIGHT,
    CMD_CURSOR_UP,
    CMD_CURSOR_DOWN,
    CMD_SELECT_ALL,
    CMD_MOVE_LINE_START,
    CMD_MOVE_LINE_END,
    CMD_MOVE_WORD_LEFT,
    CMD_MOVE_WORD_RIGHT,
    CMD_HOLD_SHIFT,
    CMD_RELEASE_SHIFT,
}

# 2.  Page navigation / search (always valid)
NAV_COMMANDS: set[str] = {
    CMD_NEW_TAB,
    CMD_SEARCH,
    CMD_OPEN_URL,
}

# 3.  Wildcard placeholders for dynamic GUI elements
BUTTON_PATTERNS: set[str] = {
    CMD_CLICK_BUTTON,
    CMD_SELECT_TAB,
    CMD_CLOSE_TAB,
}

# 4.  Single‑step smooth scroll patterns
SCROLL_PATTERNS = {
    "up": CMD_SCROLL_UP,
    "down": CMD_SCROLL_DOWN,
}

# 5.  Auto‑scroll controls grouped by state
AUTOSCROLL_START: set[str] = {CMD_START_SCROLL_UP, CMD_START_SCROLL_DOWN}
AUTOSCROLL_ACTIVE: set[str] = {CMD_STOP_SCROLLING, CMD_CONT_SCROLLING}

# Convenience export for everything (handy for schema generation etc.)
ALL_PRIMITIVES: set[str] = (
    TEXTBOX_COMMANDS
    | NAV_COMMANDS
    | set(SCROLL_PATTERNS.values())
    | AUTOSCROLL_START
    | AUTOSCROLL_ACTIVE
    | BUTTON_PATTERNS
)
