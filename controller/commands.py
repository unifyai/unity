"""
Centralised command / action literals.
Edit ONLY this file when adding, renaming or deleting a command.
Every other module must import from here.
"""

# ───────────────────────────────────────────────────────────────────────────
#  SINGLE‑COMMAND CONSTANTS (keep them alphabetically sorted, please)
# ───────────────────────────────────────────────────────────────────────────
CMD_OPEN_BROWSER = "open_browser"
CMD_CLOSE_BROWSER = "close_browser"
CMD_CLICK_OUT = "click_out"
CMD_CONT_SCROLLING = "continue_scrolling"
CMD_CURSOR_DOWN = "cursor_down"
CMD_CURSOR_LEFT = "cursor_left"
CMD_CURSOR_RIGHT = "cursor_right"
CMD_CURSOR_UP = "cursor_up"
CMD_ENTER_TEXT = "enter_text *"
CMD_HOLD_SHIFT = "hold_shift"
CMD_MOVE_LINE_END = "move_line_end"
CMD_MOVE_LINE_START = "move_line_start"
CMD_MOVE_WORD_LEFT = "move_word_left"
CMD_MOVE_WORD_RIGHT = "move_word_right"
CMD_NEW_TAB = "new_tab"
CMD_CLOSE_THIS_TAB = "close_this_tab"
CMD_PRESS_BACKSPACE = "press_backspace"
CMD_PRESS_DELETE = "press_delete"
CMD_PRESS_ENTER = "press_enter"
CMD_RELEASE_SHIFT = "release_shift"
CMD_SCROLL_DOWN = "scroll_down *"
CMD_SCROLL_UP = "scroll_up *"
CMD_SEARCH = "search *"
CMD_OPEN_URL = "open_url *"
CMD_SELECT_ALL = "select_all"
CMD_START_SCROLL_DOWN = "start_scrolling_down"
CMD_START_SCROLL_UP = "start_scrolling_up"
CMD_STOP_SCROLLING = "stop_scrolling"
CMD_CLICK_BUTTON = "click_button *"
CMD_SELECT_TAB = "select_tab *"
CMD_CLOSE_TAB = "close_tab *"
CMD_ACCEPT_DIALOG = "accept_dialog"
CMD_CLOSE_POPUP = "close_popup"
CMD_DISMISS_DIALOG = "dismiss_dialog"
CMD_SELECT_POPUP = "select_popup *"
CMD_TYPE_DIALOG = "type_dialog *"
CMD_BACK_NAV = "nav_back"
CMD_FORWARD_NAV = "nav_forward"
CMD_RELOAD_PAGE = "nav_reload"

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
    CMD_CLOSE_THIS_TAB,
    CMD_SEARCH,
    CMD_OPEN_URL,
    CMD_BACK_NAV,
    CMD_FORWARD_NAV,
    CMD_RELOAD_PAGE,
}

# 3.  Wildcard placeholders for dynamic GUI elements
BUTTON_PATTERNS: set[str] = {
    CMD_CLICK_BUTTON,
    CMD_SELECT_TAB,
    CMD_CLOSE_TAB,
    CMD_SELECT_POPUP,
}

# 4.  Single‑step smooth scroll patterns
SCROLL_PATTERNS = {
    "up": {CMD_SCROLL_UP, CMD_START_SCROLL_UP},
    "down": {CMD_SCROLL_DOWN, CMD_START_SCROLL_DOWN},
}

# 5.  Auto‑scroll controls grouped by state
AUTOSCROLL_START: set[str] = {CMD_START_SCROLL_UP, CMD_START_SCROLL_DOWN}
AUTOSCROLL_ACTIVE: set[str] = {CMD_STOP_SCROLLING, CMD_CONT_SCROLLING}

# 6.  Dialog-specific primitives (shown only when a JS dialog is open)
DIALOG_COMMANDS: set[str] = {
    CMD_ACCEPT_DIALOG,
    CMD_DISMISS_DIALOG,
    CMD_TYPE_DIALOG,
}

# 7.  Popup-window primitives (shown whenever pop-ups are present)
POPUP_COMMANDS: set[str] = {
    CMD_CLOSE_POPUP,
    CMD_SELECT_POPUP,
}

# Convenience export for everything (handy for schema generation etc.)
ALL_PRIMITIVES: set[str] = (
    TEXTBOX_COMMANDS
    | NAV_COMMANDS
    | SCROLL_PATTERNS["up"]
    | SCROLL_PATTERNS["down"]
    | AUTOSCROLL_START
    | AUTOSCROLL_ACTIVE
    | BUTTON_PATTERNS
    | DIALOG_COMMANDS
    | POPUP_COMMANDS
)
