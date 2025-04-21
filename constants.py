"""
Centralised command / action literals so we never duplicate strings again.
Add to, or import from, this module instead of hard‑coding commands.
"""

# one‑off smooth scroll
CMD_SCROLL_UP = "scroll up"
CMD_SCROLL_DOWN = "scroll down"

# auto‑scroll
CMD_START_SCROLL_UP = "start scrolling up"
CMD_START_SCROLL_DOWN = "start scrolling down"
CMD_STOP_SCROLLING = "stop scrolling"
CMD_CONT_SCROLLING = "continue scrolling"

# tabs
CMD_NEW_TAB = "new tab"
CMD_CLOSE_TAB = "close tab"
CMD_SWITCH_TAB = "switch to tab"

# search / nav
CMD_SEARCH = "search"
CMD_OPEN_URL = "open url"
