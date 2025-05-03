"""
Helper functions to create Primitive actions from controller command literals.
Each function corresponds to a CMD_* constant and returns a Primitive instance.
"""

import queue
import threading
from typing import Callable, Optional
from .model import Primitive

from controller.commands import (
    CMD_OPEN_BROWSER,
    CMD_CLOSE_BROWSER,
    CMD_CLICK_OUT,
    CMD_CONT_SCROLLING,
    CMD_CURSOR_DOWN,
    CMD_CURSOR_LEFT,
    CMD_CURSOR_RIGHT,
    CMD_CURSOR_UP,
    CMD_ENTER_TEXT,
    CMD_HOLD_SHIFT,
    CMD_MOVE_LINE_END,
    CMD_MOVE_LINE_START,
    CMD_MOVE_WORD_LEFT,
    CMD_MOVE_WORD_RIGHT,
    CMD_NEW_TAB,
    CMD_CLOSE_THIS_TAB,
    CMD_PRESS_BACKSPACE,
    CMD_PRESS_DELETE,
    CMD_PRESS_ENTER,
    CMD_RELEASE_SHIFT,
    CMD_SCROLL_DOWN,
    CMD_SCROLL_UP,
    CMD_SEARCH,
    CMD_OPEN_URL,
    CMD_SELECT_ALL,
    CMD_START_SCROLL_DOWN,
    CMD_START_SCROLL_UP,
    CMD_STOP_SCROLLING,
    CMD_CLICK_BUTTON,
    CMD_SELECT_TAB,
    CMD_CLOSE_TAB,
    CMD_DUPLICATE_TAB,
    CMD_GO_BACK,
    CMD_GO_FORWARD,
)

# Queue for sending commands to the controller
_text_q = None
# Queue for receiving acknowledgments from the controller
_ack_q = None
# Event for pausing primitive execution
_pause_event: Optional[threading.Event] = None
# Queue for scheduling callables to run during pauses
_call_queue: Optional["queue.Queue[Callable]"] = None


def set_queues(text_q: queue.Queue, ack_q: queue.Queue):
    """Set the queues for sending commands and receiving acknowledgments."""
    global _text_q, _ack_q
    _text_q = text_q
    _ack_q = ack_q


def set_runtime_controls(
    pause_event: threading.Event, call_queue: "queue.Queue[Callable]"
):
    """Set the runtime control mechanisms for pausing and scheduling callables."""
    global _pause_event, _call_queue
    _pause_event = pause_event
    _call_queue = call_queue


def _to_queue(fn):
    """
    Decorator that wraps primitive constructors to enqueue their call_literals
    and block until acknowledgment is received.
    """

    def wrapper(*args, **kwargs):
        # Assert that queues are initialized
        if _text_q is None or _ack_q is None:
            raise RuntimeError("Queues not initialised; call set_queues() first.")

        # Create the primitive first
        primitive = fn(*args, **kwargs)

        # If paused, continuously drain the call queue until unpaused
        if _pause_event is not None and _call_queue is not None:
            import time

            while not _pause_event.is_set():
                drained = False
                while True:
                    try:
                        _call_queue.get_nowait()()
                        drained = True
                    except queue.Empty:
                        break
                if not drained:
                    time.sleep(0.01)

        # Enqueue the command (only after pause is lifted)
        _text_q.put(primitive.call_literal)

        # Wait for acknowledgment in a loop
        while True:
            ack = _ack_q.get()  # block until a message arrives
            if ack == primitive.call_literal:
                break  # our acknowledgment arrived
            _ack_q.put(ack)  # not ours, put it back

        return primitive

    return wrapper


def _raw_open_browser() -> Primitive:
    return Primitive("open_browser", {}, CMD_OPEN_BROWSER)


open_browser = _to_queue(_raw_open_browser)


def _raw_close_browser() -> Primitive:
    return Primitive("close_browser", {}, CMD_CLOSE_BROWSER)


close_browser = _to_queue(_raw_close_browser)


def _raw_click_out() -> Primitive:
    return Primitive("click_out", {}, CMD_CLICK_OUT)


click_out = _to_queue(_raw_click_out)


def _raw_continue_scrolling() -> Primitive:
    return Primitive("continue_scrolling", {}, CMD_CONT_SCROLLING)


continue_scrolling = _to_queue(_raw_continue_scrolling)


def _raw_cursor_down() -> Primitive:
    return Primitive("cursor_down", {}, CMD_CURSOR_DOWN)


cursor_down = _to_queue(_raw_cursor_down)


def _raw_cursor_left() -> Primitive:
    return Primitive("cursor_left", {}, CMD_CURSOR_LEFT)


cursor_left = _to_queue(_raw_cursor_left)


def _raw_cursor_right() -> Primitive:
    return Primitive("cursor_right", {}, CMD_CURSOR_RIGHT)


cursor_right = _to_queue(_raw_cursor_right)


def _raw_cursor_up() -> Primitive:
    return Primitive("cursor_up", {}, CMD_CURSOR_UP)


cursor_up = _to_queue(_raw_cursor_up)


def _raw_enter_text(text: str) -> Primitive:
    command = CMD_ENTER_TEXT.replace("*", text)
    return Primitive("enter_text", {"text": text}, command)


enter_text = _to_queue(_raw_enter_text)


def _raw_hold_shift() -> Primitive:
    return Primitive("hold_shift", {}, CMD_HOLD_SHIFT)


hold_shift = _to_queue(_raw_hold_shift)


def _raw_move_line_end() -> Primitive:
    return Primitive("move_line_end", {}, CMD_MOVE_LINE_END)


move_line_end = _to_queue(_raw_move_line_end)


def _raw_move_line_start() -> Primitive:
    return Primitive("move_line_start", {}, CMD_MOVE_LINE_START)


move_line_start = _to_queue(_raw_move_line_start)


def _raw_move_word_left() -> Primitive:
    return Primitive("move_word_left", {}, CMD_MOVE_WORD_LEFT)


move_word_left = _to_queue(_raw_move_word_left)


def _raw_move_word_right() -> Primitive:
    return Primitive("move_word_right", {}, CMD_MOVE_WORD_RIGHT)


move_word_right = _to_queue(_raw_move_word_right)


def _raw_new_tab() -> Primitive:
    return Primitive("new_tab", {}, CMD_NEW_TAB)


new_tab = _to_queue(_raw_new_tab)


def _raw_close_this_tab() -> Primitive:
    return Primitive("close_this_tab", {}, CMD_CLOSE_THIS_TAB)


close_this_tab = _to_queue(_raw_close_this_tab)


def _raw_press_backspace() -> Primitive:
    return Primitive("press_backspace", {}, CMD_PRESS_BACKSPACE)


press_backspace = _to_queue(_raw_press_backspace)


def _raw_press_delete() -> Primitive:
    return Primitive("press_delete", {}, CMD_PRESS_DELETE)


press_delete = _to_queue(_raw_press_delete)


def _raw_press_enter() -> Primitive:
    return Primitive("press_enter", {}, CMD_PRESS_ENTER)


press_enter = _to_queue(_raw_press_enter)


def _raw_release_shift() -> Primitive:
    return Primitive("release_shift", {}, CMD_RELEASE_SHIFT)


release_shift = _to_queue(_raw_release_shift)


def _raw_scroll_down(amount: str) -> Primitive:
    command = CMD_SCROLL_DOWN.replace("*", amount)
    return Primitive("scroll_down", {"amount": amount}, command)


scroll_down = _to_queue(_raw_scroll_down)


def _raw_scroll_up(amount: str) -> Primitive:
    command = CMD_SCROLL_UP.replace("*", amount)
    return Primitive("scroll_up", {"amount": amount}, command)


scroll_up = _to_queue(_raw_scroll_up)


def _raw_search(query: str) -> Primitive:
    command = CMD_SEARCH.replace("*", query)
    return Primitive("search", {"query": query}, command)


search = _to_queue(_raw_search)


def _raw_open_url(url: str) -> Primitive:
    command = CMD_OPEN_URL.replace("*", url)
    return Primitive("open_url", {"url": url}, command)


open_url = _to_queue(_raw_open_url)


def _raw_select_all() -> Primitive:
    return Primitive("select_all", {}, CMD_SELECT_ALL)


select_all = _to_queue(_raw_select_all)


def _raw_start_scrolling_down() -> Primitive:
    return Primitive("start_scrolling_down", {}, CMD_START_SCROLL_DOWN)


start_scrolling_down = _to_queue(_raw_start_scrolling_down)


def _raw_start_scrolling_up() -> Primitive:
    return Primitive("start_scrolling_up", {}, CMD_START_SCROLL_UP)


start_scrolling_up = _to_queue(_raw_start_scrolling_up)


def _raw_stop_scrolling() -> Primitive:
    return Primitive("stop_scrolling", {}, CMD_STOP_SCROLLING)


stop_scrolling = _to_queue(_raw_stop_scrolling)


def _raw_click_button(target: str) -> Primitive:
    command = CMD_CLICK_BUTTON.replace("*", target)
    return Primitive("click_button", {"target": target}, command)


click_button = _to_queue(_raw_click_button)


def _raw_select_tab(tab: str) -> Primitive:
    command = CMD_SELECT_TAB.replace("*", tab)
    return Primitive("select_tab", {"tab": tab}, command)


select_tab = _to_queue(_raw_select_tab)


def _raw_close_tab(tab: str) -> Primitive:
    command = CMD_CLOSE_TAB.replace("*", tab)
    return Primitive("close_tab", {"tab": tab}, command)


close_tab = _to_queue(_raw_close_tab)


def _raw_go_back() -> Primitive:
    return Primitive("go_back", {}, CMD_GO_BACK)


go_back = _to_queue(_raw_go_back)


def _raw_go_forward() -> Primitive:
    return Primitive("go_forward", {}, CMD_GO_FORWARD)


go_forward = _to_queue(_raw_go_forward)


def _raw_duplicate_tab() -> Primitive:
    return Primitive("duplicate_tab", {}, CMD_DUPLICATE_TAB)


duplicate_tab = _to_queue(_raw_duplicate_tab)


def _raw_wait_for_user_signal(timeout: float = None) -> Primitive:
    """
    Creates a primitive that waits for the user to signal continuation.

    This function blocks until the pause event is set, indicating that the user
    has signaled to continue execution.

    Args:
        timeout: Optional timeout in seconds. If None, waits indefinitely.

    Returns:
        A Primitive representing the wait operation.
    """
    # Check if pause event is initialized
    if _pause_event is None:
        raise RuntimeError(
            "Pause event not initialized; call set_runtime_controls() first."
        )

    # Wait for the pause event to be set (i.e., wait until unpaused)
    if not _pause_event.wait(timeout):
        # If we get here with a timeout, the event wasn't set within the timeout period
        pass  # We still return the primitive even if we timed out

    # Return a primitive with no actual controller command
    return Primitive("wait_for_user_signal", {"timeout": timeout}, "")


wait_for_user_signal = _to_queue(_raw_wait_for_user_signal)

_PRIMITIVE_PUBLIC_ALIASES = {
    "click_on": click_button,
    "select_tab": select_tab,
    "go_to_url": open_url,
    "wait_for_user_signal": wait_for_user_signal,
}

# Update globals with aliases
globals().update(_PRIMITIVE_PUBLIC_ALIASES)

# Define what's publicly available from this module
__all__ = [
    "open_browser",
    "close_browser",
    "click_out",
    "continue_scrolling",
    "cursor_down",
    "cursor_left",
    "cursor_right",
    "cursor_up",
    "enter_text",
    "hold_shift",
    "move_line_end",
    "move_line_start",
    "move_word_left",
    "move_word_right",
    "new_tab",
    "close_this_tab",
    "press_backspace",
    "press_delete",
    "press_enter",
    "release_shift",
    "scroll_down",
    "scroll_up",
    "search",
    "open_url",
    "select_all",
    "start_scrolling_down",
    "start_scrolling_up",
    "stop_scrolling",
    "click_button",
    "select_tab",
    "close_tab",
    "go_back",
    "go_forward",
    "duplicate_tab",
    "wait_for_user_signal",
    # Aliases
    "click_on",
    "go_to_url",
]
