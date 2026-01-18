"""
Input Listener for Guided Learning.

Captures OS-level mouse and keyboard events using pynput.
This provides accurate interaction capture (clicks, typing, drags)
that cannot be detected from pixel changes alone.

Cross-Platform Compatibility:
    - pynput works on Windows, Linux, and macOS
    - Uses time.time() for timestamps (works identically on all platforms)
    - Key modifiers (ctrl, shift, alt) are normalized by pynput

Platform-Specific Permissions:
    macOS:
        System Settings > Privacy & Security > Accessibility → Enable app
        System Settings > Privacy & Security > Input Monitoring → Enable app

    Windows:
        Run as Administrator for full keyboard/mouse capture
        Some elevated apps (UAC dialogs) may block input monitoring

    Linux:
        User needs access to /dev/input/* devices
        Add user to 'input' group: sudo usermod -aG input $USER
        X11: May need xhost +local: for some setups
        Wayland: Input capture is limited by design

Usage:
    listener = InputEventListener(
        on_event=lambda event: print(event),
        settings=InputListenerSettings(),
    )
    listener.start()
    # ... do stuff ...
    listener.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class InputEventType(Enum):
    """Type of user input event."""

    CLICK = auto()  # Mouse click (left, right, middle)
    DOUBLE_CLICK = auto()  # Double click
    DRAG_START = auto()  # Mouse button down + move
    DRAG_END = auto()  # Mouse button up after drag
    SCROLL = auto()  # Mouse scroll
    KEY_PRESS = auto()  # Single key press
    KEY_COMBO = auto()  # Key combination (e.g., Cmd+C)
    TYPING = auto()  # Accumulated typing (batched characters)


@dataclass
class Point:
    """Screen coordinates."""

    x: int
    y: int


@dataclass
class InputEvent:
    """
    A captured input event from pynput.

    Contains the event type, timestamp, position (for mouse events),
    and additional metadata like keys pressed or text typed.
    """

    event_type: InputEventType
    timestamp: float
    position: Optional[Point] = None

    # For mouse events
    button: Optional[str] = None  # "left", "right", "middle"

    # For keyboard events
    key: Optional[str] = None  # Single key
    modifiers: List[str] = field(default_factory=list)  # ["cmd", "shift", etc.]
    text: Optional[str] = None  # Accumulated text for TYPING events

    # For scroll events
    scroll_dx: int = 0
    scroll_dy: int = 0

    # For drag events
    drag_start: Optional[Point] = None
    drag_end: Optional[Point] = None


class InputListenerSettings(BaseModel):
    """Settings for the input event listener."""

    # Timing settings
    double_click_threshold_ms: float = Field(
        default=300.0,
        description="Max time between clicks to count as double-click (ms).",
    )
    typing_batch_interval_ms: float = Field(
        default=500.0,
        description="Batch keystrokes within this window into single TYPING event.",
    )
    drag_threshold_px: int = Field(
        default=5,
        description="Minimum movement to count as drag vs click.",
    )

    # Event filtering
    capture_mouse: bool = Field(
        default=True,
        description="Capture mouse events (clicks, drags, scrolls).",
    )
    capture_keyboard: bool = Field(
        default=True,
        description="Capture keyboard events.",
    )
    capture_scroll: bool = Field(
        default=False,
        description="Capture scroll events (can be noisy).",
    )

    # Batching
    batch_scroll_events: bool = Field(
        default=True,
        description="Batch rapid scroll events into single event.",
    )


class InputEventListener:
    """
    Listens for OS-level mouse and keyboard events using pynput.

    Provides accurate interaction capture that cannot be detected
    from pixel changes alone (clicks, typing, drags).

    Note: Requires accessibility permissions on macOS.
    """

    def __init__(
        self,
        on_event: Callable[[InputEvent], None],
        settings: Optional[InputListenerSettings] = None,
    ):
        """
        Initialize the input listener.

        Args:
            on_event: Callback function called for each input event.
            settings: Configuration settings.
        """
        self.on_event = on_event
        self.settings = settings or InputListenerSettings()

        self._mouse_listener = None
        self._keyboard_listener = None
        self._running = False

        # State for event detection
        self._last_click_time: float = 0.0
        self._last_click_pos: Optional[Point] = None
        self._mouse_pressed: bool = False
        self._mouse_press_pos: Optional[Point] = None
        self._mouse_press_time: float = 0.0
        self._current_modifiers: set = set()

        # Typing accumulation
        self._typing_buffer: str = ""
        self._typing_start_time: float = 0.0
        self._typing_timer: Optional[threading.Timer] = None
        self._typing_lock = threading.Lock()

        # Track if we've moved during a press (for drag detection)
        self._moved_during_press: bool = False

    def start(self) -> None:
        """Start listening for input events."""
        if self._running:
            logger.warning("InputEventListener already running")
            return

        try:
            from pynput import mouse, keyboard
        except ImportError:
            logger.error(
                "pynput not installed. Install with: pip install pynput\n"
                "Also ensure accessibility permissions are granted.",
            )
            raise

        self._running = True

        if self.settings.capture_mouse:
            self._mouse_listener = mouse.Listener(
                on_click=self._on_mouse_click,
                on_move=self._on_mouse_move,
                on_scroll=(
                    self._on_mouse_scroll if self.settings.capture_scroll else None
                ),
            )
            self._mouse_listener.start()
            logger.info("🖱️ Mouse listener started")

        if self.settings.capture_keyboard:
            self._keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
            )
            self._keyboard_listener.start()
            logger.info("⌨️ Keyboard listener started")

    def stop(self) -> None:
        """Stop listening for input events."""
        self._running = False

        # Flush any pending typing
        self._flush_typing_buffer()

        if self._mouse_listener:
            self._mouse_listener.stop()
            self._mouse_listener = None

        if self._keyboard_listener:
            self._keyboard_listener.stop()
            self._keyboard_listener = None

        logger.info("Input listeners stopped")

    def _emit(self, event: InputEvent) -> None:
        """Emit an event to the callback."""
        if self._running:
            try:
                self.on_event(event)
            except Exception as e:
                logger.error(f"Error in input event callback: {e}")

    # ─── Mouse Event Handlers ───────────────────────────────────────────

    def _on_mouse_click(self, x: int, y: int, button, pressed: bool) -> None:
        """Handle mouse click events."""
        now = time.time()
        pos = Point(x=int(x), y=int(y))
        button_name = str(button).split(".")[-1]  # e.g., "Button.left" -> "left"

        if pressed:
            self._mouse_pressed = True
            self._mouse_press_pos = pos
            self._mouse_press_time = now
            self._moved_during_press = False
        else:
            # Mouse released
            self._mouse_pressed = False

            if self._moved_during_press and self._mouse_press_pos:
                # This was a drag
                dx = abs(pos.x - self._mouse_press_pos.x)
                dy = abs(pos.y - self._mouse_press_pos.y)

                if (
                    dx > self.settings.drag_threshold_px
                    or dy > self.settings.drag_threshold_px
                ):
                    self._emit(
                        InputEvent(
                            event_type=InputEventType.DRAG_END,
                            timestamp=now,
                            position=pos,
                            button=button_name,
                            drag_start=self._mouse_press_pos,
                            drag_end=pos,
                        ),
                    )
                    return

            # Check for double-click
            time_since_last = (now - self._last_click_time) * 1000  # to ms

            if (
                time_since_last < self.settings.double_click_threshold_ms
                and self._last_click_pos
                and abs(pos.x - self._last_click_pos.x) < 10
                and abs(pos.y - self._last_click_pos.y) < 10
            ):
                self._emit(
                    InputEvent(
                        event_type=InputEventType.DOUBLE_CLICK,
                        timestamp=now,
                        position=pos,
                        button=button_name,
                    ),
                )
                # Reset to prevent triple-click counting as another double
                self._last_click_time = 0.0
            else:
                self._emit(
                    InputEvent(
                        event_type=InputEventType.CLICK,
                        timestamp=now,
                        position=pos,
                        button=button_name,
                    ),
                )
                self._last_click_time = now
                self._last_click_pos = pos

    def _on_mouse_move(self, x: int, y: int) -> None:
        """Handle mouse move events."""
        if self._mouse_pressed and self._mouse_press_pos:
            dx = abs(int(x) - self._mouse_press_pos.x)
            dy = abs(int(y) - self._mouse_press_pos.y)

            if (
                dx > self.settings.drag_threshold_px
                or dy > self.settings.drag_threshold_px
            ):
                if not self._moved_during_press:
                    # First time we've moved enough to count as drag
                    self._moved_during_press = True
                    self._emit(
                        InputEvent(
                            event_type=InputEventType.DRAG_START,
                            timestamp=time.time(),
                            position=self._mouse_press_pos,
                            button="left",  # Assuming left button for drags
                            drag_start=self._mouse_press_pos,
                        ),
                    )

    def _on_mouse_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        """Handle mouse scroll events."""
        self._emit(
            InputEvent(
                event_type=InputEventType.SCROLL,
                timestamp=time.time(),
                position=Point(x=int(x), y=int(y)),
                scroll_dx=dx,
                scroll_dy=dy,
            ),
        )

    # ─── Keyboard Event Handlers ────────────────────────────────────────

    def _on_key_press(self, key) -> None:
        """Handle key press events."""
        now = time.time()

        try:
            from pynput.keyboard import Key
        except ImportError:
            return

        # Track modifiers
        modifier_map = {
            Key.cmd: "cmd",
            Key.cmd_l: "cmd",
            Key.cmd_r: "cmd",
            Key.ctrl: "ctrl",
            Key.ctrl_l: "ctrl",
            Key.ctrl_r: "ctrl",
            Key.alt: "alt",
            Key.alt_l: "alt",
            Key.alt_r: "alt",
            Key.shift: "shift",
            Key.shift_l: "shift",
            Key.shift_r: "shift",
        }

        if key in modifier_map:
            self._current_modifiers.add(modifier_map[key])
            return

        # Get the key character
        key_char = None
        key_name = None

        try:
            key_char = key.char
        except AttributeError:
            key_name = str(key).split(".")[-1]  # e.g., "Key.enter" -> "enter"

        # Check if this is a key combo (modifier + key)
        if self._current_modifiers and (key_char or key_name):
            self._emit(
                InputEvent(
                    event_type=InputEventType.KEY_COMBO,
                    timestamp=now,
                    key=key_char or key_name,
                    modifiers=list(self._current_modifiers),
                ),
            )
            return

        # Regular key press - accumulate for typing
        if key_char:
            self._accumulate_typing(key_char)
        elif key_name in ("space",):
            self._accumulate_typing(" ")
        elif key_name in ("enter", "return"):
            self._accumulate_typing("\n")
        elif key_name == "backspace":
            # Handle backspace in buffer
            with self._typing_lock:
                if self._typing_buffer:
                    self._typing_buffer = self._typing_buffer[:-1]
        elif key_name == "tab":
            self._accumulate_typing("\t")
        else:
            # Special key (arrow, function key, etc.)
            self._flush_typing_buffer()
            self._emit(
                InputEvent(
                    event_type=InputEventType.KEY_PRESS,
                    timestamp=now,
                    key=key_name,
                ),
            )

    def _on_key_release(self, key) -> None:
        """Handle key release events."""
        try:
            from pynput.keyboard import Key
        except ImportError:
            return

        modifier_map = {
            Key.cmd: "cmd",
            Key.cmd_l: "cmd",
            Key.cmd_r: "cmd",
            Key.ctrl: "ctrl",
            Key.ctrl_l: "ctrl",
            Key.ctrl_r: "ctrl",
            Key.alt: "alt",
            Key.alt_l: "alt",
            Key.alt_r: "alt",
            Key.shift: "shift",
            Key.shift_l: "shift",
            Key.shift_r: "shift",
        }

        if key in modifier_map:
            self._current_modifiers.discard(modifier_map[key])

    def _accumulate_typing(self, char: str) -> None:
        """Accumulate typed characters and emit batched TYPING events."""
        with self._typing_lock:
            if not self._typing_buffer:
                self._typing_start_time = time.time()

            self._typing_buffer += char

            # Cancel existing timer
            if self._typing_timer:
                self._typing_timer.cancel()
                self._typing_timer = None

            # Flush IMMEDIATELY on newline (Enter pressed) - this is when commands execute!
            # This ensures we capture the typing BEFORE the command output appears
            if char == "\n":
                self._flush_typing_buffer_locked()
            else:
                # Set new timer to flush after interval
                self._typing_timer = threading.Timer(
                    self.settings.typing_batch_interval_ms / 1000.0,
                    self._flush_typing_buffer,
                )
                self._typing_timer.start()

    def _flush_typing_buffer_locked(self) -> None:
        """Flush typing buffer (must be called while holding _typing_lock)."""
        if self._typing_buffer:
            self._emit(
                InputEvent(
                    event_type=InputEventType.TYPING,
                    timestamp=self._typing_start_time,
                    text=self._typing_buffer,
                ),
            )
            self._typing_buffer = ""

        if self._typing_timer:
            self._typing_timer.cancel()
            self._typing_timer = None

    def _flush_typing_buffer(self) -> None:
        """Emit accumulated typing as a single event."""
        with self._typing_lock:
            if self._typing_buffer:
                self._emit(
                    InputEvent(
                        event_type=InputEventType.TYPING,
                        timestamp=self._typing_start_time,
                        text=self._typing_buffer,
                    ),
                )
                self._typing_buffer = ""

            if self._typing_timer:
                self._typing_timer.cancel()
                self._typing_timer = None
