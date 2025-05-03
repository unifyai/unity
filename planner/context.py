"""
Context module for the planner.

This module provides a singleton PlannerContext class that maintains shared state
across different planner components, including browser state snapshots and call stack information.
"""

import queue
import threading
from typing import List, Optional, Any


class PlannerContext:
    """
    Singleton class that maintains shared state across different planner components.

    This class provides access to:
    - The latest browser state snapshot
    - The current function call stack
    - The current active plan module

    All planner modules should import and use this singleton to share context.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Ensure only one instance of PlannerContext exists."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PlannerContext, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """Initialize the singleton instance attributes."""
        self._broadcast_queue = None
        self._current_plan = None
        self._call_stack = []
        self._context_lock = threading.Lock()

    def set_broadcast_queue(self, broadcast_queue: "queue.Queue[Any]"):
        """
        Set the broadcast queue for browser state snapshots.

        Args:
            broadcast_queue: Queue containing browser state snapshots
        """
        self._broadcast_queue = broadcast_queue

    def _normalize_snapshot(self, snapshot: Any) -> Optional[dict]:
        """Convert incoming snapshot payloads to a canonical flat dict.

        We expect BrowserWorker to send a payload like
        {"state": {"url": ..., "title": ..., "scroll_y": ..., "in_textbox": ..., "active_tab": ...}, ...}
        but we defensively handle raw dicts too. Returns None if *snapshot* is None.
        """
        if snapshot is None:
            return None
        if not isinstance(snapshot, dict):
            return None
        # BrowserWorker packages vars(self.runner.state) under "state" key.
        state_part = snapshot.get("state", snapshot)
        if not isinstance(state_part, dict):
            return None
        canonical = {
            "url": state_part.get("url"),
            "title": state_part.get("title"),
            "scroll_y": state_part.get("scroll_y"),
            "in_textbox": state_part.get("in_textbox"),
            "active_tab": state_part.get("active_tab"),
            "dom_sha": state_part.get("dom_sha"),
            "focused_xpath": state_part.get("focused_xpath"),
        }
        return canonical

    def last_state_snapshot(self) -> Optional[Any]:
        """
        Get the latest browser state from a non-blocking read of the broadcast queue.
        Returns the canonicalised state dict (or None).
        """
        if self._broadcast_queue is None:
            return None
        latest_state = None
        try:
            while True:
                latest_state = self._broadcast_queue.get_nowait()
        except queue.Empty:
            pass
        return self._normalize_snapshot(latest_state)

    def get_call_stack(self) -> List[str]:
        """
        Get the names of active function frames in the call stack.

        Returns:
            List of function names in the call stack, from oldest to newest
        """
        with self._context_lock:
            return self._call_stack.copy()

    def push_frame(self, fn_name: str) -> None:
        """
        Record entry into a function call by adding it to the call stack.

        Args:
            fn_name: Name of the function being entered
        """
        with self._context_lock:
            self._call_stack.append(fn_name)

    def pop_frame(self) -> Optional[str]:
        """
        Record exit from a function call by removing it from the call stack.

        Returns:
            The name of the function that was exited, or None if the stack was empty
        """
        with self._context_lock:
            if not self._call_stack:
                return None
            return self._call_stack.pop()

    def get_current_plan(self) -> Optional[Any]:
        """
        Get the currently active plan module.

        Returns:
            The current plan module object, or None if no plan is active
        """
        with self._context_lock:
            return self._current_plan

    def set_current_plan(self, plan: Any) -> None:
        """
        Set the currently active plan module.

        Args:
            plan: The plan module object to set as current
        """
        with self._context_lock:
            self._current_plan = plan

    def get_snapshot(self) -> Optional[dict]:
        """
        Get the latest browser state snapshot.

        This is a thin wrapper over last_state_snapshot() that retrieves
        the latest browser state from the broadcast queue.

        Returns:
            The canonicalized state dict, or None if no snapshot is available
        """
        return self.last_state_snapshot()


context = PlannerContext()

get_snapshot = context.get_snapshot
push_frame = context.push_frame
pop_frame = context.pop_frame
get_call_stack = context.get_call_stack
