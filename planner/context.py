"""
Context module for the planner.

This module provides a singleton PlannerContext class that maintains shared state
across different planner components, including browser state snapshots and call stack information.
"""

import queue
import threading
from typing import List, Optional, Any, Dict

try:
    from planner.verifier_utils import _hash_dom
except ImportError:
    _hash_dom = None


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
        self._exploration_depth = 0
        self._exploration_tab_id = None
        self._main_tab_id = None

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

    def summarise_snapshot(self, snap: dict, max_elems: int = 50) -> dict:
        """
        Create a summarized version of a snapshot suitable for LLM prompting.

        This method:
        1. Removes binary fields like 'screenshot'
        2. Truncates 'elements' list to max_elems
        3. Computes dom_hash if missing but dom is present

        Args:
            snap: The snapshot dictionary to summarize
            max_elems: Maximum number of elements to include (default: 50)

        Returns:
            A summarized copy of the snapshot
        """
        if snap is None:
            return None

        # Create a shallow copy to avoid modifying the original
        summary = {k: v for k, v in snap.items() if k != "screenshot"}

        # Truncate elements list if present
        if "elements" in summary and isinstance(summary["elements"], list):
            elements = summary["elements"]
            if len(elements) > max_elems:
                summary["elements"] = elements[:max_elems]
                summary["elements"].append({"_truncated": len(elements) - max_elems})

        # Compute dom_hash if missing but dom is present
        if "dom" in summary and "dom_hash" not in summary and _hash_dom is not None:
            try:
                summary["dom_hash"] = _hash_dom(summary["dom"])
                # Legacy alias for existing tests
                summary["dom_sha"] = summary["dom_hash"]
            except Exception:
                # If hashing fails, continue without the hash
                pass

        return summary

    def last_state_snapshot(self) -> Optional[Any]:
        """
        Get the latest browser state from a non-blocking read of the broadcast queue.
        Returns the canonicalised state dict (or None).

        This is a convenience method that drains the queue to get the latest state.
        For waiting on a new snapshot, use get_snapshot() with a wait parameter.
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

    def get_snapshot(self, wait: float = 0.0) -> Optional[dict]:
        """
        Get the latest browser state snapshot.

        Args:
            wait: Time in seconds to wait for a snapshot. If 0, returns immediately.

        Returns:
            The canonicalized state dict, or None if no snapshot is available
        """
        if self._broadcast_queue is None:
            return None

        try:
            if wait > 0:
                snapshot = self._broadcast_queue.get(timeout=wait)
            else:
                snapshot = self._broadcast_queue.get_nowait()
            return self._normalize_snapshot(snapshot)
        except queue.Empty:
            return None

    def enter_exploration(self, tab_id, main_tab=None) -> None:
        """
        Enter an exploration context with the given tab ID.

        Args:
            tab_id: The ID of the tab being used for exploration
            main_tab: The ID of the main tab to return to after exploration
        """
        with self._context_lock:
            self._exploration_depth += 1
            self._exploration_tab_id = tab_id
            # Only store the main tab when first entering exploration
            if self._exploration_depth == 1 and main_tab is not None:
                self._main_tab_id = main_tab

    def exit_exploration(self) -> Optional[Any]:
        """
        Exit the current exploration context.
        Decrements the exploration depth and clears the tab ID if depth reaches 0.

        Returns:
            The ID of the main tab when exiting the outermost exploration, or None otherwise
        """
        with self._context_lock:
            main_tab = None
            if self._exploration_depth > 0:
                self._exploration_depth -= 1
            if self._exploration_depth == 0:
                main_tab = self._main_tab_id
                self._exploration_tab_id = None
                self._main_tab_id = None
            return main_tab

    def in_exploration(self) -> bool:
        """
        Check if currently in an exploration context.

        Returns:
            True if in an exploration context, False otherwise
        """
        with self._context_lock:
            return self._exploration_depth > 0

    def get_exploration_tab(self) -> Optional[Any]:
        """
        Get the ID of the current exploration tab.

        Returns:
            The ID of the current exploration tab, or None if not in exploration
        """
        with self._context_lock:
            return self._exploration_tab_id

    def get_main_tab(self) -> Optional[Any]:
        """
        Get the ID of the main tab to return to after exploration.

        Returns:
            The ID of the main tab, or None if not in exploration or no main tab was set
        """
        with self._context_lock:
            return self._main_tab_id


context = PlannerContext()

get_snapshot = context.get_snapshot
push_frame = context.push_frame
pop_frame = context.pop_frame
get_call_stack = context.get_call_stack
