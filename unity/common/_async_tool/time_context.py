"""
unity/common/_async_tool/time_context.py
=========================================

Time-awareness context for async tool loops.

Tracks conversation start time and tool execution history, making this
information available to the LLM via system messages. All time operations
use helpers that can be monkey-patched in tests:
- `now()` from `prompt_helpers` for datetime (respects assistant timezone)
- `perf_counter()` for monotonic timing (tool start offsets, durations)
"""

import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from ..prompt_helpers import now

# --------------------------------------------------------------------------- #
#  MONOTONIC TIME HELPER (monkey-patchable for tests)                         #
# --------------------------------------------------------------------------- #


def perf_counter() -> float:
    """Return a monotonic time value for measuring elapsed durations.

    This wraps time.perf_counter() to enable monkey-patching in tests,
    making tool timing ("Started (relative)", "Duration") deterministic.

    Returns
    -------
    float
        A monotonic time value in seconds (relative to an arbitrary origin).
    """
    return _time.perf_counter()


@dataclass(frozen=True)
class ToolTiming:
    """Record of a single tool execution."""

    call_id: str
    name: str
    started_offset_secs: float
    duration_secs: float


@dataclass
class TimeContext:
    """Tracks time-related context for an async tool loop.

    Captures the conversation start time and maintains a history of tool
    executions with their timing information. This context is injected
    into system messages so the LLM can reason about time.

    Attributes
    ----------
    loop_start_time : datetime
        The datetime when the loop started (via now(as_string=False)).
    perf_counter_start : float
        The time.perf_counter() value at loop start, used to compute
        tool start offsets from ToolCallMetadata.scheduled_time.
    tool_history : List[ToolTiming]
        Chronological list of completed tool executions.
    """

    loop_start_time: datetime
    perf_counter_start: float
    tool_history: List[ToolTiming] = field(default_factory=list)

    def elapsed_since_start(self) -> float:
        """Return seconds elapsed since conversation started."""
        current = now(as_string=False)
        return (current - self.loop_start_time).total_seconds()

    def compute_start_offset(self, scheduled_perf_counter: float) -> float:
        """Compute the start offset relative to loop start.

        Parameters
        ----------
        scheduled_perf_counter : float
            The time.perf_counter() value when the tool was scheduled
            (from ToolCallMetadata.scheduled_time).

        Returns
        -------
        float
            Seconds after loop_start_time when the tool was scheduled.
        """
        return scheduled_perf_counter - self.perf_counter_start

    def add_tool_timing(
        self,
        call_id: str,
        name: str,
        start_offset: float,
        duration: float,
    ) -> None:
        """Record a completed tool's timing information.

        Parameters
        ----------
        call_id : str
            Unique identifier for this tool invocation.
        name : str
            Tool name (may be shared by multiple calls).
        start_offset : float
            Seconds after loop_start_time when the tool was scheduled.
        duration : float
            How long the tool took to execute in seconds.
        """
        self.tool_history.append(
            ToolTiming(
                call_id=call_id,
                name=name,
                started_offset_secs=start_offset,
                duration_secs=duration,
            ),
        )

    def _format_elapsed(self, seconds: float) -> str:
        """Format elapsed seconds as human-readable duration."""
        if seconds < 0:
            return "0s"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            # Show decimal for sub-minute durations
            if hours == 0 and minutes == 0:
                parts.append(f"{secs:.1f}s")
            else:
                parts.append(f"{int(secs)}s")

        return " ".join(parts)

    def build_system_message(self) -> str:
        """Build the time context section for injection into system messages.

        Note: Current time is already injected via compose_system_prompt's
        time footer (uses now()), so we don't duplicate it here. This method
        focuses on:
        - Conversation start (relative to current time)
        - Tool execution history with call_id for unique identification

        Returns
        -------
        str
            Formatted time context section for the system message.
        """
        elapsed = self.elapsed_since_start()
        elapsed_str = self._format_elapsed(elapsed)

        lines = [
            "## Time Context",
            "Below is your temporal awareness for this conversation: when it started, "
            "how long it's been running, and a log of every tool call with timing. "
            "Consult it when reasoning about elapsed time or the order of events.",
            f"- Conversation started: {elapsed_str} ago",
        ]

        if self.tool_history:
            lines.append("")
            lines.append("### Tool Execution History")
            lines.append("| call_id | tool | started (relative) | duration |")
            lines.append("| --- | --- | --- | --- |")

            for tool in self.tool_history:
                started = f"+{tool.started_offset_secs:.1f}s"
                duration = f"{tool.duration_secs:.2f}s"
                lines.append(
                    f"| {tool.call_id} | {tool.name[:32]} | {started} | {duration} |",
                )

        return "\n".join(lines)

    def update_system_message(self, msg: dict) -> None:
        """Update a system message dict with refreshed time context.

        Replaces the content of the dedicated time context system message
        with the latest build (including any new tool timings).

        Parameters
        ----------
        msg : dict
            The time context system message dict (must have "_time_context" marker).
        """
        if not msg or not msg.get("_time_context"):
            return

        msg["content"] = self.build_system_message()


def create_time_context() -> TimeContext:
    """Create a new TimeContext with the current time as start.

    Uses now(as_string=False) to get a datetime object that respects
    the assistant's timezone and is monkey-patchable in tests.

    Also captures perf_counter() at the same moment to enable
    computing tool start offsets from ToolCallMetadata.scheduled_time.
    Both helpers are monkey-patchable for deterministic tests.

    Returns
    -------
    TimeContext
        A new time context initialized with the current time.
    """
    return TimeContext(
        loop_start_time=now(as_string=False),
        perf_counter_start=perf_counter(),
    )
