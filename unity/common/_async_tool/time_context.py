from __future__ import annotations

import json
import math
import time as _time
from dataclasses import dataclass
from typing import Union

# --------------------------------------------------------------------------- #
#  MONOTONIC TIME HELPER (monkey-patchable for tests)                         #
# --------------------------------------------------------------------------- #


def perf_counter() -> float:
    """Return a monotonic time value for measuring elapsed durations.

    This wraps time.perf_counter() to enable monkey-patching in tests.

    Returns
    -------
    float
        A monotonic time value in seconds (relative to an arbitrary origin).
    """
    return _time.perf_counter()


# --------------------------------------------------------------------------- #
#  FORMATTING HELPERS                                                         #
# --------------------------------------------------------------------------- #


def format_offset(seconds: float) -> str:
    """Format *seconds* as a compact signed offset string.

    Examples: ``+0s``, ``+45s``, ``+1m30s``, ``+1h2m30s``.
    """
    if seconds < 0:
        return "+0s"

    total = int(math.floor(seconds))
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)

    parts: list[str] = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s or not parts:
        parts.append(f"{s}s")
    return "+" + "".join(parts)


def format_duration(seconds: float) -> str:
    """Format *seconds* as a compact human-readable duration.

    Sub-second precision uses milliseconds.
    Examples: ``0s``, ``100ms``, ``2s45ms``, ``1m30s``, ``1h2m30s``.
    """
    if seconds < 0:
        return "0s"

    total_ms = int(round(seconds * 1000))
    if total_ms == 0:
        return "0s"

    h, remainder_ms = divmod(total_ms, 3_600_000)
    m, remainder_ms = divmod(remainder_ms, 60_000)
    s, ms = divmod(remainder_ms, 1000)

    parts: list[str] = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s:
        parts.append(f"{s}s")
    if ms and not h:
        parts.append(f"{ms}ms")
    if not parts:
        parts.append("0s")
    return "".join(parts)


# --------------------------------------------------------------------------- #
#  TIME CONTEXT                                                               #
# --------------------------------------------------------------------------- #

_EXPLANATION_PROMPT = (
    "## Time Annotations\n"
    "This conversation includes inline timing metadata so you can reason "
    "about elapsed time and tool execution order.\n\n"
    "- **Tool results** from non-steering tools include a JSON envelope:\n"
    '  `{"tool_result": <result>, "metadata": {"called_at": "+1m30s", "duration": "2s45ms"}}`\n'
    "  `called_at` is the offset since conversation start when the tool was invoked; "
    "`duration` is wall-clock execution time.\n"
    "- **User messages** are prefixed with `[elapsed: +XmYs]` showing "
    "when the message was sent relative to conversation start.\n"
    '- **Pending tool placeholders** include `"meta:started"` with the '
    "invocation offset.\n\n"
    "Use these annotations when reasoning about timing, ordering, or how "
    "long operations took. Do NOT reproduce the annotations in your replies."
)


@dataclass
class TimeContext:
    """Tracks wall-clock offsets for an async tool loop.

    Provides compact offset/duration formatting and result-wrapping
    helpers consumed by the tool loop infrastructure.

    Attributes
    ----------
    perf_counter_start : float
        Monotonic ``perf_counter()`` value captured at loop start.
    """

    perf_counter_start: float

    # -- offset helpers -------------------------------------------------------

    def current_offset(self) -> str:
        """Return the current elapsed offset since loop start."""
        return format_offset(perf_counter() - self.perf_counter_start)

    def offset_at(self, perf_time: float) -> str:
        """Return the offset string for a given ``perf_counter`` snapshot."""
        return format_offset(perf_time - self.perf_counter_start)

    def duration_since(self, perf_time: float) -> str:
        """Return the duration string from *perf_time* until now."""
        return format_duration(perf_counter() - perf_time)

    # -- result wrapping ------------------------------------------------------

    def wrap_result(
        self,
        content: Union[str, list],
        scheduled_time: float,
    ) -> Union[str, list]:
        """Wrap a serialized tool result with timing metadata.

        Parameters
        ----------
        content
            The value returned by ``serialize_tool_content`` — either a
            JSON string or a list of content blocks (when images are present).
        scheduled_time
            The ``perf_counter`` value when the tool was scheduled.

        Returns
        -------
        str | list
            * **str content** -- a JSON string of
              ``{"tool_result": <content>, "metadata": {"called_at": …, "duration": …}}``.
              ``tool_result`` holds the original *content* verbatim.
            * **list content** (image blocks) -- the original list with a
              metadata text block prepended.
        """
        called_at = self.offset_at(scheduled_time)
        duration = self.duration_since(scheduled_time)
        meta = {"called_at": called_at, "duration": duration}

        if isinstance(content, list):
            meta_block = {
                "type": "text",
                "text": json.dumps({"metadata": meta}),
            }
            return [meta_block, *content]

        envelope = {"tool_result": content, "metadata": meta}
        return json.dumps(envelope, indent=4)

    # -- user message annotation ----------------------------------------------

    def prefix_user_message(self, text: str) -> str:
        """Prepend the elapsed offset to a user message string."""
        return f"[elapsed: {self.current_offset()}] {text}"

    # -- system prompt --------------------------------------------------------

    @staticmethod
    def build_explanation_prompt() -> str:
        """Return the static system-message content explaining inline time annotations."""
        return _EXPLANATION_PROMPT


def create_time_context() -> TimeContext:
    """Create a new ``TimeContext`` anchored at the current instant."""
    return TimeContext(perf_counter_start=perf_counter())
