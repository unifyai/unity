from __future__ import annotations

from typing import TYPE_CHECKING

from ..common.async_tool_loop import AsyncToolLoopHandle


if TYPE_CHECKING:  # type hints only
    pass


class ConductorRequestHandle(AsyncToolLoopHandle):
    """
    Custom handle for `Conductor.request` sessions.

    Extends the default async tool loop handle with convenience helpers that
    target common nested steering scenarios for Conductor-driven workflows.
    """
