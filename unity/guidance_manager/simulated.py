from __future__ import annotations

import functools
from typing import Any, Optional, TYPE_CHECKING

from .base import BaseGuidanceManager
from ..common.simulated import (
    maybe_tool_log_scheduled,
    maybe_tool_log_completed,
)


class SimulatedGuidanceManager(BaseGuidanceManager):
    """Drop-in replacement for GuidanceManager with no backing store.

    Only the ``clear`` method is needed; ``ask`` and ``update`` tool-loop
    methods have been removed from the GuidanceManager contract.
    """

    def __init__(
        self,
        description: str = "nothing fixed, make up some imaginary scenario",
        *,
        log_events: bool = False,
        rolling_summary_in_prompts: bool = True,
        simulation_guidance: Optional[str] = None,
        hold_completion: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._description = description
        self._log_events = log_events
        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        self._simulation_guidance = simulation_guidance
        self._hold_completion = hold_completion

    @functools.wraps(BaseGuidanceManager.clear, updated=())
    def clear(self) -> None:
        sched = maybe_tool_log_scheduled(
            "SimulatedGuidanceManager.clear",
            "clear",
            {},
        )
        type(self).__init__(
            self,
            description=getattr(
                self,
                "_description",
                "nothing fixed, make up some imaginary scenario",
            ),
            log_events=getattr(self, "_log_events", False),
            rolling_summary_in_prompts=getattr(
                self,
                "_rolling_summary_in_prompts",
                True,
            ),
            simulation_guidance=getattr(self, "_simulation_guidance", None),
            hold_completion=getattr(self, "_hold_completion", False),
        )
        if sched:
            label, cid, t0 = sched
            maybe_tool_log_completed(label, cid, "clear", {"outcome": "reset"}, t0)


if TYPE_CHECKING:
    from ..common.tool_outcome import ToolOutcome  # noqa: F401
