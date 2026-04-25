from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseGuidanceManager
from .types.guidance import Guidance
from ..image_manager.types import AnnotatedImageRefs
from ..common.tool_outcome import details_with_near_duplicates
from ..common.simulated import (
    maybe_tool_log_scheduled,
    maybe_tool_log_completed,
)


class SimulatedGuidanceManager(BaseGuidanceManager):
    """Drop-in replacement for GuidanceManager with an in-memory store."""

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
        self._entries: Dict[int, Guidance] = {}
        self._next_id: int = 1

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #

    @functools.wraps(BaseGuidanceManager.search, updated=())
    def search(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[Guidance]:
        return list(self._entries.values())[:k]

    @functools.wraps(BaseGuidanceManager.filter, updated=())
    def filter(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Guidance]:
        rows = list(self._entries.values())
        if filter is not None:
            matched = []
            for g in rows:
                try:
                    if eval(filter, {"__builtins__": {}}, g.model_dump()):
                        matched.append(g)
                except Exception:
                    pass
            rows = matched
        return rows[offset : offset + limit]

    @functools.wraps(BaseGuidanceManager.add_guidance, updated=())
    def add_guidance(
        self,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        images: Optional[AnnotatedImageRefs] = None,
        function_ids: Optional[List[int]] = None,
        _near_duplicates: Optional[List[Dict[str, Any]]] = None,
    ) -> "ToolOutcome":
        if not title and not content and not images:
            raise ValueError(
                "At least one field (title/content/images) must be provided.",
            )
        gid = self._next_id
        self._next_id += 1
        self._entries[gid] = Guidance(
            guidance_id=gid,
            title=title or "",
            content=content or "",
            images=images or AnnotatedImageRefs.model_validate([]),
            function_ids=function_ids or [],
        )
        return {
            "outcome": "guidance created successfully",
            "details": details_with_near_duplicates(
                {"guidance_id": gid},
                _near_duplicates,
            ),
        }

    @functools.wraps(BaseGuidanceManager.update_guidance, updated=())
    def update_guidance(
        self,
        *,
        guidance_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        images: Optional[AnnotatedImageRefs] = None,
        function_ids: Optional[List[int]] = None,
        _near_duplicates: Optional[List[Dict[str, Any]]] = None,
    ) -> "ToolOutcome":
        existing = self._entries.get(guidance_id)
        if existing is None:
            raise ValueError(
                f"No guidance found with guidance_id {guidance_id} to update.",
            )
        updates: Dict[str, Any] = {}
        if title is not None:
            updates["title"] = title
        if content is not None:
            updates["content"] = content
        if images is not None:
            updates["images"] = images
        if function_ids is not None:
            updates["function_ids"] = function_ids
        if not updates:
            raise ValueError("At least one field must be provided for an update.")
        self._entries[guidance_id] = existing.model_copy(update=updates)
        return {
            "outcome": "guidance updated",
            "details": details_with_near_duplicates(
                {"guidance_id": guidance_id},
                _near_duplicates,
            ),
        }

    @functools.wraps(BaseGuidanceManager.delete_guidance, updated=())
    def delete_guidance(
        self,
        *,
        guidance_id: int,
    ) -> "ToolOutcome":
        if guidance_id not in self._entries:
            raise ValueError(
                f"No guidance found with guidance_id {guidance_id} to delete.",
            )
        del self._entries[guidance_id]
        return {"outcome": "guidance deleted", "details": {"guidance_id": guidance_id}}

    # ------------------------------------------------------------------ #
    # clear
    # ------------------------------------------------------------------ #

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
