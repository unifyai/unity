from __future__ import annotations

from typing import Dict, Any

from ..common.async_tool_loop import AsyncToolLoopHandle


class ConductorRequestHandle(AsyncToolLoopHandle):
    """
    Custom handle for `Conductor.request` sessions.

    Extends the default async tool loop handle with convenience helpers that
    target common nested steering scenarios for Conductor-driven workflows.
    """

    async def pause_actor(self, reason: str) -> Dict[str, Any]:
        """
        Pause any in-flight Actor/TaskScheduler execution and announce the pause.

        Parameters
        ----------
        reason : str
            Human-readable reason to include in the interjection.

        Returns
        -------
        dict
            Summary of applied/skipped operations from nested steering.
        """

        message = f"<Actor has been paused due to {reason}>"
        spec: Dict[str, Any] = {
            "method": "interject",
            "args": message,
            "children": {
                "TaskScheduler.execute": {"method": "pause"},
                "Actor.act": {"method": "pause"},
            },
        }
        return await self.nested_steer(spec)

    async def resume_actor(self, reason: str) -> Dict[str, Any]:
        """
        Resume any in-flight Actor/TaskScheduler execution and announce the resume.

        Parameters
        ----------
        reason : str
            Human-readable reason to include in the interjection.

        Returns
        -------
        dict
            Summary of applied/skipped operations from nested steering.
        """

        message = f"<Actor has been resumed due to {reason}>"
        spec: Dict[str, Any] = {
            "method": "interject",
            "args": message,
            "children": {
                "TaskScheduler.execute": {"method": "resume"},
                "Actor.act": {"method": "resume"},
            },
        }
        return await self.nested_steer(spec)
