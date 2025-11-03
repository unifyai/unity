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
        # Interject only if at least one pause actually applies to a child.
        spec: Dict[str, Any] = {
            "children": {
                "TaskScheduler.execute": {"steps": [{"method": "pause"}]},
                "Actor.act": {"steps": [{"method": "pause"}]},
            },
            "conditions": [
                {
                    "when": {
                        "any": [
                            {"selector": "TaskScheduler.execute", "status": "full"},
                            {"selector": "Actor.act", "status": "full"},
                        ],
                    },
                    "then": [{"method": "interject", "args": message}],
                },
            ],
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
        # Interject only if at least one resume actually applies to a child.
        spec: Dict[str, Any] = {
            "children": {
                "TaskScheduler.execute": {"steps": [{"method": "resume"}]},
                "Actor.act": {"steps": [{"method": "resume"}]},
            },
            "conditions": [
                {
                    "when": {
                        "any": [
                            {"selector": "TaskScheduler.execute", "status": "full"},
                            {"selector": "Actor.act", "status": "full"},
                        ],
                    },
                    "then": [{"method": "interject", "args": message}],
                },
            ],
        }
        return await self.nested_steer(spec)
