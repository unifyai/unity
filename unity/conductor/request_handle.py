from __future__ import annotations

from typing import Dict, Any, TYPE_CHECKING

from ..common.async_tool_loop import AsyncToolLoopHandle


if TYPE_CHECKING:  # type hints only
    from ..image_manager.types.image_refs import ImageRefs


class ConductorRequestHandle(AsyncToolLoopHandle):
    """
    Custom handle for `Conductor.request` sessions.

    Extends the default async tool loop handle with convenience helpers that
    target common nested steering scenarios for Conductor-driven workflows.
    """

    async def pause_actor(
        self,
        reason: str,
        images: "ImageRefs | list | None" = None,
    ) -> Dict[str, Any]:
        """
        Pause any in-flight Actor/TaskScheduler execution and announce the pause.

        Parameters
        ----------
        reason : str
            Human-readable reason to include in the interjection.
        images : ImageRefs | list[RawImageRef | AnnotatedImageRef] | None
            Optional image references to include with the interjections. When provided,
            these are threaded into both the Conductor interjection and the child
            Actor/TaskScheduler interjections (if supported by the child handle).

        Returns
        -------
        dict
            Summary of applied/skipped operations from nested steering.
        """

        message = f"<Actor has been paused due to {reason}>"
        child_message = f"<execution was paused due to {reason}>"
        # When images are supplied, forward them as images for both root and child interjections
        interject_kwargs = {"args": child_message}
        if images is not None:
            interject_kwargs["kwargs"] = {"images": images}
        # Interject only if at least one pause actually applies to a child.
        # For each matched child, pause first, then immediately interject within the child's context.
        spec: Dict[str, Any] = {
            "children": {
                "TaskScheduler.execute": {
                    "steps": [
                        {"method": "pause"},
                        {"method": "interject", **interject_kwargs},
                    ],
                },
                "Actor.act": {
                    "steps": [
                        {"method": "pause"},
                        {"method": "interject", **interject_kwargs},
                    ],
                },
            },
            "conditions": [
                {
                    "when": {
                        "any": [
                            {"selector": "TaskScheduler.execute", "status": "full"},
                            {"selector": "Actor.act", "status": "full"},
                        ],
                    },
                    "then": [
                        {"method": "interject", "args": message, **interject_kwargs},
                    ],
                },
            ],
        }
        return await self.nested_steer(spec)

    async def resume_actor(
        self,
        reason: str,
        images: "ImageRefs | list | None" = None,
    ) -> Dict[str, Any]:
        """
        Resume any in-flight Actor/TaskScheduler execution and announce the resume.

        Parameters
        ----------
        reason : str
            Human-readable reason to include in the interjection.
        images : ImageRefs | list[RawImageRef | AnnotatedImageRef] | None
            Optional image references to include with the interjections. When provided,
            these are threaded into both the Conductor interjection and the child
            Actor/TaskScheduler interjections (if supported by the child handle).

        Returns
        -------
        dict
            Summary of applied/skipped operations from nested steering.
        """

        message = f"<Actor has been resumed due to {reason}>"
        child_message = f"<execution was resumed due to {reason}>"
        # When images are supplied, forward them as images for both root and child interjections
        interject_kwargs = {"args": child_message}
        if images is not None:
            interject_kwargs["kwargs"] = {"images": images}
        # Interject only if at least one resume actually applies to a child.
        # For each matched child, interject first (so the user sees the reason), then resume immediately after.
        spec: Dict[str, Any] = {
            "children": {
                "TaskScheduler.execute": {
                    "steps": [
                        {"method": "interject", **interject_kwargs},
                        {"method": "resume"},
                    ],
                },
                "Actor.act": {
                    "steps": [
                        {"method": "interject", **interject_kwargs},
                        {"method": "resume"},
                    ],
                },
            },
            "conditions": [
                {
                    "when": {
                        "any": [
                            {"selector": "TaskScheduler.execute", "status": "full"},
                            {"selector": "Actor.act", "status": "full"},
                        ],
                    },
                    "then": [
                        {"method": "interject", "args": message, **interject_kwargs},
                    ],
                },
            ],
        }
        return await self.nested_steer(spec)
