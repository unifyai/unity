"""Task model: queue membership, scheduling/triggering, priority, and metadata."""

from pydantic import (
    BaseModel,
    Field,
    model_validator,
    model_serializer,
    SerializationInfo,
    SerializerFunctionWrapHandler,
)
from typing import Optional, List, ClassVar

from .priority import Priority
from .status import Status
from .schedule import Schedule
from .trigger import Trigger
from .repetition import RepeatPattern
from .activated_by import ActivatedBy
from datetime import datetime

UNASSIGNED = -1


class Task(BaseModel):
    # Top-level queue identifier for tasks that are members of a runnable queue.
    # When a task is queued/scheduled, this must be populated. The schedule
    # object never carries a queue_id field; use this top-level column solely.
    queue_id: Optional[int] = Field(
        default=None,
        description=("Identifier of the runnable queue this task belongs to."),
    )
    task_id: int = Field(
        default=UNASSIGNED,
        description="Unique identifier for the task",
        ge=UNASSIGNED,
    )
    instance_id: int = Field(
        default=UNASSIGNED,
        description=(
            "Auto-incrementing counter that distinguishes multiple *instances* "
            "of the same logical task.  The very first row receives `0`; "
            "each subsequent clone is incremented by the backend."
        ),
        ge=UNASSIGNED,
    )
    name: str = Field(description="Short title of the task")
    description: str = Field(
        description="Detailed explanation of what the task involves",
    )
    status: Status = Field(
        description="Current state of the task (e.g., queued, active, completed)",
    )
    schedule: Optional[Schedule] = Field(
        default=None,
        description="Information about task scheduling, including adjacent tasks in the queue and ideal start time",
    )
    trigger: Optional[Trigger] = Field(
        default=None,
        description="Event definition that starts the task (mutually exclusive with *schedule*)",
    )
    deadline: Optional[datetime] = Field(
        default=None,
        description="Due date/time for the task in ISO-8601 format",
    )
    repeat: Optional[List[RepeatPattern]] = Field(
        default=None,
        description="Pattern defining how the task recurs over time",
    )
    priority: Priority = Field(
        description="Importance level of the task (low, normal, high, urgent)",
    )
    response_policy: Optional[str] = Field(
        default=None,
        description=(
            "Freeform policy for contact handling during this task (authority, "
            "information visibility, who may interject/steer). When it conflicts with "
            "a contact's own response_policy, the task-level policy takes precedence."
        ),
    )
    entrypoint: Optional[int] = Field(
        default=None,
        description=(
            "Optional function_id from the Functions table that should be invoked to perform this task. "
            "When null, the task is executed by an Actor interpreting the free-form description on the fly."
        ),
    )
    activated_by: Optional[ActivatedBy] = Field(
        default=None,
        description=(
            "Reason the task instance transitioned to the active state.\n"
            "This is set automatically at activation time and is never directly editable."
        ),
    )

    # Central, single source of truth for shorthand aliases (full → shorthand)
    SHORTHAND_MAP: ClassVar[dict[str, str]] = {
        "queue_id": "qid",
        "task_id": "tid",
        "instance_id": "iid",
        "name": "nm",
        "description": "desc",
        "status": "st",
        "schedule": "sched",
        "trigger": "trig",
        "deadline": "dl",
        "repeat": "rep",
        "priority": "prio",
        "response_policy": "policy",
        "entrypoint": "entry",
        "activated_by": "ab",
    }

    @model_validator(mode="before")
    @classmethod
    def _inject_sentinel(cls, data: dict) -> dict:
        data.setdefault("task_id", UNASSIGNED)
        data.setdefault("instance_id", UNASSIGNED)
        return data

    @model_validator(mode="after")
    def _mutually_exclusive_schedule_trigger(self):
        """
        - schedule XOR trigger: never both
        - If trigger is set, status must be triggerable
        - triggerable status requires a non-null trigger
        """
        if self.schedule is not None and self.trigger is not None:
            raise ValueError("A task cannot have both *schedule* and *trigger*.")

        if self.trigger is not None and self.status != Status.triggerable:
            raise ValueError(
                "When *trigger* is set the status must be 'triggerable'.",
            )

        if self.status == Status.triggerable and self.trigger is None:
            raise ValueError(
                "Status 'triggerable' requires a non-null *trigger* definition.",
            )

        # `activated_by` may only be present once the task is actually active
        if self.status != Status.active and self.activated_by is not None:
            raise ValueError(
                "`activated_by` may only be set when status is 'active'",
            )

        return self

    def to_post_json(self) -> dict:
        exclude: set[str] = set()
        if self.task_id == UNASSIGNED:
            exclude.add("task_id")
        if self.instance_id == UNASSIGNED:
            exclude.add("instance_id")
        # Allow backend auto-increment for queue_id by omitting it when unset
        if self.queue_id is None:
            exclude.add("queue_id")
        return self.model_dump(mode="json", exclude=exclude, exclude_none=True)

    # Shorthand helpers (parity with Message/Contact models)
    @classmethod
    def shorthand_map(cls) -> dict[str, str]:
        return dict(cls.SHORTHAND_MAP)

    @classmethod
    def shorthand_inverse_map(cls) -> dict[str, str]:
        return {v: k for k, v in cls.SHORTHAND_MAP.items()}

    # Only affect JSON-mode serialisation: optionally prune empty fields and/or
    # alias keys when explicitly requested via context (prune_empty/shorthand)
    @model_serializer(mode="wrap")
    def _prune_empty_on_serialize(
        self,
        handler: SerializerFunctionWrapHandler,
        info: SerializationInfo,
    ) -> dict:  # type: ignore[no-redef]
        data = handler(self)

        prune = False
        shorthand = False
        try:
            ctx = info.context or {}
            if "prune_empty" in ctx:
                prune = bool(ctx["prune_empty"])  # explicit override
            if "shorthand" in ctx:
                shorthand = bool(ctx["shorthand"])  # explicit aliasing
        except Exception:
            pass

        out = data
        if prune:

            def _is_empty(value):
                try:
                    if value is None:
                        return True
                    # Treat empty strings as empty; keep False/0 as meaningful
                    if isinstance(value, str):
                        return value.strip() == ""
                    if isinstance(value, (list, tuple, set, dict)):
                        return len(value) == 0
                    return False
                except Exception:
                    return False

            def _prune(obj):
                try:
                    if isinstance(obj, dict):
                        pruned = {k: _prune(v) for k, v in obj.items()}
                        return {k: v for k, v in pruned.items() if not _is_empty(v)}
                    if isinstance(obj, list):
                        pruned_list = [_prune(v) for v in obj]
                        return [v for v in pruned_list if not _is_empty(v)]
                    return obj
                except Exception:
                    return obj

            try:
                out = _prune(out)
            except Exception:
                out = data

        if shorthand and isinstance(out, dict):
            alias_map = type(self).SHORTHAND_MAP
            try:
                out = {alias_map.get(k, k): v for k, v in out.items()}
            except Exception:
                out = out

        return out
