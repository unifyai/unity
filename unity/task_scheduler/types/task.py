from pydantic import BaseModel, Field, model_validator
from typing import Optional, List

from .priority import Priority
from .status import Status
from .schedule import Schedule
from .trigger import Trigger
from .repetition import RepeatPattern
from .activated_by import ActivatedBy
from datetime import datetime

UNASSIGNED = -1


class Task(BaseModel):
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
    activated_by: Optional[ActivatedBy] = Field(
        default=None,
        description=(
            "Reason the task instance transitioned to the active state.\n"
            "This is set automatically at activation time and is never directly editable."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _inject_sentinel(cls, data: dict) -> dict:
        data.setdefault("task_id", UNASSIGNED)
        data.setdefault("instance_id", UNASSIGNED)
        return data

    @model_validator(mode="after")
    def _mutually_exclusive_schedule_trigger(self):
        """
        * `schedule` **xor** `trigger` &nbsp;– never both.
        * If `trigger` is present the status **must** be *triggerable*.
        * Status *triggerable* **requires** a non-null trigger.
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
        return self.model_dump(mode="json", exclude=exclude)
