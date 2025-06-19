from pydantic import BaseModel, Field, model_validator
from typing import Optional, List

from .priority import Priority
from .status import Status
from .schedule import Schedule
from .repetition import RepeatPattern
from datetime import datetime

UNASSIGNED = -1


class Task(BaseModel):
    task_id: int = Field(description="Unique identifier for the task", ge=-1)
    name: str = Field(description="Short title of the task")
    description: str = Field(
        description="Detailed explanation of what the task involves",
    )
    status: Status = Field(
        description="Current state of the task (e.g., queued, active, completed)",
    )
    schedule: Optional[Schedule] = Field(
        description="Information about task scheduling, including adjacent tasks in the queue and ideal start time",
    )
    deadline: Optional[datetime] = Field(
        description="Due date/time for the task in ISO-8601 format",
    )
    repeat: Optional[List[RepeatPattern]] = Field(
        description="Pattern defining how the task recurs over time",
    )
    priority: Priority = Field(
        description="Importance level of the task (low, normal, high, urgent)",
    )

    @model_validator(mode="before")
    @classmethod
    def _inject_sentinel(cls, data: dict) -> dict:
        data.setdefault("task_id", UNASSIGNED)
        return data

    def to_post_json(self) -> dict:
        exclude = {"task_id"} if self.task_id == UNASSIGNED else {}
        return self.model_dump(mode="json", exclude=exclude)
