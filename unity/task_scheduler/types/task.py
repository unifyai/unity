from pydantic import BaseModel, Field
from typing import Optional

from .priority import Priority
from .status import Status
from .schedule import Schedule
from .repetition import RepeatPattern


class Task(BaseModel):
    task_id: int = Field(description="Unique identifier for the task")
    name: str = Field(description="Short title of the task")
    description: str = Field(
        description="Detailed explanation of what the task involves",
    )
    status: Status = Field(
        description="Current state of the task (e.g., queued, active, completed)",
    )
    schedule: Schedule = Field(
        description="Information about task scheduling, including adjacent tasks in the queue and ideal start time",
    )
    deadline: Optional[str] = Field(
        description="Due date/time for the task in ISO-8601 format",
    )
    repeat: Optional[RepeatPattern] = Field(
        description="Pattern defining how the task recurs over time",
    )
    priority: Priority = Field(
        description="Importance level of the task (low, normal, high, urgent)",
    )
