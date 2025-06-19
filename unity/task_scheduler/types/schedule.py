from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Schedule(BaseModel):
    next_task: Optional[int] = Field(
        description="ID of the next task in the sequence, used for task dependencies and ordering",
    )
    prev_task: Optional[int] = Field(
        description="ID of the previous task in the sequence, used for task dependencies and ordering",
    )
    start_at: Optional[datetime] = Field(
        default=None,
        description="The scheduled start time for the task in ISO-8601 format. Only set when the user explicitly schedules the task.",
    )
