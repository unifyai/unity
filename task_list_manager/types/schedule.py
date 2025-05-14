from pydantic import BaseModel, Field


class Schedule(BaseModel):
    next_task: int = Field(
        description="ID of the next task in the sequence, used for task dependencies and ordering",
    )
    prev_task: int = Field(
        description="ID of the previous task in the sequence, used for task dependencies and ordering",
    )
    start_time: str = Field(
        description="The scheduled start time for the task in ISO-8601 format",
    )
