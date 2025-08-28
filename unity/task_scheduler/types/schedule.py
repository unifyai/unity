from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, model_validator


class Schedule(BaseModel):
    queue_id: Optional[int] = Field(
        default=None,
        description="Identifier of the logical queue/chain this task belongs to",
    )
    next_task: Optional[int] = Field(
        default=None,
        description="ID of the next task in the sequence, used for task dependencies and ordering",
    )
    prev_task: Optional[int] = Field(
        default=None,
        description="ID of the previous task in the sequence, used for task dependencies and ordering. If this is set, then `start_at` *must* be `None`",
    )
    start_at: Optional[datetime] = Field(
        default=None,
        description="The scheduled start time for the task in ISO-8601 format. Can *only* be set if the task is the head of the queue (`prev_task` is `None`)",
    )

    @model_validator(mode="before")
    @classmethod
    def _no_start_at_with_prev(cls, data: dict) -> dict:
        """
        **Invariant #1** – A task that sits *behind* another one
        (``prev_task`` ≠ None) inherits its timing from the **head** of the
        queue and therefore **MUST NOT** carry its own ``start_at``.
        """
        if data.get("prev_task") is not None and data.get("start_at") is not None:
            raise ValueError(
                "Cannot specify 'start_at' together with 'prev_task'. "
                "Place the timestamp on the queue head instead.",
            )
        return data
