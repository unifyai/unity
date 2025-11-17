from pydantic import BaseModel, Field
from typing import List, Optional


class LaterGroup(BaseModel):
    task_ids: List[int] = Field(min_length=1)
    queue_start_at: Optional[str] = None


class QueuePlan(BaseModel):
    now: List[int] = Field(min_length=1)
    later_groups: List[LaterGroup] = Field(default_factory=list)
    notes: Optional[str] = None
