from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ReintegrationPlan:
    """
    Minimal corrective plan used to surgically restore a task back to its
    previous queue/schedule position after an isolated activation.

    This value is kept transiently in memory by `TaskScheduler` and cleared
    once applied or when the associated instance reaches a terminal state.
    """

    task_id: int
    instance_id: Optional[int]
    prev_task: Optional[int]
    next_task: Optional[int]
    start_at: Optional[str]
    head_start_at: Optional[str] = None
    was_head: bool
    original_status: Optional[str]
