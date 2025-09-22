"""Plan for restoring a task to its previous queue position after isolated activation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ReintegrationPlan:
    """
    Corrective plan that restores a task to its prior queue position after an
    isolated activation.

    Stored transiently by `TaskScheduler` and cleared once applied or when the
    associated instance reaches a terminal state.
    """

    task_id: int
    instance_id: Optional[int]
    prev_task: Optional[int]
    next_task: Optional[int]
    start_at: Optional[str]
    was_head: bool
    original_status: Optional[str]
    head_start_at: Optional[str] = None
    queue_id: Optional[int] = None
