"""Plan for restoring a task to its previous queue position after isolated activation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from .status import Status


@dataclass
class ReintegrationPlan:
    """
    Corrective plan that restores a task to its prior queue position after an
    isolated activation.

    Stored transiently by `TaskScheduler` and cleared once applied or when the
    associated instance reaches a terminal state.
    """

    task_id: int
    instance_id: int
    prev_task: Optional[int]
    next_task: Optional[int]
    start_at: Optional[datetime]
    was_head: bool
    original_status: Optional[Status]
    head_start_at: Optional[datetime] = None
    queue_id: Optional[int] = None
