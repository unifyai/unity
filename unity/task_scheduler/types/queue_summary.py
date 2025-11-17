from dataclasses import dataclass
from typing import List


@dataclass
class QueueSummary:
    queue_id: int
    order: List[int]
    start_at: str | None
