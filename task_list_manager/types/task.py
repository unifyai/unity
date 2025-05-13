from pydantic import BaseModel
from typing import Optional, List
from task_list_manager.types.priority import Priority
from task_list_manager.types.status import Status


class Task(BaseModel):
    task_id: int
    name: str
    description: str
    status: Status
    start_at: Optional[str]
    deadline: Optional[str]
    repeat: Optional[List[str]]
    priority: Optional[Priority]
