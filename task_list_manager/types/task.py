from pydantic import BaseModel
from typing import Optional
from task_list_manager.types.priority import Priority
from task_list_manager.types.status import Status
from task_list_manager.types.schedule import Schedule
from task_list_manager.types.repetition import RepeatPattern


class Task(BaseModel):
    task_id: int
    name: str
    description: str
    status: Status
    schedule: Schedule
    deadline: Optional[str]
    repeat: Optional[RepeatPattern]
    priority: Priority
