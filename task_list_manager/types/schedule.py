from pydantic import BaseModel


class Schedule(BaseModel):
    before_task: int
    after_task: int
    start_time: str
