from typing import List
from pydantic import BaseModel


class Summary(BaseModel):
    exchange_ids: List[int]
    summary: str
