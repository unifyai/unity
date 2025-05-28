from typing import List
from pydantic import BaseModel


class MessageExchangeSummary(BaseModel):
    exchange_ids: List[int]
    summary: str
