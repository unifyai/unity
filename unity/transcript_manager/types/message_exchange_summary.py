from typing import List
from pydantic import BaseModel, Field


class MessageExchangeSummary(BaseModel):
    summary_id: int = Field(description="Unique identifier for the message summary")
    exchange_ids: List[int] = Field(
        description="List of message exchange IDs that this summary covers",
    )
    summary: str = Field(
        description="A concise summary of the key points and outcomes from the message exchanges",
    )
