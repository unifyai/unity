from typing import List
from pydantic import BaseModel, Field, model_validator

UNASSIGNED = -1  # indicates “server will assign this later”


class MessageExchangeSummary(BaseModel):
    summary_id: int = Field(
        description="Unique identifier for the message summary",
        ge=-1,
    )
    exchange_ids: List[int] = Field(
        description="List of message exchange IDs that this summary covers",
    )
    message_ids: List[int] = Field(
        description="Exact message IDs that the assistant used when generating the summary",
        default_factory=list,
    )
    summary: str = Field(
        description="A concise summary of the key points and outcomes from the message exchanges",
    )

    # Inject sentinel when missing
    @model_validator(mode="before")
    @classmethod
    def _inject_sentinel(cls, data: dict) -> dict:
        data.setdefault("summary_id", UNASSIGNED)
        return data

    def to_post_json(self) -> dict:
        """Return JSON payload for POST – omit dummy id."""
        exclude = {"summary_id"} if self.summary_id == UNASSIGNED else {}
        return self.model_dump(mode="json", exclude=exclude)
