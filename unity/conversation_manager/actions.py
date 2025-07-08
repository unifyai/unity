from typing import List, Optional

from pydantic import BaseModel, Field


from typing import List, Literal, Union, Optional
from pydantic import BaseModel, Field


# -------- action variants --------


class SendCallAction(BaseModel):
    type: Literal["call"]


class ConductorAction(BaseModel):
    type: Literal["ask", "request"]
    query: str = Field(..., description="The query to perform")
    show_steps: bool = Field(
        ...,
        description="Whether to show the reasoning steps of the conductor",
    )


class ConductorHandleAction(BaseModel):
    handle_id: int = Field(..., description="The id of the conductor handle")
    type: Literal["ask", "interject", "stop", "pause", "resume"]
    query: str = Field(None, description="The query to perform")


# -------- discriminated union --------
# actually, call and non-call modes will probably have some difference between their actions (as in more actions probably for the voice one),
# but for now lets keep them the same
ActionModel = Union[
    SendCallAction,
    ConductorAction,
    ConductorHandleAction,
]


# -------- assistant output --------
class CallAssistantOutput(BaseModel):
    phone_utterance: str = Field(
        ...,
        description="Your response to the user over the phone",
    )
    actions: Optional[List[ActionModel]] = Field(
        ...,
        description="Actions the assistant should perform",
    )


class AssistantOutput(BaseModel):
    actions: List[ActionModel] = Field(
        ...,
        description="Actions the assistant should perform",
    )
