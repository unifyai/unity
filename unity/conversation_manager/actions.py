from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
from unity.contact_manager.types.contact import Contact


# -------- action variants --------


class SendCallAction(BaseModel):
    type: Literal["call"]


class ToolUseAction(BaseModel):
    # type: Literal["ask", "request"]
    query: str = Field(..., description="The query to perform")
    show_steps: bool = Field(
        ...,
        description="Whether to show the reasoning steps of the tool_use",
    )


class ToolUseHandleAction(BaseModel):
    handle_id: int = Field(..., description="The id of the tool_use handle")
    type: Literal["ask", "interject", "stop", "pause", "resume"]
    query: str = Field(None, description="The query to perform")


# -------- discriminated union --------
# actually, call and non-call modes will probably have some difference between their actions (as in more actions probably for the voice one),
# but for now lets keep them the same
ActionModel = Union[
    # SendCallAction,
    ToolUseAction,
    ToolUseHandleAction,
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


class NewContact(Contact):
    model_config = {"extra": "forbid"}


class ImplicitContactOutput(BaseModel):
    new_contact: bool = Field(
        ...,
        description="Whether a new contact should be created",
    )
    contact: NewContact = Field(
        ...,
        description="The contact to create",
    )
