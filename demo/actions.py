from typing import List, Optional

from pydantic import BaseModel, Field


from typing import List, Literal, Union, Optional
from pydantic import BaseModel, Field


# -------- action variants --------
class SendWhatsAppMessageAction(BaseModel):
    type: Literal["whatsapp"]
    message: str


class SendSMSMessageAction(BaseModel):
    type: Literal["sms"]
    message: str


class SendEmailAction(BaseModel):
    type: Literal["email"]
    subject: str
    body: str


class CreateCommunicationTask(BaseModel):
    contact_name: str = Field(..., description="contact name, MUST be provided")
    contact_number: str = Field(
        ...,
        description="contact number with country code, MUST be provided",
    )
    detailed_task_description: str = Field(
        ...,
        description="very detailed description of the task",
    )


class RespondToAgent(BaseModel):
    agent_id: str = Field(..., description="Id of the agent to respond to")
    task_id: str = Field(..., description="Id of the response is related to")
    response: str = Field(..., description="Your resoonse to the agent's query")


class EndTask(BaseModel):
    task_id: str = Field(..., description="ID of the task")
    task_status: Literal["failed", "sucess"] = Field(
        ...,
        description="The end status of the task",
    )
    task_result: str = Field(
        ...,
        description="Summary of the task results, what happened, and what was the conclusion, will be reported back to the main user agent",
    )


class AskUserAgent(BaseModel):
    agent_id: str = Field(..., description="Id of the agent asking the query")
    task_id: str = Field(..., description="id of the task the query is associated with")
    query: str = Field(
        ...,
        description="The question or clarification that is going to be sent to the main user agent",
    )


class SendCallAction(BaseModel):
    type: Literal["call"]


class ContactManagerAction(BaseModel):
    type: Literal["update", "ask"]
    query: str = Field(..., description="The query to perform")
    show_steps: bool = Field(
        ...,
        description="Whether to show the reasoning steps of the contact manager",
    )


# -------- discriminated union --------
# actually, call and non-call modes will probably have some difference between their actions (as in more actions probably for the voice one),
# but for now lets keep them the same
ActionModel = Union[
    SendEmailAction,
    SendWhatsAppMessageAction,
    SendSMSMessageAction,
    CreateCommunicationTask,
    RespondToAgent,
    SendCallAction,
    # ContactManagerAction,
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


CommsActionModel = Union[
    SendEmailAction,
    SendWhatsAppMessageAction,
    SendSMSMessageAction,
    SendCallAction,
    AskUserAgent,
    EndTask,
]


class CommsAgentOutput(BaseModel):
    actions: List[CommsActionModel] = Field(
        ...,
        description="Actions the comms agent should perform",
    )


class CallCommsAgentOutput(BaseModel):
    phone_utterance: str = Field(
        ...,
        description="Your response to the user over the phone",
    )
    actions: Optional[List[CommsActionModel]] = Field(
        ...,
        description="Actions the assistant should perform",
    )
