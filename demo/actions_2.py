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
    contact_name: str = Field(..., description="contact name")
    contact_number: str = Field(..., description="contact number with country code")
    detailed_task_description: str = Field(..., description="very detailed description of the task")

class EndTask(BaseModel):
    task_status: Literal["failed", "sucess"] = Field(..., description="The end status of the task")
    task_result: str = Field(..., 
                             description="Summary of the task results, what happened, and what was the conclusion, will be reported back to the main user agent")

class AskUserAgent(BaseModel):
    query: str = Field(...,
                       "The question or clarification that is going to be sent to the main user agent")



# -------- discriminated union --------
# actually, call and non-call modes will probably have some difference between their actions (as in more actions probably for the voice one),
# but for now lets keep them the same
ActionModel = Union[
    SendEmailAction,
    SendWhatsAppMessageAction,
    SendSMSMessageAction,
    CreateCommunicationTask
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
    AskUserAgent,
    EndTask
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