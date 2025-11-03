import asyncio
import inspect
from typing import Literal, Optional, Union
import asyncio
from pydantic import BaseModel, Field, create_model
from unity.conversation_manager_2.domains import comms_utils


# conductor
class ConductorAction(BaseModel):
    """Ask or request the Conductor to perform a task."""

    action_name: Literal["conductor_ask", "conductor_request"] = Field(
        ...,
        description=(
            "The action to perform on the Conductor. Options are:\n"
            "'conductor_ask': read-only request\n"
            "'conductor_request': read-write request\n"
        ),
    )


class ConductorHandleAction(BaseModel):
    """Intervene on an existing Conductor handle."""

    handle_id: int
    action_name: Literal[
        "conductor_handle_ask",
        "conductor_handle_interject",
        "conductor_handle_stop",
        "conductor_handle_pause",
        "conductor_handle_resume",
        "conductor_handle_done",
        "conductor_handle_answer_clarification",
    ] = Field(
        ...,
        description=(
            "The action to perform on the handle. Options are:\n"
            "'conductor_handle_ask': ask about the conductor status to the handle\n"
            "'conductor_handle_interject': interject the handle with more information\n"
            "'conductor_handle_stop': stop the handle\n"
            "'conductor_handle_pause': pause the handle\n"
            "'conductor_handle_resume': resume the handle\n"
            "'conductor_handle_done': check if the handle is done\n"
            "'conductor_handle_answer_clarification': answer a clarification question\n"
        ),
    )


class ConductorAnswerClarificationAction(BaseModel):
    """Answer a clarification question."""

    action_name: Literal["conductor_answer_clarification"]
    handle_id: int
    call_id: str


# wait
class WaitForNextEvent(BaseModel):
    action_name: Literal["wait"]


# comms actions (main user)
# whatsapp has some issues, will deal with it later
# class SendWhatsapp(BaseModel):
#     ...


class SendEmail(BaseModel):
    """Comms method to send emails"""

    action_name: Literal["send_email"]
    contact_id: int = Field(
        ...,
        description="contact id, should be -1 if you can not infer the contact from the active conversation, otherwise the contact's id as shown in active conversations",
    )
    first_name: str
    surname: Optional[str]
    email_address: str
    subject: str
    body: str


class SendSMS(BaseModel):
    """Comms method to send sms"""

    action_name: Literal["send_sms"]
    contact_id: int = Field(
        ...,
        description="contact id, should be -1 if you can not infer the contact from the active conversation, otherwise the contact's id as shown in active conversations",
    )
    first_name: str
    surname: Optional[str]
    phone_number: str
    message: str


class MakeCall(BaseModel):
    """Comms method to make outbound calls"""

    action_name: Literal["make_call"]
    contact_id: int = Field(
        ...,
        description="contact id, should be -1 if you can not infer the contact from the active conversation, otherwise the contact's id as shown in active conversations",
    )
    first_name: Optional[str]
    surname: Optional[str]
    phone_number: str


class SendUnifyMessage(BaseModel):
    """Send a message to the boss chat (no-phone medium)"""

    action_name: Literal["send_unify_message"]
    message: str
    contact_id: Literal[1] = 1


def build_dynamic_response_models(
    include_email: bool = True,
    include_sms: bool = True,
    include_call: bool = True,
):
    """
    Dynamically create response models with conditional actions based on available contact info.

    Args:
        include_email: Whether SendEmail action should be available
        include_sms: Whether SendSMS action should be available
        include_call: Whether MakeCall action should be available

    Returns:
        dict: Response models for different modes (call, gmeet, text)
    """
    # Build list of always available action types
    available_actions = [
        ConductorAction,
        ConductorHandleAction,
        WaitForNextEvent,
        SendUnifyMessage,
    ]

    if include_email:
        available_actions.append(SendEmail)
    if include_sms:
        available_actions.append(SendSMS)
    if include_call:
        available_actions.append(MakeCall)

    # Create dynamic Union of available actions
    ActionsUnion = Union[tuple(available_actions)]

    # Dynamically create Response model for text mode
    DynamicResponse = create_model(
        "DynamicResponse",
        thoughts=(str, ...),
        actions=(Optional[list[ActionsUnion]], ...),
        __base__=BaseModel,
    )

    # Dynamically create ResponsePhone model for call/gmeet modes
    DynamicResponsePhone = create_model(
        "DynamicResponsePhone",
        thoughts=(str, ...),
        phone_utterance=(str, ...),
        actions=(Optional[list[ActionsUnion]], ...),
        __base__=BaseModel,
    )

    return {
        "call": DynamicResponsePhone,
        "gmeet": DynamicResponsePhone,
        "unify_call": DynamicResponsePhone,
        "text": DynamicResponse,
    }



class Action:
    action_handlers = {}
    
    @classmethod
    def take_action(cls, action_name, _as_task=True, *args, **kwargs):
        f = cls.action_handlers.get(action_name)
        if not f:
            raise Exception(f"unregisted action: {action_name}, make sure to register action")
        if inspect.iscoroutinefunction(f):
            if _as_task:
                return asyncio.create_task(f(*args, **kwargs))
            else:
                return f(*args, **kwargs)
        else:
            # could be awaitable
            return f(*args, **kwargs)
        

    @classmethod
    def register(cls, action_name: str | list[str]=None):
        def wrapper(func):
            names = [action_name or func.__name__] if not isinstance(action_name, list) else action_name
            for name in names:
                cls.action_handlers[name] = func
            return func
        return wrapper
            


# registered actions, make sure to add *args, **kwargs to make calling these actions easier

@Action.register()
async def send_sms(*args, **kwargs):
    to_number = kwargs.get("phone_number")
    message = kwargs.get("message")
    await comms_utils.send_sms_message_via_number(to_number=to_number, message=message)

@Action.register()
async def send_email(*args, **kwargs):
    to_email = kwargs.get("email_address")
    subject = kwargs.get("subject")
    body = kwargs.get("body")
    await comms_utils.send_email_via_address(to_email=to_email, subject=subject, body=body)

@Action.register()
async def make_call(*args, **kwargs):
    from_number = kwargs.get("assistant_number")
    to_number = kwargs.get("phone_number")
    await comms_utils.start_call(from_number=from_number, to_number=to_number)

@Action.register(
    [
        "conductor_...",
        "conductor_..."
    ]
)
async def _(*args, **kwargs):
    ...