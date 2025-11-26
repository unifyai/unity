import asyncio
import inspect
from typing import Literal, Optional, Union, TYPE_CHECKING
import asyncio
from pydantic import BaseModel, Field, create_model
from unity.conversation_manager.domains import comms_utils
from unity.conversation_manager.domains import managers_utils
from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.new_events import *
from unity.conversation_manager.domains.utils import log_task_exc
from unity.conversation_manager.domains.contact_index import Contact

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager

event_broker = get_event_broker()


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
    query: str = Field(
        ...,
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


class ContactDetails(BaseModel):
    first_name: Optional[str]
    surename: Optional[str]


class ContactDetailsPhone(ContactDetails):
    phone_number: Optional[str]


class ContactDetailsEmail(ContactDetails):
    email_address: Optional[str]


class SendEmail(BaseModel):
    """Comms method to send emails"""

    action_name: Literal["send_email"]
    contact_id: Optional[int] = Field(
        ...,
        description="contact id, leave as None if you can not infer the contact from the active conversation, otherwise the contact's id as shown in active conversations",
    )
    contact_details: Optional[ContactDetailsEmail]
    subject: str = Field(
        ...,
        description="the subject of the email, should be the same as the subject of the received email without any prefix.",
    )
    body: str
    message_id: Optional[str] = Field(
        ...,
        description="the message id of the email, should be the same as the message id of the received email.",
    )


class SendSMS(BaseModel):
    """Comms method to send sms"""

    action_name: Literal["send_sms"]
    contact_id: Optional[int] = Field(
        ...,
        description="contact id, leave as None if you can not infer the contact from the active conversation, otherwise the contact's id as shown in active conversations",
    )
    contact_details: Optional[ContactDetailsPhone] = Field(
        ...,
        description="contact details if you can not infer the contac_id (because it is not in the active conversations), contact details will be used to retrieve the contact if it exists or create a new one",
    )
    message: str


class MakeCall(BaseModel):
    """Comms method to make outbound calls"""

    action_name: Literal["make_call"]
    contact_id: Optional[int] = Field(
        ...,
        description="contact id, leave as None if you can not infer the contact from the active conversation, otherwise the contact's id as shown in active conversations",
    )
    contact_details: Optional[ContactDetailsPhone]


class SendUnifyMessage(BaseModel):
    """Send a message to the boss chat on the unify platform (no-phone medium)"""

    action_name: Literal["send_unify_message"]
    message: str
    # could remove this if the contact_id is always 1
    contact_id: Literal[1] = 1


def build_dynamic_response_models(realtime=False):
    """
    Create response models.

    Args:
        realtime: Whether the response model is for realtime mode

    Returns:
        dict: Response models for different modes (call, gmeet, text)
    """
    # Build list of always available action types
    available_actions = [
        ConductorAction,
        ConductorHandleAction,
        WaitForNextEvent,
        SendUnifyMessage,
        SendEmail,
        SendSMS,
        MakeCall,
    ]

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
    if not realtime:
        DynamicResponsePhone = create_model(
            "DynamicResponsePhone",
            thoughts=(str, ...),
            phone_utterance=(str, ...),
            actions=(Optional[list[ActionsUnion]], ...),
            __base__=BaseModel,
        )
    else:
        DynamicResponsePhone = create_model(
            "DynamicResponsePhone",
            thoughts=(str, ...),
            phone_guidance=(str, ...),
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
    def take_action(cls, cm, action_name, _as_task=True, *args, **kwargs):
        f = cls.action_handlers.get(action_name)
        if not f:
            raise Exception(
                f"unregisted action: {action_name}, make sure to register action",
            )
        if inspect.iscoroutinefunction(f):
            if _as_task:
                t = asyncio.create_task(f(cm, action_name, *args, **kwargs))
                t.add_done_callback(log_task_exc)
                return t
            else:
                return f(*args, **kwargs)
        else:
            # could be awaitable
            return f(*args, **kwargs)

    @classmethod
    def register(cls, action_name: str | list[str] = None):
        def wrapper(func):
            names = (
                [action_name or func.__name__]
                if not isinstance(action_name, list)
                else action_name
            )
            for name in names:
                cls.action_handlers[name] = func
            return func

        return wrapper


# utils
async def get_update_or_create_contact(
    cm: "ConversationManager",
    contact_id: int = None,
    details: dict = None,
):
    if not contact_id and not details:
        # bad
        ...
    # means update
    elif contact_id and details:
        contact = cm.contact_index.get_contact(contact_id=contact_id)
        data_to_insert = {}
        for k, v in details.items():
            if v:
                if contact[k] != v:
                    data_to_insert[k] = v
        updated_contacts_raw = cm.contact_manager.get_contact_info(
            contact_id=list(cm.contact_index.contacts.keys()),
        )
        # Update contacts dict with Contact objects
        for cid, uc in updated_contacts_raw.items():
            if cid in cm.contact_index.contacts:
                existing = cm.contact_index.contacts[cid]
                cm.contact_index.contacts[cid] = Contact(
                    **{**existing.model_dump(), **uc, "threads": existing.threads},
                )
            else:
                cm.contact_index.contacts[cid] = Contact(**uc)
        # Update active_conversations similarly
        for cid, c in cm.contact_index.active_conversations.items():
            if cid in updated_contacts_raw:
                uc = updated_contacts_raw[cid]
                cm.contact_index.active_conversations[cid] = Contact(
                    **{**c.model_dump(), **uc, "threads": c.threads},
                )
        contact = (
            cm.contact_index.get_contact(phone_number=phone)
            if phone
            else cm.contact_index.get_contact(email=email)
        )
        return contact

    # means retrieve if exists, create if not
    elif details:
        phone, email = details.get("phone_number"), details.get("email")
        maybe_contact = cm.contact_index.get_contact(
            phone_number=phone,
        ) or cm.contact_index.get_contact(email=email)
        if maybe_contact:
            return maybe_contact
        tool_outcome = await asyncio.to_thread(
            cm.contact_manager._create_contact,
            **details,
        )
        new_contact_id = tool_outcome["details"]["contact_id"]
        new_contact = await asyncio.to_thread(
            cm.contact_manager.get_contact_info,
            new_contact_id,
        )
        cm.contact_index.contacts[new_contact_id] = Contact(
            **new_contact[new_contact_id],
        )
        # all good, maybe no need to get all contacts here
        return new_contact[new_contact_id]

    # just retrieve
    elif contact_id:
        # means just message this person directly
        return cm.contact_index.get_contact(contact_id=contact_id)


# registered actions, make sure to add *args, **kwargs to make calling these actions easier
# TODO: add sending/performing [action] notification when actions are made


@Action.register()
async def wait(cm, action_name, *args, **kwargs):
    # does nothing
    pass


@Action.register()
async def send_sms(cm: "ConversationManager", action_name: str, *args, **kwargs):
    # ToDo: either include contact details in prompt and uncomment this
    # or remove this altogether
    # contact_id = kwargs.get("contact_id")
    contact_id = kwargs.get("contact_id")
    contact_details = kwargs.get("contact_details")
    message = kwargs.get("message")
    contact = await get_update_or_create_contact(
        cm,
        contact_id,
        contact_details,
    )
    to_number = contact.get("phone_number")
    response = await comms_utils.send_sms_message_via_number(
        to_number=to_number,
        message=message,
    )

    if response["success"]:
        contact = cm.contact_index.get_contact(phone_number=to_number)
        event = SMSSent(contact=contact, content=message)
    else:
        if not cm.assistant_number:
            error_msg = "You don't have a number, please provision one."
        else:
            error_msg = f"Failed to send sms to {to_number}"
        event = Error(error_msg)
    await event_broker.publish("app:comms:sms_sent", event.to_json())


@Action.register()
async def send_unify_message(
    cm: "ConversationManager",
    action_name: str,
    *args,
    **kwargs,
):
    message = kwargs.get("message")
    contact_id = kwargs.get("contact_id")
    response = await comms_utils.send_unify_message(message=message)
    if response["success"]:
        contact = cm.contact_index.get_contact(contact_id=contact_id)
        event = UnifyMessageSent(contact=contact, content=message)
    else:
        event = Error(f"Failed to send unify message")
    await event_broker.publish("app:comms:unify_message_sent", event.to_json())


@Action.register()
async def send_email(cm: "ConversationManager", action_name: str, *args, **kwargs):
    # ToDo: either include contact details in prompt and uncomment this
    # or remove this altogether
    contact_id = kwargs.get("contact_id")
    contact_details = kwargs.get("contact_details")
    contact = await get_update_or_create_contact(
        cm,
        contact_id,
        contact_details,
    )
    to_email = contact.get("email")
    subject = kwargs.get("subject")
    body = kwargs.get("body")
    message_id = kwargs.get("message_id")
    response = await comms_utils.send_email_via_address(
        to_email=to_email,
        subject=subject,
        body=body,
        message_id=message_id,
    )
    if response["success"]:
        contact = cm.contact_index.get_contact(email=to_email)
        event = EmailSent(
            contact=contact,
            body=body,
            subject=subject,
            message_id=message_id,
        )
    else:
        if not cm.assistant_email:
            error_msg = "You don't have an email address, please provision one."
        else:
            error_msg = f"Failed to send email to {to_email}"
        event = Error(error_msg)
    await event_broker.publish("app:comms:email_sent", event.to_json())


@Action.register()
async def make_call(cm: "ConversationManager", action_name: str, *args, **kwargs):
    # ToDo: either include contact details in prompt and uncomment this
    # or remove this altogether
    contact_id = kwargs.get("contact_id")
    contact_details = kwargs.get("contact_details")
    contact = await get_update_or_create_contact(
        cm,
        contact_id,
        contact_details,
    )
    to_number = contact.get("phone_number")
    response = await comms_utils.start_call(to_number=to_number)
    if response["success"]:
        contact = cm.contact_index.get_contact(phone_number=to_number)
        event = PhoneCallSent(contact=contact)
    else:
        if not cm.assistant_number:
            error_msg = "You don't have a number, please provision one."
        else:
            error_msg = f"Failed to send call to {to_number}"
        event = Error(error_msg)
    await event_broker.publish("app:comms:make_call", event.to_json())


_next_handle_id = 0


@Action.register(["conductor_ask", "conductor_request"])
async def conductor_ask_request(
    cm: "ConversationManager",
    action_name: str,
    *args,
    **kwargs,
):
    """Start a Conductor ask/request, store handle, and publish started."""
    global _next_handle_id
    query = kwargs["query"]
    if "ask" in action_name:
        handle = await cm.conductor.ask(
            query,
            _parent_chat_context=cm.chat_history,
        )
    else:
        handle = await cm.conductor.request(
            query,
            _parent_chat_context=cm.chat_history,
        )

    # allocate handle id and register
    handle_id = _next_handle_id
    _next_handle_id += 1
    cm.conductor_handles[handle_id] = {
        "handle": handle,
        "query": query,
        "handle_actions": [],
    }

    # publish started
    await event_broker.publish(
        f"app:conductor:conductor_started_handle_{handle_id}",
        ConductorHandleStarted(
            handle_id=handle_id,
            action_name=action_name,
            query=query,
        ).to_json(),
    )

    # spawn watchers
    asyncio.create_task(managers_utils.conductor_watch_result(handle_id, handle))
    asyncio.create_task(managers_utils.conductor_watch_notifications(handle_id, handle))
    asyncio.create_task(
        managers_utils.conductor_watch_clarifications(handle_id, handle),
    )


@Action.register([...])
async def conductor_handle_actions(
    cm: "ConversationManager",
    action_name: str,
    *args,
    **kwargs,
):
    handle_id = kwargs["handle_id"]
    query = kwargs["query"]
    handle_data = cm.handle_registry.get(handle_id)
    if not handle_data:
        print(f"[ManagersWorker] Unknown handle_id={handle_id} for action")
        return

    # record intervention
    handle_data["handle_actions"].append(
        {"action_name": action_name, "query": query},
    )
    handle = handle_data["handle"]

    # perform intervention
    result = ""
    try:
        match action_name:
            case "ask":
                ask_handle = await handle.ask(
                    query,
                    parent_chat_context_cont=cm.chat_history,
                )
                result = await ask_handle.result()
            case "interject":
                await handle.interject(
                    query,
                    parent_chat_context_cont=cm.chat_history,
                )
                result = "Handle Interjected"
            case "stop":
                handle.stop(reason=query)
                result = "Handle Stopped"
            case "pause":
                handle.pause()
                result = "Handle Paused"
            case "resume":
                handle.resume()
                result = "Handle Resumed"
            case "done":
                done_result = handle.done()
                result = "Handle Done" if done_result else "Handle Not Done"
            case _:
                print(
                    f"[ManagersWorker] Unknown action_name={action_name} for intervention",
                )
                return
    except Exception as e:
        result = f"Error in conductor handle request: {e}"
        print(f"[ManagersWorker] {result}")

    # publish response
    await event_broker.publish(
        f"app:conductor:handle_{handle_id}_{action_name}_issued",
        ConductorHandleResponse(
            handle_id=handle_id,
            action_name=action_name,
            query=query,
            response=f"Intervened: {action_name} {result}",
        ).to_json(),
    )
