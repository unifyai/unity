import asyncio
import inspect
from typing import Literal, Optional, Union, TYPE_CHECKING

from pydantic import BaseModel, Field, create_model

from unity.conversation_manager.domains import comms_utils
from unity.conversation_manager.domains import managers_utils
from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.events import *
from unity.conversation_manager.domains.utils import log_task_exc
from unity.conversation_manager.domains.contact_index import Contact
from unity.conversation_manager.task_actions import (
    STEERING_OPERATIONS,
    derive_short_name,
    build_action_name,
    safe_call_id_suffix,
    parse_action_name,
    is_dynamic_action,
)

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager

event_broker = get_event_broker()


# Starting a new task
class StartTaskAction(BaseModel):
    """Start a new task for the assistant to work on."""

    action_name: Literal["start_task"] = Field(default="start_task")
    query: str = Field(..., description="The task description or question")


# wait
class WaitForNextEvent(BaseModel):
    action_name: Literal["wait"]


# comms actions (main user)


class ContactDetails(BaseModel):
    first_name: Optional[str]
    surname: Optional[str]


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
    email_id_to_reply_to: Optional[str] = Field(
        ...,
        description=(
            "the email identifier of the received email that you are replying to "
            "(shown as `Email ID` in active conversations). "
            "This is used for threading (In-Reply-To / References)."
        ),
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
        description="contact details if you can not infer the contact_id (because it is not in the active conversations), contact details will be used to retrieve the contact if it exists or create a new one",
    )
    content: str


class MakeCall(BaseModel):
    """Comms method to make outbound calls"""

    action_name: Literal["make_call"]
    contact_id: Optional[int] = Field(
        ...,
        description="contact id, leave as None if you can not infer the contact from the active conversation, otherwise the contact's id as shown in active conversations",
    )
    contact_details: Optional[ContactDetailsPhone]


class SendUnifyMessage(BaseModel):
    """Send a Unify message to a contact via the Unify platform (in-app messaging)."""

    action_name: Literal["send_unify_message"]
    content: str
    contact_id: int = Field(
        ...,
        description="Target contact_id as shown in active conversations.",
    )


def _generate_dynamic_task_actions(active_tasks: dict) -> list[type[BaseModel]]:
    """Generate dynamic Pydantic action models for each active task.

    Uses STEERING_OPERATIONS from task_actions module to programmatically
    generate actions based on SteerableToolHandle's methods.
    """
    dynamic_actions = []

    for handle_id, handle_data in (active_tasks or {}).items():
        query = handle_data.get("query", "")
        short_name = derive_short_name(query)
        handle_actions = handle_data.get("handle_actions", [])

        # Get pending clarifications for this handle
        pending_clarifications = [
            a
            for a in handle_actions
            if a.get("action_name") == "clarification_request" and not a.get("response")
        ]

        for op in STEERING_OPERATIONS:
            docstring = op.get_docstring()

            if op.requires_clarification:
                # Only generate if there are pending clarifications
                for clar in pending_clarifications:
                    call_id = clar.get("call_id", "")
                    suffix = safe_call_id_suffix(call_id)
                    action_name = build_action_name(
                        op.name,
                        short_name,
                        handle_id,
                        suffix,
                    )
                    model_name = f"{op.name.title().replace('_', '')}{short_name.title().replace('_', '')}{handle_id}_{suffix}"

                    # Build fields based on operation's param_name
                    fields = {
                        "action_name": (
                            Literal[action_name],
                            Field(default=action_name),
                        ),
                    }
                    if op.param_name:
                        fields[op.param_name] = (str, Field(..., description=docstring))

                    dynamic_actions.append(
                        create_model(model_name, __base__=BaseModel, **fields),
                    )
            else:
                action_name = build_action_name(op.name, short_name, handle_id)
                model_name = f"{op.name.title().replace('_', '')}{short_name.title().replace('_', '')}{handle_id}"

                # Build fields based on operation's param_name
                fields = {
                    "action_name": (Literal[action_name], Field(default=action_name)),
                }
                if op.param_name:
                    # Some params are optional (like "reason" for stop)
                    if op.name == "stop":
                        fields[op.param_name] = (
                            Optional[str],
                            Field(default=None, description=docstring),
                        )
                    else:
                        fields[op.param_name] = (str, Field(..., description=docstring))

                dynamic_actions.append(
                    create_model(model_name, __base__=BaseModel, **fields),
                )

    return dynamic_actions


def build_dynamic_response_models(active_tasks: dict = None):
    """
    Create response models with dynamic per-task actions.

    Args:
        active_tasks: Dict of active task handles {handle_id: {"query": str, "handle": ..., ...}}

    Returns:
        dict: Response models for different modes (call, unify_meet, text)
    """
    # Build list of always available action types
    available_actions = [
        StartTaskAction,
        WaitForNextEvent,
    ]

    # Add dynamic per-task actions
    available_actions.extend(_generate_dynamic_task_actions(active_tasks))

    # Create dynamic Union of available actions
    ActionsUnion = Union[tuple(available_actions)]

    # Dynamically create Response model for text mode
    DynamicResponse = create_model(
        "DynamicResponse",
        thoughts=(str, ...),
        actions=(Optional[list[ActionsUnion]], ...),
        __base__=BaseModel,
    )

    # Dynamically create ResponseVoice model for call/unify_meet modes
    # Both TTS and Realtime modes use call_guidance - the Main CM Brain
    # provides guidance/data to the voice agent (fast brain) which handles
    # the actual conversation. This enables careful orchestration decisions
    # without blocking the voice stream.
    DynamicResponseVoice = create_model(
        "DynamicResponseVoice",
        thoughts=(str, ...),
        call_guidance=(str, ...),
        actions=(Optional[list[ActionsUnion]], ...),
        __base__=BaseModel,
    )

    return {
        "call": DynamicResponseVoice,
        "unify_meet": DynamicResponseVoice,
        "text": DynamicResponse,
    }


class Action:
    action_handlers = {}

    @classmethod
    def take_action(cls, cm, action_name, _as_task=True, *args, **kwargs):
        # Check static handlers first
        f = cls.action_handlers.get(action_name)

        # If not found, check if it's a dynamic task action
        if not f and is_dynamic_action(action_name):
            f = DynamicTaskActionHandler.handle

        if not f:
            raise Exception(
                f"unregistered action: {action_name}, make sure to register action",
            )
        result = f(cm, action_name, *args, **kwargs)
        if inspect.isawaitable(result):
            if _as_task:
                t = asyncio.create_task(result)
                t.add_done_callback(log_task_exc)
                return t
            return result
        return result

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
        phone_number, email_address = details.get("phone_number"), details.get(
            "email_address",
        )
        contact = (
            cm.contact_index.get_contact(phone_number=phone_number)
            if phone_number
            else cm.contact_index.get_contact(email=email_address)
        )
        return contact

    # means retrieve if exists, create if not
    elif details:
        phone_number, email_address = details.get("phone_number"), details.get(
            "email_address",
        )
        maybe_contact = cm.contact_index.get_contact(
            phone_number=phone_number,
        ) or cm.contact_index.get_contact(email=email_address)
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
    content = kwargs.get("content")
    contact = await get_update_or_create_contact(
        cm,
        contact_id,
        contact_details,
    )
    to_number = contact.get("phone_number")
    response = await comms_utils.send_sms_message_via_number(
        to_number=to_number,
        content=content,
    )

    if response["success"]:
        contact = cm.contact_index.get_contact(phone_number=to_number)
        event = SMSSent(contact=contact, content=content)
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
    content = kwargs.get("content")
    contact_id = kwargs.get("contact_id")
    if contact_id is None:
        raise ValueError("contact_id is required for send_unify_message")
    response = await comms_utils.send_unify_message(
        content=content,
        contact_id=contact_id,
    )
    if response["success"]:
        contact = cm.contact_index.get_contact(contact_id=contact_id)
        event = UnifyMessageSent(contact=contact, content=content)
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
    to_email = contact.get("email_address")
    subject = kwargs.get("subject")
    body = kwargs.get("body")
    email_id_to_reply_to = kwargs.get("email_id_to_reply_to")

    # ------------------------------------------------------------------
    # Reduce flakiness: prefer the most recent inbound email's Message-ID
    # for this contact+subject, rather than trusting the LLM to copy it.
    #
    # In practice the model may output `null` (or an incorrect value) because
    # the schema allows Optional[str]. For replies, we can infer the correct
    # Message-ID from the active conversation email thread.
    # ------------------------------------------------------------------
    inferred_reply_id: str | None = None
    try:
        convo = None
        if contact_id is not None:
            convo = cm.contact_index.active_conversations.get(contact_id)
        if convo is None and to_email:
            convo = next(
                (
                    c
                    for c in cm.contact_index.active_conversations.values()
                    if getattr(c, "email_address", None) == to_email
                ),
                None,
            )
        if convo is not None:
            thread = getattr(convo, "threads", {}).get("email")
            if thread:
                for m in reversed(thread):
                    # Prefer the most recent *user* email (name != "You") with the
                    # same subject and a non-empty email_id.
                    if (
                        getattr(m, "name", None) != "You"
                        and getattr(m, "subject", None) == subject
                        and getattr(m, "email_id", None)
                    ):
                        inferred_reply_id = m.email_id
                        break
    except Exception:
        inferred_reply_id = None

    if inferred_reply_id and inferred_reply_id != email_id_to_reply_to:
        email_id_to_reply_to = inferred_reply_id
    response = await comms_utils.send_email_via_address(
        to_email=to_email,
        subject=subject,
        body=body,
        email_id=email_id_to_reply_to,
    )
    if response["success"]:
        contact = cm.contact_index.get_contact(email=to_email)
        event = EmailSent(
            contact=contact,
            body=body,
            subject=subject,
            email_id_replied_to=email_id_to_reply_to,
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


@Action.register(["start_task"])
async def start_task_action(
    cm: "ConversationManager",
    action_name: str,
    *args,
    **kwargs,
):
    """Start a new task, store handle, and publish started."""
    global _next_handle_id

    await managers_utils.wait_for_initialization(cm)
    query = kwargs["query"]

    handle = await cm.conductor.request(
        query,
        _parent_chat_context=cm.chat_history,
    )

    # allocate handle id and register
    handle_id = _next_handle_id
    _next_handle_id += 1
    cm.active_tasks[handle_id] = {
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


class DynamicTaskActionHandler:
    """Handler for dynamic per-task actions derived from SteerableToolHandle methods."""

    @staticmethod
    async def handle(
        cm: "ConversationManager",
        action_name: str,
        *args,
        **kwargs,
    ):
        await managers_utils.wait_for_initialization(cm)

        parsed = parse_action_name(action_name)
        operation = parsed.operation
        handle_id = parsed.handle_id
        call_id_suffix = parsed.call_id_suffix

        handle_data = cm.active_tasks.get(handle_id)
        if not handle_data:
            print(
                f"[TaskHandler] Unknown handle_id={handle_id} for action {action_name}",
            )
            return

        handle = handle_data["handle"]
        steering_op = parsed.steering_operation

        # Extract the appropriate parameter based on operation's param_name
        param_value = ""
        if steering_op and steering_op.param_name:
            param_value = kwargs.get(steering_op.param_name, "")
        # Fallback: check common parameter names
        if not param_value:
            param_value = kwargs.get(
                "query",
                kwargs.get("message", kwargs.get("answer", kwargs.get("reason", ""))),
            )

        # Record intervention
        handle_data["handle_actions"].append(
            {"action_name": action_name, "query": param_value},
        )

        # Perform intervention by calling the corresponding method on handle
        result = ""
        full_call_id = ""
        try:
            match operation:
                case "ask":
                    ask_handle = await handle.ask(
                        param_value,
                        parent_chat_context_cont=cm.chat_history,
                    )
                    result = await ask_handle.result()
                case "interject":
                    await handle.interject(
                        param_value,
                        parent_chat_context_cont=cm.chat_history,
                    )
                    result = "Interjected"
                case "stop":
                    handle.stop(reason=param_value or None)
                    result = "Stopped"
                    cm.active_tasks.pop(handle_id, None)
                case "pause":
                    handle.pause()
                    result = "Paused"
                case "resume":
                    handle.resume()
                    result = "Resumed"
                case "answer_clarification":
                    # Find the full call_id from pending clarifications
                    pending_clars = [
                        a
                        for a in handle_data.get("handle_actions", [])
                        if a.get("action_name") == "clarification_request"
                        and not a.get("response")
                    ]
                    for clar in pending_clars:
                        cid = clar.get("call_id", "")
                        if call_id_suffix and cid.endswith(call_id_suffix):
                            full_call_id = cid
                            break
                    if not full_call_id and pending_clars:
                        full_call_id = pending_clars[0].get("call_id", "")

                    if full_call_id:
                        await handle.answer_clarification(full_call_id, param_value)
                        result = "Clarification Answered"
                    else:
                        result = "No pending clarification found"
                case _:
                    print(
                        f"[TaskHandler] Unknown operation={operation} for {action_name}",
                    )
                    return
        except Exception as e:
            result = f"Error: {e}"
            print(f"[TaskHandler] {result}")

        # publish response
        await event_broker.publish(
            f"app:conductor:handle_{handle_id}_{operation}_issued",
            ConductorHandleResponse(
                handle_id=handle_id,
                action_name=action_name,
                query=param_value,
                response=f"{operation.title()}: {result}",
                call_id=full_call_id,
            ).to_json(),
        )
