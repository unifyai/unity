import asyncio
import os
from time import perf_counter
from typing import TYPE_CHECKING, Union

from unity.contact_manager.types.contact import UNASSIGNED
from unity.conversation_manager.debug_logger import log_job_startup
from unity.conversation_manager.new_events import *
from unity.conversation_manager.domains import managers_utils

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager


class EventHandler:
    _registry = {}

    @classmethod
    def register(cls, event_cls: list[Event] | Event):
        def wrapper(func):
            events_classes = (
                [event_cls] if not isinstance(event_cls, (list, tuple)) else event_cls
            )
            for e in events_classes:
                cls._registry[e] = func
            return func

        return wrapper

    @classmethod
    def handle_event(cls, event: Event, cm: "ConversationManager", *args, **kwargs):
        # maybe add the event bus logging thing here
        print(f"Recieved EVENT: {event}")
        if event.__class__.loggable:
            asyncio.create_task(managers_utils.publish_bus_events(event))
        print(event)
        f = cls._registry.get(event.__class__)
        if not f:
            # do nothing basically (?)
            return asyncio.sleep(0)
        return f(event, cm, *args, **kwargs)


CallEvents = Union[PhoneCallReceived, PhoneCallSent, UnifyCallReceived]


@EventHandler.register((PhoneCallReceived, PhoneCallSent, UnifyCallReceived))
async def _(event: CallEvents, cm: "ConversationManager", *args, **kwargs):
    if cm.mode in ["phone", "gmeet", "unify_call"]:
        # can't make call
        # TODO: we should handle this somehow tbh
        # for now do nothing, but we can think of adding a notification of an attempted call
        ...
    else:
        # update state
        message_content = None
        notif_content = None
        match event:
            case PhoneCallReceived() as e:
                cm.call_manager.start_call(e.contact["phone_number"])
                message_content = "<Recvieving Call...>"
                notif_content = f"Call received from {e.contact['first_name']}"
            case PhoneCallSent() as e:
                cm.call_manager.start_call(e.contact["phone_number"])
                message_content = "<Sending Call...>"
                notif_content = f"Call sent to {e.contact['first_name']}"
            case UnifyCallReceived() as e:
                cm.call_manager.start_unify_call(e.agent_name, e.room_name)
                message_content = "<Recieving Call...>"
                notif_content = f"Call received from {e.contact['first_name']}"

        cm.notifications_bar.push_notif("Comms", notif_content, event.timestamp)
        cm.contact_index.push_message(
            event.contact,
            "phone",
            message_content=message_content,
            role=(
                "user"
                if "received" in event.__class__.__name__.lower()
                else "assistant"
            ),
            timestamp=event.timestamp,
        )


@EventHandler.register((PhoneCallStarted, UnifyCallStarted))
async def _(
    event: PhoneCallStarted | UnifyCallStarted,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    if isinstance(event, PhoneCallStarted):
        cm.mode = "call"
        phone_number = event.contact
        contact = cm.contact_index.get_contact(phone_number=phone_number)
    else:
        cm.mode = "unify_call"
        contact = cm.contact_index.get_contact(contact_id=1)

    cm.call_contact = contact
    cm.notifications_bar.push_notif(
        "Comms",
        f"Phone Call started with {contact['first_name']}",
        timestamp=event.timestamp,
    )
    cm.contact_index.push_message(
        contact, "phone", "<Call Started>", timestamp=event.timestamp
    )
    cm.contact_index.active_conversations[contact["contact_id"]].on_call = True
    await cm.run_llm(delay=0)


@EventHandler.register(
    (
        PhoneUtterance,
        UnifyCallUtterance,
        AssistantPhoneUtterance,
        AssistantUnifyCallUtterance,
    )
)
async def _(event: PhoneCallEnded, cm: "ConversationManager", *args, **kwargs):
    # publish transcript
    # asyncio.create_task(managers_utils.log_message(cm, event))
    if isinstance(event, (PhoneUtterance, AssistantPhoneUtterance)):
        contact = cm.contact_index.get_contact(phone_number=event.contact)
        cm.contact_index.push_message(
            contact,
            "phone",
            event.content,
            role=(
                "user"
                if "assistant" not in event.__class__.__name__.lower()
                else "assistant"
            ),
        )
    # start filler only in non-realtime
    if isinstance(event, (PhoneUtterance, UnifyCallUtterance)):
        if not cm.realtime:
            await cm.cancel_filler()
            asyncio.create_task(cm.run_filler_once())
        await cm.run_llm(delay=0, cancel_running=True)


@EventHandler.register((PhoneCallEnded, UnifyCallEnded))
async def _(
    event: PhoneCallEnded | UnifyCallEnded, cm: "ConversationManager", *args, **kwargs
):
    cm.mode = "text"
    if isinstance(event, PhoneCallEnded):
        cm.call_contact = None
    elif isinstance(event, UnifyCallEnded):
        cm.unify_call_contact = None
    contact = cm.contact_index.get_contact(phone_number=event.contact)
    cm.contact_index.active_conversations[contact["contact_id"]].on_call = False
    cm.call_manager.cleanup_call_proc()
    await cm.cancel_filler()
    await cm.run_llm(delay=0, cancel_running=True)


@EventHandler.register(
    (
        ConductorResponse,
        ConductorHandleResponse,
        ConductorResult,
        ConductorClarificationRequest,
    )
)
async def _(event, cm: "ConversationManager", *args, **kwargs):
    # just run llm here
    ...


@EventHandler.register(
    (
        SMSSent,
        SMSReceived,
        EmailSent,
        EmailReceived,
        UnifyMessageSent,
        UnifyMessageReceived,
    )
)
async def _(event, cm: "ConversationManager", *args, **kwargs):
    asyncio.create_task(managers_utils.log_message(cm, event))

    # update state
    thread = None
    message_content = None
    subject = None
    body = None
    message_id = None
    notif_content = None

    contact = cm.contact_index.get_contact(event.contact["contact_id"])

    match event:
        case SMSSent():
            thread = "sms"
            message_content = event.content
            notif_content = f"SMS sent to {contact['first_name']}"
            role = "assistant"
        case SMSReceived():
            thread = "sms"
            message_content = event.content
            notif_content = f"SMS Received from {contact['first_name']}"
            role = "user"
        case EmailSent():
            thread = "email"
            subject = event.subject
            body = event.body
            message_id = event.message_id
            notif_content = f"Email sent to {contact['first_name']}"
            role = "assistant"
        case EmailReceived():
            thread = "email"
            subject = event.subject
            body = event.body
            message_id = event.message_id
            notif_content = f"Email Received from {contact['first_name']}"
            role = "user"
        case UnifyMessageSent():
            thread = "unify"
            message_content = event.content
            notif_content = f"Unify message sent to {contact['first_name']}"
            role = "assistant"
        case UnifyMessageReceived():
            thread = "unify"
            message_content = event.content
            notif_content = f"Unify message from {contact['first_name']}"
            role = "user"

    message_content = event.content
    cm.contact_index.push_message(
        event.contact,
        thread,
        message_content=message_content,
        subject=subject,
        body=body,
        message_id=message_id,
        timestamp=event.timestamp,
        role=role,
    )
    cm.notifications_bar.push_notif("comms", notif_content, event.timestamp)

    await cm.run_llm(delay=2)


# TODO: put all managers in the cm and move start up logic from managers worker to here


@EventHandler.register((StartupEvent))
async def _(event: StartupEvent, cm: "ConversationManager", *args, **kwargs):
    print("recieved start up event")
    payload = event.to_dict()["payload"]
    cm.set_details(payload)
    if not os.getenv("TEST"):
        kwargs = {
            "timestamp": payload["timestamp"],
            "medium": payload["medium"],
            **cm.get_details(),
        }
        asyncio.create_task(asyncio.to_thread(log_job_startup, **kwargs))


@EventHandler.register(GetContactsResponse)
async def _(event: GetContactsResponse, cm: "ConversationManager", *args, **kwargs):
    print("recieved and setting contacts")
    cm.contact_index.contacts = {c["contact_id"]: c for c in event.contacts}
    print(cm.contact_index.contacts)


@EventHandler.register(GetBusEventsResponse)
async def _(
    event: GetBusEventsResponse, cm: "ConversationManager", *args, **kwargs
): ...


@EventHandler.register(ConductorHandleStarted)
async def _(event: ConductorResult, cm: "ConversationManager", *args, **kwargs):
    # update the conductor handles state
    cm.notifications_bar.push_notif(
        "Conductor",
        f"Conductor handle started with id {event.handle_id}",
        event.timestamp,
    )
    await cm.run_llm()


@EventHandler.register(ConductorResult)
async def _(event: ConductorResult, cm: "ConversationManager", *args, **kwargs):
    cm.notifications_bar.push_notif(
        "Conductor",
        f"Recieved result for handle_id: {event.handle_id}\nResult: {event.result}",
        event.timestamp,
    )
    cm.conductor_handles.pop(event.handle_id)
    await cm.run_llm()


@EventHandler.register((ConductorPauseActor, ConductorResumeActor))
async def _(
    event: ConductorPauseActor | ConductorResumeActor,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    print("recieved conductor pause/resume event", event.to_dict())
    action = "pause" if isinstance(event, ConductorPauseActor) else "resume"
    reason = getattr(event, "reason", "")
    affected: list[int] = []
    for hid, data in list(cm.conductor_handles.items()):
        # get conductor request handle
        handle = data.get("handle")
        if handle is None:
            continue

        # pause or resume handle
        try:
            if action == "pause" and hasattr(handle, "pause_actor"):
                await handle.pause_actor(reason)
            elif action == "resume" and hasattr(handle, "resume_actor"):
                await handle.resume_actor(reason)
            else:
                print(f"Handle {hid} does not have {action} method")
                continue
            affected.append(int(hid))
        except Exception as e:
            print(f"Failed to {action} handle {hid}: {e}")

    # notify per handle without triggering LLM runs
    for hid in affected:
        try:
            await cm.event_broker.publish(
                "app:conductor:notification",
                ConductorNotification(
                    handle_id=int(hid),
                    response=f"Actor {action}d: {reason}",
                ).to_json(),
            )
        except Exception as e:
            print(
                f"Failed to publish {action} notification for {hid}: {e}",
            )


@EventHandler.register(LogMessageResponse)
async def _(event: LogMessageResponse, cm: "ConversationManager", *args, **kwargs):
    # ToDo: Get this working for email and whatsapp as well
    # Email: Replying to the same thread
    # Whatsapp: Managing different kinds of chat such as groups, etc.
    if event.medium == "phone_call" and cm.call_exchange_id == UNASSIGNED:
        cm.call_exchange_id = event.exchange_id
    if event.medium == "unify_call" and cm.unify_call_exchange_id == UNASSIGNED:
        cm.unify_call_exchange_id = event.exchange_id


@EventHandler.register(PreHireMessage)
async def _(event: PreHireMessage, cm: "ConversationManager", *args, **kwargs):
    asyncio.create_task(managers_utils.log_message(cm, event))
