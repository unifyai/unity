import asyncio
from time import perf_counter
from typing import TYPE_CHECKING, Union

from unity.conversation_manager_2.new_events import *
from unity.conversation_manager_2.domains import managers_utils

if TYPE_CHECKING:
    from unity.conversation_manager_2.conversation_manager import ConversationManager
class EventHandler:
    _registry = {}

    @classmethod
    def register(cls, event_cls: list[Event] | Event):
        def wrapper(func):
            events_classes = [event_cls] if not isinstance(event_cls, (list, tuple)) else event_cls
            for e in events_classes:
                cls._registry[e] = func
            return func
        return wrapper
    
    @classmethod
    def handle_event(cls, event: Event, cm: "ConversationManager", *args, **kwargs):
        # maybe add the event bus logging thing here
        if event.__class__.loggable:
            asyncio.create_task(managers_utils.publish_bus_events(event))
        print(event)
        f = cls._registry.get(event.__class__)
        if not f:
            # do nothing basically (?)
            return asyncio.sleep(0)
        return f(event, cm, *args, **kwargs)

CallEvents = Union[PhoneCallRecieved, PhoneCallSent, UnifyCallReceived]

@EventHandler.register(
        (PhoneCallRecieved, PhoneCallSent, UnifyCallReceived)
        )
async def _(event: CallEvents, cm: 'ConversationManager', *args, **kwargs):
    if cm.mode in ["phone", "gmeet", "unify_call"]:
        # can't make call
        # TODO: we should handle this somehow tbh
        ...
    else:
        # update state
        message_content = None
        notif_content = None
        match event:
            case PhoneCallRecieved():
                ...
            case PhoneCallSent():
                ...
            case UnifyCallReceived():
                ...

        cm.notifications_bar.push_notification(...)
        cm.contact_index.push_message(event.contact["contact_id"], "phone", )

        # start call process
        cm.call_manager.start_call()

        # make an llm run
        await cm.run_llm()


@EventHandler.register((PhoneUtterance, UnifyCallUtterance))
async def _(event: PhoneCallEnded, cm: 'ConversationManager', *args, **kwargs):
    # publish transcript
    asyncio.create_task(managers_utils.log_message(cm, event))
    ...

@EventHandler.register((PhoneCallEnded, UnifyCallEnded))
async def _(event: PhoneCallEnded | UnifyCallEnded, cm: 'ConversationManager', *args, **kwargs):
    ...

@EventHandler.register((
                ConductorResponse,
                ConductorHandleResponse,
                ConductorResult,
                ConductorClarificationRequest,
            ))
async def _(event, cm: 'ConversationManager', *args, **kwargs):
    # just run llm here
    ...

@EventHandler.register((
                    SMSSent,
                    SMSRecieved,
                    EmailSent,
                    EmailRecieved,
                    UnifyMessageSent,
                    UnifyMessageRecieved,
                ))
async def _(event, cm: 'ConversationManager', *args, **kwargs):
    # update state
    thread = None
    message_content = None
    subject = None
    body = None
    notif_content = None

    contact = cm.contact_index.get_contact(event.contact["contact_id"])

    match event:
        case SMSSent():
            thread = "sms"
            message_content = event.content
            notif_content = f"SMS sent to {contact.full_name}"
        case SMSRecieved():
            thread = "sms"
            message_content = event.content
            notif_content = f"SMS recieved from {contact.full_name}"
        case EmailSent():
            thread = "email"
            subject = event.subject
            body = event.body
            notif_content = f"Email sent to {contact.full_name}"
        case EmailRecieved():
            thread = "email"
            subject = event.subject
            body = event.body
            notif_content = f"Email recieved from {contact.full_name}"
        case UnifyMessageSent():
            thread = "unify"
            message_content = event.content
            notif_content = f"Unify message sent to {contact.full_name}"
        case UnifyMessageRecieved():
            thread = "unify"
            message_content = event.content
            notif_content = f"Unify message from {contact.full_name}"
            
    
    message_content = event.content
    cm.contact_index.push_message(event.contact, thread, message_content=message_content, subject=subject, body=body, timestamp=event.timestamp)
    cm.notifications_bar.push_notif("comms", notif_content, event.timestamp)
    
    # run llm (TODO: add cancel running if not on a call)
    await cm.run_llm(delay=2)

# TODO: put all managers in the cm and move start up logic from managers worker to here

@EventHandler.register((
    StartupEvent
))
async def _(event: StartupEvent, cm: 'ConversationManager', *args, **kwargs):
    print("recieved start up event")
    payload = event.to_dict()["payload"]
    cm.set_details(payload)

@EventHandler.register(GetContactsResponse)
async def _(event: GetContactsResponse, cm: 'ConversationManager', *args, **kwargs):
    print("recieved and setting contacts")
    cm.contact_index.contacts = {c["contact_id"]:c for c in event.contacts}
    print(cm.contact_index.contacts)

@EventHandler.register(GetBusEventsResponse)
async def _(event: GetBusEventsResponse, cm: 'ConversationManager', *args, **kwargs):
    ...


@EventHandler.register(ConductorResult)
async def _(event: ConductorResult, cm: 'ConversationManager', *args, **kwargs):
    # update the conductor handles state
    ...