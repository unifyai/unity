import asyncio
from typing import TYPE_CHECKING, Union

from unity.conversation_manager_2.new_events import *

if TYPE_CHECKING:
    from unity.conversation_manager_2.conversation_manager import ConversationManager
from unity.conversation_manager_2.domains.call_manager import LivekitCallManager


livekit_call_manager = LivekitCallManager()

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
            ...
        print(event)
        f = cls._registry.get(event.__class__)
        if not f:
            raise Exception(f"class: {event.__class__} is not registed!")
        return f(event, cm, *args, **kwargs)
    


CallEvents = Union[PhoneCallRecieved, PhoneCallSent, PhoneCallEnded, UnifyCallReceived]

@EventHandler.register(
        (PhoneCallRecieved, PhoneCallSent, PhoneCallEnded, UnifyCallReceived)
        )
async def _(event: CallEvents, cm: 'ConversationManager', *args, **kwargs):
    if cm.mode in ["phone", "gmeet", "unify_call"]:
        # can't make call
        ...
    else:
        # update state
        cm.notification_bar.push_notification(event.notification())
        cm.contact_index.push_message(event.contact.id, "phone", event.thread_message())

        # start call process
        livekit_call_manager.start_call()

        # make an llm run
        await cm.run_llm()


@EventHandler.register((PhoneUtterance, UnifyCallUtterance))
async def _(event: PhoneCallEnded, cm: 'ConversationManager', *args, **kwargs):
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
    match thread:
        case "sms": thread = "sms"
        case "email": thread = "email"
        case "unify": thread = "unify"
    
    cm.contact_index.push_message(event.contact, thread, message)
    cm.notifications_bar.push_notif(...)
    
    # run llm
    print("got contact", event.contact)
    await cm.run_llm(delay=1)

@EventHandler.register(
    (
        ManagersStartupResponse
    )
)
async def _(event: ManagersStartupResponse, cm: 'ConversationManager', *args, **kwargs):
    if not event.initialized:
        raise Exception("Managers failed to initialize")
    cm.initialized = True

@EventHandler.register((
    StartupEvent
))
async def _(event: StartupEvent, cm: 'ConversationManager', *args, **kwargs):
    print("recieved start up event")
    payload = event.to_dict()["payload"]
    cm.set_details(payload)

@EventHandler.register(GetContactsResponse)
async def _(event: GetContactsResponse, cm: 'ConversationManager', *args, **kwargs):
    print("got contacts")

@EventHandler.register(GetBusEventsResponse)
async def _(event: GetBusEventsResponse, cm: 'ConversationManager', *args, **kwargs):
    ...