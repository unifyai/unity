import asyncio
from typing import TYPE_CHECKING, Union

from unity.contact_manager.types.contact import UNASSIGNED
from unity.conversation_manager import debug_logger
from unity.conversation_manager.domains.contact_index import Contact
from unity.conversation_manager.events import *
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
        print(f"Received EVENT: {event}")
        if event.__class__.loggable:
            asyncio.create_task(
                managers_utils.queue_operation(
                    managers_utils.publish_bus_events,
                    event,
                ),
            )
        print(event)
        f = cls._registry.get(event.__class__)
        if not f:
            # do nothing basically (?)
            return asyncio.sleep(0)
        return f(event, cm, *args, **kwargs)


@EventHandler.register(Ping)
async def _(event: Ping, cm: "ConversationManager", *args, **kwargs):
    print("ping received - keeping conversation manager alive")


CallEvents = Union[
    PhoneCallReceived,
    PhoneCallSent,
    UnifyCallReceived,
    PhoneCallAnswered,
]


@EventHandler.register(
    (PhoneCallReceived, PhoneCallSent, UnifyCallReceived, PhoneCallAnswered),
)
async def _(event: CallEvents, cm: "ConversationManager", *args, **kwargs):
    if cm.mode in ["call", "gmeet", "unify_call"]:
        # can't make call
        # TODO: we should handle this somehow tbh
        # for now do nothing, but we can think of adding a notification of an attempted call

        # if an outbound call has been answered, we should send a notification to the call script
        if isinstance(event, PhoneCallAnswered):
            await cm.event_broker.publish(
                "app:call:status",
                json.dumps({"type": "call_answered"}),
            )
    else:
        # update state
        message_content = None
        notif_content = None
        boss = cm.contact_index.get_contact(contact_id=1)
        if isinstance(event, UnifyCallReceived):
            contact = boss
        else:
            contact = cm.contact_index.get_contact(
                phone_number=event.contact["phone_number"],
            )
        match event:
            case PhoneCallReceived() as e:
                cm.call_manager.conference_name = e.conference_name
                cm.call_manager.start_call(contact, boss)
                message_content = "<Recvieving Call...>"
                notif_content = f"Call received from {contact['first_name']}"
            case PhoneCallSent() as e:
                cm.call_manager.start_call(contact, boss, outbound=True)
                message_content = "<Sending Call...>"
                notif_content = f"Call sent to {contact['first_name']}"
            case UnifyCallReceived() as e:
                cm.call_manager.start_unify_call(
                    contact,
                    boss,
                    e.agent_name,
                    e.room_name,
                )
                message_content = "<Recieving Call...>"
                notif_content = f"Call received from {contact['first_name']}"

        cm.notifications_bar.push_notif("Comms", notif_content, event.timestamp)
        cm.contact_index.push_message(
            contact,
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
        phone_number = event.contact["phone_number"]
        contact = cm.contact_index.get_contact(phone_number=phone_number)
    else:
        cm.mode = "unify_call"
        contact = cm.contact_index.get_contact(contact_id=1)

    cm.call_manager.call_contact = contact
    cm.notifications_bar.push_notif(
        "Comms",
        f"Phone Call started with {contact['first_name']}",
        timestamp=event.timestamp,
    )
    cm.contact_index.push_message(
        contact,
        "phone",
        "<Call Started>",
        timestamp=event.timestamp,
    )
    cm.contact_index.active_conversations[contact["contact_id"]].on_call = True
    await cm.run_llm(delay=0)


@EventHandler.register(
    (
        InboundPhoneUtterance,
        InboundUnifyCallUtterance,
        OutboundPhoneUtterance,
        OutboundUnifyCallUtterance,
    ),
)
async def _(event: Event, cm: "ConversationManager", *args, **kwargs):
    # publish transcript
    print("publishing transcript", event)
    await managers_utils.queue_operation(managers_utils.log_message, cm, event)

    # push message to contact index
    contact_id = event.contact["contact_id"]
    contact = cm.contact_index.get_contact(contact_id=contact_id)
    role = (
        "user" if "assistant" not in event.__class__.__name__.lower() else "assistant"
    )
    cm.contact_index.push_message(contact, "phone", event.content, role=role)

    # cancel proactive speech
    if role == "user":
        await cm.cancel_proactive_speech()

    # trigger LLM runs for user events in non-realtime mode
    if not cm.call_manager.realtime and role == "user":
        # start filler only in non-realtime
        await cm.cancel_filler()
        asyncio.create_task(cm.run_filler_once())

        await cm.interject_or_run(event.content)

    # trigger LLM runs for assistant events in realtime mode
    elif cm.call_manager.realtime and role == "assistant":
        await cm.interject_or_run(event.content)


@EventHandler.register(AssistantRealtimeGuidance)
async def _(
    event: AssistantRealtimeGuidance,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    print("received realtime guidance", event)
    contact_id = event.contact["contact_id"]
    contact = cm.contact_index.get_contact(contact_id=contact_id)
    cm.contact_index.push_message(contact, "phone", event.content, role="Guidance")


@EventHandler.register((PhoneCallEnded, UnifyCallEnded))
async def _(
    event: PhoneCallEnded | UnifyCallEnded,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm.mode = "text"
    cm.call_manager.call_contact = None
    if isinstance(event, PhoneCallEnded):
        cm.call_manager.conference_name = None
    if isinstance(event, UnifyCallEnded):
        contact = cm.contact_index.get_contact(contact_id=1)
    else:
        contact = cm.contact_index.get_contact(
            phone_number=event.contact["phone_number"],
        )
    cm.contact_index.active_conversations[contact["contact_id"]].on_call = False
    cm.call_manager.cleanup_call_proc()
    await cm.cancel_filler()
    await cm.cancel_proactive_speech()
    await cm.run_llm(delay=0, cancel_running=True)


@EventHandler.register(
    (
        ConductorResponse,
        ConductorHandleResponse,
        ConductorResult,
        ConductorClarificationRequest,
    ),
)
async def _(event, cm: "ConversationManager", *args, **kwargs):
    # just run llm here
    if isinstance(event, ConductorClarificationRequest):
        cm.conductor_handles[event.handle_id]["handle_actions"].append(
            {
                "action_name": "conductor_handle_clarification_request",
                "query": event.query,
                "call_id": event.call_id,
            },
        )
    else:
        ...


@EventHandler.register(
    (
        SMSSent,
        SMSReceived,
        EmailSent,
        EmailReceived,
        UnifyMessageSent,
        UnifyMessageReceived,
    ),
)
async def _(event, cm: "ConversationManager", *args, **kwargs):
    await managers_utils.queue_operation(managers_utils.log_message, cm, event)

    # update state
    thread = None
    message_content = None
    subject = None
    body = None
    email_id = None
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
            email_id = event.email_id_replied_to
            notif_content = f"Email sent to {contact['first_name']}"
            role = "assistant"
        case EmailReceived():
            thread = "email"
            subject = event.subject
            body = event.body
            email_id = event.email_id
            notif_content = f"Email Received from {contact['first_name']}"
            role = "user"
        case UnifyMessageSent():
            thread = "unify_message"
            message_content = event.content
            notif_content = f"Unify message sent to {contact['first_name']}"
            role = "assistant"
        case UnifyMessageReceived():
            thread = "unify_message"
            message_content = event.content
            notif_content = f"Unify message from {contact['first_name']}"
            role = "user"

    cm.contact_index.push_message(
        contact,
        thread,
        message_content=message_content,
        subject=subject,
        body=body,
        email_id=email_id,
        timestamp=event.timestamp,
        role=role,
    )
    cm.notifications_bar.push_notif("comms", notif_content, event.timestamp)

    # Cancel proactive speech on message received
    if role == "user":
        await cm.cancel_proactive_speech()

    await cm.run_llm(delay=2)


# TODO: put all managers in the cm and move start up logic from managers worker to here


@EventHandler.register((StartupEvent))
async def _(event: StartupEvent, cm: "ConversationManager", *args, **kwargs):
    print("Received start up event")
    payload = event.to_dict()["payload"]
    cm.set_details(payload)
    cm.call_manager.set_config(cm.get_call_config())
    kwargs = {
        "timestamp": payload["timestamp"],
        "medium": payload["medium"],
        **cm.get_details(),
    }
    asyncio.create_task(asyncio.to_thread(debug_logger.log_job_startup, **kwargs))

    # Start initialization and operations listener
    asyncio.create_task(managers_utils.init_conv_manager(cm))
    asyncio.create_task(managers_utils.listen_to_operations(cm))


@EventHandler.register(GetContactsResponse)
async def _(event: GetContactsResponse, cm: "ConversationManager", *args, **kwargs):
    print("Received and setting contacts")
    cm.contact_index.set_contacts(event.contacts)
    # print(cm.contact_index.contacts)


@EventHandler.register(StoreChatHistory)
async def _(
    event: StoreChatHistory,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    print("Received store chat history")


@EventHandler.register(GetChatHistory)
async def _(event: GetChatHistory, cm: "ConversationManager", *args, **kwargs):
    print("Received get chat history")
    cm.chat_history = event.chat_history + cm.chat_history


@EventHandler.register(ConductorHandleStarted)
async def _(event: ConductorResult, cm: "ConversationManager", *args, **kwargs):
    # update the conductor handles state
    cm.notifications_bar.push_notif(
        "Conductor",
        f"Conductor handle started with id {event.handle_id}",
        event.timestamp,
    )
    await cm.run_llm()


@EventHandler.register(NotificationInjectedEvent)
async def _(
    event: NotificationInjectedEvent,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    print(f"Received NotificationInjectedEvent: {event.content}")

    # Push to notification bar
    cm.notifications_bar.push_notif(
        event.source,
        event.content,
        event.timestamp,
        pinned=event.pinned,
        id=event.interjection_id,
    )

    # Cancel proactive speech because we are injecting something
    await cm.cancel_proactive_speech()

    # Trigger LLM to react to the notification
    await cm.run_llm(delay=0, cancel_running=True)


@EventHandler.register(NotificationUnpinnedEvent)
async def _(
    event: NotificationUnpinnedEvent,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    print(f"Received NotificationUnpinnedEvent: {event.interjection_id}")

    # Remove from notification bar
    cm.notifications_bar.remove_notif(event.interjection_id)


@EventHandler.register(ConductorResult)
async def _(event: ConductorResult, cm: "ConversationManager", *args, **kwargs):
    cm.notifications_bar.push_notif(
        "Conductor",
        f"Received result for handle_id: {event.handle_id}\nResult: {event.result}",
        event.timestamp,
    )
    cm.conductor_handles.pop(event.handle_id, None)
    await cm.run_llm()


@EventHandler.register((ConductorPauseActor, ConductorResumeActor))
async def _(
    event: ConductorPauseActor | ConductorResumeActor,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    print("Received conductor pause/resume event", event.to_dict())
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
    if event.medium == "phone_call" and cm.call_manager.call_exchange_id == UNASSIGNED:
        cm.call_manager.call_exchange_id = event.exchange_id
    if (
        event.medium == "unify_call"
        and cm.call_manager.unify_call_exchange_id == UNASSIGNED
    ):
        cm.call_manager.unify_call_exchange_id = event.exchange_id


@EventHandler.register(PreHireMessage)
async def _(event: PreHireMessage, cm: "ConversationManager", *args, **kwargs):
    await managers_utils.queue_operation(managers_utils.log_message, cm, event)


@EventHandler.register(SummarizeContext)
async def _(event: SummarizeContext, cm: "ConversationManager", *args, **kwargs):
    async def summarize_task():
        res = [
            (
                cid,
                cm.prompt_renderer.render_contact(
                    c,
                    max_messages=25,
                    last_snapshot=cm.last_snapshot,
                ),
            )
            for cid, c in cm.contact_index.active_conversations.items()
        ]
        tasks = [
            asyncio.create_task(
                cm.memory_manager.update_contact_rolling_summary(t, contact_id=cid),
            )
            for cid, t in res
        ]
        try:
            await asyncio.gather(*tasks)
            updated_active_contacts = cm.contact_manager.get_contact_info(
                contact_id=[cid for cid in cm.contact_index.active_conversations],
            )
            updated_active_contacts = {
                cid: Contact(**{**c.model_dump(), **uc, "threads": c.threads})
                for (cid, c), uc in zip(
                    cm.contact_index.active_conversations.items(),
                    updated_active_contacts.values(),
                )
            }
            print("updated contact", updated_active_contacts)
            cm.contact_index.active_conversations = updated_active_contacts
            cm.is_summarizing = False
            cm.chat_history = []
            print("[ManagersWorker] Contact rolling summary updated")
        except Exception as e:
            print(f"[ManagersWorker] Error updating contact rolling summary: {e}")

    asyncio.create_task(summarize_task())


@EventHandler.register(DirectMessageEvent)
async def _(event: DirectMessageEvent, cm: "ConversationManager", *args, **kwargs):
    print(f"Received DirectMessageEvent: {event.content}")

    # Speak to voice layer using appropriate channel
    if cm.mode in ["call", "unify_call", "gmeet"]:
        if cm.call_manager.realtime:
            # Realtime API: Send as notification
            await cm.event_broker.publish(
                "app:call:call_notifs",
                json.dumps({"content": event.content}),
            )
        else:
            # STT-TTS pipeline: Send to response_gen channel
            channel = f"app:{cm.mode}:response_gen"
            await cm.event_broker.publish(channel, json.dumps({"type": "start_gen"}))
            await cm.event_broker.publish(
                channel,
                json.dumps({"type": "gen_chunk", "chunk": event.content}),
            )
            await cm.event_broker.publish(channel, json.dumps({"type": "end_gen"}))

    # Record in contact_index for transcript access
    contact = cm.call_manager.call_contact or cm.contact_index.get_contact(contact_id=1)
    cm.contact_index.push_message(
        contact,
        "phone" if cm.mode == "call" else "unify_call",
        message_content=event.content,
        role="assistant",
        timestamp=event.timestamp,
    )
