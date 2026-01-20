import asyncio
from typing import TYPE_CHECKING, Union

from unity.contact_manager.types.contact import UNASSIGNED
from unity.conversation_manager import debug_logger
from unity.conversation_manager.domains.contact_index import Contact
from unity.conversation_manager.events import *
from unity.conversation_manager.domains import managers_utils

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager


def _event_type_to_log_key(event_cls) -> str:
    """Convert an event class name to a log key for icon lookup."""
    name = event_cls.__name__
    # Convert CamelCase to snake_case
    import re

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


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
        # Log the event using the session logger
        event_key = _event_type_to_log_key(event.__class__)
        if hasattr(cm, "_session_logger"):
            cm._session_logger.info(event_key, f"Event: {event.__class__.__name__}")

        if event.__class__.loggable:
            asyncio.create_task(
                managers_utils.queue_operation(
                    managers_utils.publish_bus_events,
                    event,
                ),
            )

        f = cls._registry.get(event.__class__)
        if not f:
            # do nothing basically (?)
            return asyncio.sleep(0)
        return f(event, cm, *args, **kwargs)


@EventHandler.register(Ping)
async def _(event: Ping, cm: "ConversationManager", *args, **kwargs):
    log_str = "Ping received - keeping conversation manager alive"
    print(log_str)  # need console logging of ping to detect idle containers
    cm._session_logger.debug("ping", log_str)


CallEvents = Union[
    PhoneCallReceived,
    PhoneCallSent,
    UnifyMeetReceived,
    PhoneCallAnswered,
]


@EventHandler.register(
    (PhoneCallReceived, PhoneCallSent, UnifyMeetReceived, PhoneCallAnswered),
)
async def _(event: CallEvents, cm: "ConversationManager", *args, **kwargs):
    if cm.mode in ["call", "unify_meet"]:
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
        if isinstance(event, UnifyMeetReceived):
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
            case UnifyMeetReceived() as e:
                cm.call_manager.start_unify_meet(
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
            "voice",
            message_content=message_content,
            role=(
                "user"
                if "received" in event.__class__.__name__.lower()
                else "assistant"
            ),
            timestamp=event.timestamp,
        )


@EventHandler.register((PhoneCallStarted, UnifyMeetStarted))
async def _(
    event: PhoneCallStarted | UnifyMeetStarted,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    if isinstance(event, PhoneCallStarted):
        cm.mode = "call"
        phone_number = event.contact["phone_number"]
        contact = cm.contact_index.get_contact(phone_number=phone_number)
    else:
        cm.mode = "unify_meet"
        contact = cm.contact_index.get_contact(contact_id=1)

    cm.call_manager.call_contact = contact
    cm.notifications_bar.push_notif(
        "Comms",
        f"Phone Call started with {contact['first_name']}",
        timestamp=event.timestamp,
    )
    cm.contact_index.push_message(
        contact,
        "voice",
        "<Call Started>",
        timestamp=event.timestamp,
    )
    cm.contact_index.active_conversations[contact["contact_id"]].on_call = True
    await cm.request_llm_run(delay=0)


@EventHandler.register(
    (
        InboundPhoneUtterance,
        InboundUnifyMeetUtterance,
        OutboundPhoneUtterance,
        OutboundUnifyMeetUtterance,
    ),
)
async def _(event: Event, cm: "ConversationManager", *args, **kwargs):
    # publish transcript
    cm._session_logger.debug(
        "event",
        f"Publishing transcript: {event.__class__.__name__}",
    )
    await managers_utils.queue_operation(managers_utils.log_message, cm, event)

    # push message to contact index
    contact_id = event.contact["contact_id"]
    contact = cm.contact_index.get_contact(contact_id=contact_id)
    role = "user" if event.__class__.__name__.startswith("Inbound") else "assistant"
    cm.contact_index.push_message(contact, "voice", event.content, role=role)

    # cancel proactive speech on user input
    if role == "user":
        await cm.cancel_proactive_speech()
        # Trigger Main CM Brain to process user input and potentially send guidance
        await cm.interject_or_run(event.content)


@EventHandler.register(CallGuidance)
async def _(
    event: CallGuidance,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm._session_logger.info(
        "call_guidance",
        f"Received guidance: {event.content[:50]}...",
    )
    contact_id = event.contact["contact_id"]
    contact = cm.contact_index.get_contact(contact_id=contact_id)
    cm.contact_index.push_message(contact, "voice", event.content, role="Guidance")


@EventHandler.register((PhoneCallEnded, UnifyMeetEnded))
async def _(
    event: PhoneCallEnded | UnifyMeetEnded,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm.mode = "text"
    cm.call_manager.call_contact = None
    if isinstance(event, PhoneCallEnded):
        cm.call_manager.conference_name = None
    if isinstance(event, UnifyMeetEnded):
        contact = cm.contact_index.get_contact(contact_id=1)
    else:
        contact = cm.contact_index.get_contact(
            phone_number=event.contact["phone_number"],
        )

    # Guard against missing active conversation (can happen if call ended
    # before PhoneCallStarted, or after container restart)
    contact_id = contact["contact_id"]
    if contact_id in cm.contact_index.active_conversations:
        cm.contact_index.active_conversations[contact_id].on_call = False

    await cm.call_manager.cleanup_call_proc()
    await cm.cancel_proactive_speech()
    await cm.request_llm_run(delay=0, cancel_running=True)


@EventHandler.register(
    (
        ActorResponse,
        ActorHandleResponse,
        ActorResult,
        ActorClarificationRequest,
    ),
)
async def _(event, cm: "ConversationManager", *args, **kwargs):
    # Track clarification requests in the task's handle_actions and notify
    if isinstance(event, ActorClarificationRequest):
        if event.handle_id in cm.active_tasks:
            cm.active_tasks[event.handle_id]["handle_actions"].append(
                {
                    "action_name": "clarification_request",
                    "query": event.query,
                    "call_id": event.call_id,
                },
            )
            # Add notification about the clarification request
            task_query = cm.active_tasks[event.handle_id].get("query", "")
            short_desc = task_query[:30] + "..." if len(task_query) > 30 else task_query
            cm.notifications_bar.push_notif(
                "Task",
                f"Task '{short_desc}' needs clarification: {event.query}",
                event.timestamp,
            )
            # Trigger LLM run so assistant can see and respond to the clarification
            await cm.request_llm_run()
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
    attachments = None
    notif_content = None

    # Get contact info, with fallback chain:
    # 1. ContactManager (via contact_index) - source of truth with auto-syncing cache
    # 2. event.contact - ultimate fallback (CommsManager already resolved it)
    contact = cm.contact_index.get_contact(event.contact["contact_id"])
    if contact is None:
        # Use event.contact as fallback - it already has the contact info from CommsManager
        contact = event.contact

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
            attachments = event.attachments
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
            attachments = event.attachments
            notif_content = f"Unify message from {contact['first_name']}"
            role = "user"

    cm.contact_index.push_message(
        contact,
        thread,
        message_content=message_content,
        subject=subject,
        body=body,
        email_id=email_id,
        attachments=attachments,
        timestamp=event.timestamp,
        role=role,
    )
    cm.notifications_bar.push_notif("comms", notif_content, event.timestamp)

    # Cancel proactive speech on message received
    if role == "user":
        await cm.cancel_proactive_speech()

    await cm.request_llm_run(delay=2)


# TODO: put all managers in the cm and move start up logic from managers worker to here


@EventHandler.register((StartupEvent))
async def _(event: StartupEvent, cm: "ConversationManager", *args, **kwargs):
    cm._session_logger.info("startup", "Received startup event")
    payload = event.to_dict()["payload"]
    cm.set_details(payload)
    cm.call_manager.set_config(cm.get_call_config())
    # Update the running record (created by adapter) with job_name and liveview_url
    asyncio.create_task(
        asyncio.to_thread(
            debug_logger.log_job_startup,
            job_name=cm.job_name,
            user_id=cm.user_id,
            assistant_id=cm.assistant_id,
        ),
    )

    # Start initialization and operations listener
    asyncio.create_task(managers_utils.init_conv_manager(cm))
    asyncio.create_task(managers_utils.listen_to_operations(cm))


@EventHandler.register(GetContactsResponse)
async def _(event: GetContactsResponse, cm: "ConversationManager", *args, **kwargs):
    cm._session_logger.info("state_update", f"Setting {len(event.contacts)} contacts")
    cm.contact_index.set_contacts(event.contacts)
    # print(cm.contact_index.contacts)


@EventHandler.register(GetChatHistory)
async def _(event: GetChatHistory, cm: "ConversationManager", *args, **kwargs):
    cm._session_logger.debug(
        "state_update",
        f"Received chat history ({len(event.chat_history)} messages)",
    )
    cm.chat_history = event.chat_history + cm.chat_history


@EventHandler.register(ActorHandleStarted)
async def _(event: ActorHandleStarted, cm: "ConversationManager", *args, **kwargs):
    # Notify that a new task has started
    cm.notifications_bar.push_notif(
        "Task",
        f"Task started: {event.query[:50]}{'...' if len(event.query) > 50 else ''}",
        event.timestamp,
    )
    await cm.request_llm_run()


@EventHandler.register(NotificationInjectedEvent)
async def _(
    event: NotificationInjectedEvent,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm._session_logger.info(
        "notification_injected",
        f"Notification: {event.content[:50]}...",
    )

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
    await cm.request_llm_run(delay=0, cancel_running=True)


@EventHandler.register(NotificationUnpinnedEvent)
async def _(
    event: NotificationUnpinnedEvent,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm._session_logger.info(
        "notification_unpinned",
        f"Unpinned interjection: {event.interjection_id}",
    )

    # Remove from notification bar
    cm.notifications_bar.remove_notif(event.interjection_id)


@EventHandler.register(ActorResult)
async def _(event: ActorResult, cm: "ConversationManager", *args, **kwargs):
    # Get task description for notification
    task_data = cm.active_tasks.get(event.handle_id, {})
    task_query = task_data.get("query", f"Task {event.handle_id}")
    short_desc = task_query[:30] + "..." if len(task_query) > 30 else task_query

    cm.notifications_bar.push_notif(
        "Task",
        f"Task completed: {short_desc}\nResult: {event.result}",
        event.timestamp,
    )
    cm.active_tasks.pop(event.handle_id, None)
    await cm.request_llm_run()


@EventHandler.register((ActorPause, ActorResume))
async def _(
    event: ActorPause | ActorResume,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    action = "pause" if isinstance(event, ActorPause) else "resume"
    cm._session_logger.info("actor_request", f"Received task {action} event")
    reason = getattr(event, "reason", "")
    affected: list[int] = []
    for hid, data in list(cm.active_tasks.items()):
        # get task handle
        handle = data.get("handle")
        if handle is None:
            continue

        # pause or resume handle
        try:
            # Standard steerable surface: pause/resume are preferred; no actor-specific convenience methods.
            if action == "pause":
                pause_r = handle.pause()
                if asyncio.iscoroutine(pause_r) or isinstance(pause_r, asyncio.Future):
                    await pause_r
            else:
                resume_r = handle.resume()
                if asyncio.iscoroutine(resume_r) or isinstance(
                    resume_r,
                    asyncio.Future,
                ):
                    await resume_r
            affected.append(int(hid))
        except Exception as e:
            cm._session_logger.error(
                "actor_request",
                f"Failed to {action} task {hid}: {e}",
            )

    # notify per handle without triggering LLM runs
    for hid in affected:
        try:
            await cm.event_broker.publish(
                "app:actor:notification",
                ActorNotification(
                    handle_id=int(hid),
                    response=f"Task {action}d: {reason}",
                ).to_json(),
            )
        except Exception as e:
            cm._session_logger.error(
                "actor_request",
                f"Failed to publish {action} notification for task {hid}: {e}",
            )


@EventHandler.register(SyncContacts)
async def _(
    event: SyncContacts,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm._session_logger.info(
        "state_update",
        f"SyncContacts: {event.reason or 'no reason'}",
    )

    async def _sync_contacts():
        try:
            await asyncio.to_thread(cm.contact_manager._sync_required_contacts)
            cm._session_logger.info("state_update", "Contacts synced successfully")
        except Exception as e:
            cm._session_logger.error("state_update", f"Error syncing contacts: {e}")
        cm.notifications_bar.push_notif(
            "System",
            f"Contacts synced: {event.reason or 'manual sync'}",
            event.timestamp,
        )

    await managers_utils.queue_operation(_sync_contacts)


@EventHandler.register(LogMessageResponse)
async def _(event: LogMessageResponse, cm: "ConversationManager", *args, **kwargs):
    # ToDo: Get this working for email as well (replying to the same thread)
    if event.medium == "phone_call" and cm.call_manager.call_exchange_id == UNASSIGNED:
        cm.call_manager.call_exchange_id = event.exchange_id
    if (
        event.medium == "unify_meet"
        and cm.call_manager.unify_meet_exchange_id == UNASSIGNED
    ):
        cm.call_manager.unify_meet_exchange_id = event.exchange_id


@EventHandler.register(PreHireMessage)
async def _(event: PreHireMessage, cm: "ConversationManager", *args, **kwargs):
    await managers_utils.queue_operation(managers_utils.log_message, cm, event)


@EventHandler.register(SummarizeContext)
async def _(event: SummarizeContext, cm: "ConversationManager", *args, **kwargs):
    # Skip if MemoryManager is disabled
    if cm.memory_manager is None:
        cm._session_logger.debug(
            "summarize",
            "SummarizeContext skipped (MemoryManager disabled)",
        )
        cm.is_summarizing = False
        cm.chat_history = []
        return

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
            cm._session_logger.debug(
                "state_update",
                f"Updated {len(updated_active_contacts)} contacts",
            )
            cm.contact_index.active_conversations = updated_active_contacts
            cm.is_summarizing = False
            cm.chat_history = []
            cm._session_logger.info("summarize", "Contact rolling summary updated")
        except Exception as e:
            cm._session_logger.error(
                "summarize",
                f"Error updating rolling summary: {e}",
            )

    asyncio.create_task(summarize_task())


@EventHandler.register(DirectMessageEvent)
async def _(event: DirectMessageEvent, cm: "ConversationManager", *args, **kwargs):
    cm._session_logger.info(
        "direct_message",
        f"Direct message: {event.content[:50]}...",
    )

    # Send to Voice Agent via call_guidance channel
    if cm.mode in ["call", "unify_meet"]:
        await cm.event_broker.publish(
            "app:call:call_guidance",
            json.dumps({"content": event.content}),
        )

    # Record in contact_index for transcript access
    contact = cm.get_active_contact()
    cm.contact_index.push_message(
        contact,
        "voice",
        message_content=event.content,
        role="assistant",
        timestamp=event.timestamp,
    )
