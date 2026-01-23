import asyncio
from typing import TYPE_CHECKING, Union

from unity.contact_manager.types.contact import UNASSIGNED
from unity.conversation_manager import debug_logger
from unity.conversation_manager.events import *
from unity.conversation_manager.domains import managers_utils
from unity.conversation_manager.types import Medium, Mode

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager


def _event_type_to_log_key(event_cls) -> str:
    """Convert an event class name to a log key for icon lookup."""
    name = event_cls.__name__
    import re

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _get_sender_name(contact: dict | None, fallback: str = "Unknown") -> str:
    """Get display name from contact dict."""
    if not contact:
        return fallback
    first = contact.get("first_name") or ""
    last = contact.get("surname") or ""
    name = f"{first} {last}".strip()
    return name or fallback


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
            return asyncio.sleep(0)
        return f(event, cm, *args, **kwargs)


@EventHandler.register(Ping)
async def _(event: Ping, cm: "ConversationManager", *args, **kwargs):
    log_str = "Ping received - keeping conversation manager alive"
    print(log_str)
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
    if cm.mode.is_voice:
        if isinstance(event, PhoneCallAnswered):
            await cm.event_broker.publish(
                "app:call:status",
                json.dumps({"type": "call_answered"}),
            )
    else:
        message_content = None
        notif_content = None
        boss = cm.contact_index.get_contact(contact_id=1)
        print("PhoneCallReceived event:", event)
        print("Fallback contacts:", cm.contact_index._fallback_contacts)
        print("Boss contact:", boss)
        if isinstance(event, UnifyMeetReceived):
            contact = boss
        else:
            contact = cm.contact_index.get_contact(
                phone_number=event.contact["phone_number"],
            )
            if contact is None:
                contact = event.contact

        contact_id = (
            contact.get("contact_id") if contact else event.contact.get("contact_id")
        )
        sender_name = _get_sender_name(contact)

        match event:
            case PhoneCallReceived() as e:
                cm.call_manager.conference_name = e.conference_name
                cm.call_manager.start_call(contact, boss)
                message_content = "<Recvieving Call...>"
                notif_content = f"Call received from {sender_name}"
            case PhoneCallSent() as e:
                cm.call_manager.start_call(contact, boss, outbound=True)
                message_content = "<Sending Call...>"
                notif_content = f"Call sent to {sender_name}"
            case UnifyMeetReceived() as e:
                cm.call_manager.start_unify_meet(
                    contact,
                    boss,
                    e.livekit_agent_name,
                    e.room_name,
                )
                message_content = "<Recieving Call...>"
                notif_content = f"Call received from {sender_name}"

        cm.notifications_bar.push_notif("Comms", notif_content, event.timestamp)
        role = "user" if "received" in event.__class__.__name__.lower() else "assistant"
        medium = (
            Medium.UNIFY_MEET
            if isinstance(event, UnifyMeetReceived)
            else Medium.PHONE_CALL
        )
        cm.contact_index.push_message(
            contact_id=contact_id,
            sender_name=sender_name,
            thread_name=medium,
            message_content=message_content,
            role=role,
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
        cm.mode = Mode.CALL
        phone_number = event.contact["phone_number"]
        contact = cm.contact_index.get_contact(phone_number=phone_number)
    else:
        cm.mode = Mode.MEET
        contact_id = event.contact.get("contact_id")
        contact = cm.contact_index.get_contact(contact_id=contact_id)

    if contact is None:
        contact = event.contact

    contact_id = contact.get("contact_id") if contact else 1
    sender_name = _get_sender_name(contact)

    cm.call_manager.call_contact = contact
    cm.notifications_bar.push_notif(
        "Comms",
        f"Phone Call started with {sender_name}",
        timestamp=event.timestamp,
    )
    medium = (
        Medium.PHONE_CALL if isinstance(event, PhoneCallStarted) else Medium.UNIFY_MEET
    )
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=medium,
        message_content="<Call Started>",
        timestamp=event.timestamp,
    )
    conv_state = cm.contact_index.get_or_create_conversation(contact_id)
    conv_state.on_call = True
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
    cm._session_logger.debug(
        "event",
        f"Publishing transcript: {event.__class__.__name__}",
    )
    await managers_utils.queue_operation(managers_utils.log_message, cm, event)

    contact_id = event.contact["contact_id"]
    contact = cm.contact_index.get_contact(contact_id=contact_id)
    if contact is None:
        contact = event.contact
    sender_name = _get_sender_name(contact)
    role = "user" if event.__class__.__name__.startswith("Inbound") else "assistant"

    is_unify_meet = isinstance(
        event,
        (InboundUnifyMeetUtterance, OutboundUnifyMeetUtterance),
    )
    medium = Medium.UNIFY_MEET if is_unify_meet else Medium.PHONE_CALL
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=medium,
        message_content=event.content,
        role=role,
    )

    if role == "user":
        await cm.cancel_proactive_speech()
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
    if contact is None:
        contact = event.contact
    sender_name = _get_sender_name(contact)

    medium = Medium.UNIFY_MEET if cm.mode == Mode.MEET else Medium.PHONE_CALL
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=medium,
        message_content=event.content,
        role="Guidance",
    )


@EventHandler.register((PhoneCallEnded, UnifyMeetEnded))
async def _(
    event: PhoneCallEnded | UnifyMeetEnded,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    cm.mode = Mode.TEXT
    cm.call_manager.call_contact = None
    if isinstance(event, PhoneCallEnded):
        cm.call_manager.conference_name = None
    if isinstance(event, UnifyMeetEnded):
        contact = cm.contact_index.get_contact(contact_id=1)
    else:
        contact = cm.contact_index.get_contact(
            phone_number=event.contact["phone_number"],
        )
        if contact is None:
            contact = event.contact

    contact_id = (
        contact.get("contact_id") if contact else event.contact.get("contact_id")
    )
    conv_state = cm.contact_index.get_conversation_state(contact_id)
    if conv_state:
        conv_state.on_call = False

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
    if isinstance(event, ActorClarificationRequest):
        if event.handle_id in cm.in_flight_actions:
            cm.in_flight_actions[event.handle_id]["handle_actions"].append(
                {
                    "action_name": "clarification_request",
                    "query": event.query,
                    "call_id": event.call_id,
                },
            )
            action_query = cm.in_flight_actions[event.handle_id].get("query", "")
            short_desc = (
                action_query[:30] + "..." if len(action_query) > 30 else action_query
            )
            cm.notifications_bar.push_notif(
                "Action",
                f"Action '{short_desc}' needs clarification: {event.query}",
                event.timestamp,
            )
            await cm.request_llm_run()
    elif isinstance(event, ActorHandleResponse):
        # Handle response from an ask operation
        if event.handle_id in cm.in_flight_actions:
            handle_data = cm.in_flight_actions[event.handle_id]
            handle_actions = handle_data.get("handle_actions", [])

            # Find the pending ask action and update it with the response
            for action in reversed(handle_actions):
                if (
                    action.get("action_name") == f"ask_{event.handle_id}"
                    and action.get("status") == "pending"
                ):
                    action["status"] = "completed"
                    action["response"] = event.response
                    break

            # Wake the brain LLM to process the response
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

    thread = None
    message_content = None
    subject = None
    body = None
    email_id = None
    attachments = None
    notif_content = None

    # Get contact info from ContactManager, fallback to event.contact
    contact = cm.contact_index.get_contact(event.contact["contact_id"])
    if contact is None:
        contact = event.contact

    contact_id = (
        contact.get("contact_id")
        if isinstance(contact, dict)
        else event.contact["contact_id"]
    )
    sender_name = _get_sender_name(contact)

    match event:
        case SMSSent():
            medium = Medium.SMS_MESSAGE
            message_content = event.content
            notif_content = f"SMS sent to {sender_name}"
            role = "assistant"
        case SMSReceived():
            medium = Medium.SMS_MESSAGE
            message_content = event.content
            notif_content = f"SMS Received from {sender_name}"
            role = "user"
        case EmailSent():
            medium = Medium.EMAIL
            subject = event.subject
            body = event.body
            email_id = event.email_id_replied_to
            attachments = event.attachments
            notif_content = f"Email sent to {sender_name}"
            role = "assistant"
        case EmailReceived():
            medium = Medium.EMAIL
            subject = event.subject
            body = event.body
            email_id = event.email_id
            attachments = event.attachments
            notif_content = f"Email Received from {sender_name}"
            role = "user"
        case UnifyMessageSent():
            medium = Medium.UNIFY_MESSAGE
            message_content = event.content
            attachments = event.attachments
            notif_content = f"Unify message sent to {sender_name}"
            role = "assistant"
        case UnifyMessageReceived():
            medium = Medium.UNIFY_MESSAGE
            message_content = event.content
            attachments = event.attachments
            notif_content = f"Unify message from {sender_name}"
            role = "user"

    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=medium,
        message_content=message_content,
        subject=subject,
        body=body,
        email_id=email_id,
        attachments=attachments,
        timestamp=event.timestamp,
        role=role,
    )
    cm.notifications_bar.push_notif("comms", notif_content, event.timestamp)

    if role == "user":
        await cm.cancel_proactive_speech()

    await cm.request_llm_run(delay=2)


@EventHandler.register(BackupContactsEvent)
async def _(event: BackupContactsEvent, cm: "ConversationManager", *args, **kwargs):
    """
    Cache contacts from inbound messages for quick lookup.

    This handler is triggered when inbound messages arrive with contact data.
    Contacts are cached in ContactIndex and checked first in get_contact(),
    ensuring contacts from recent inbounds are always available even before
    or during ContactManager initialization.
    """
    print("BackupContactsEvent handler")
    if cm.contact_index._contact_manager:
        return
    print(f"Caching {len(event.contacts)} contacts from inbound")
    cm._session_logger.debug(
        "backup_contacts",
        f"Caching {len(event.contacts)} contacts from inbound",
    )
    cm.contact_index.set_fallback_contacts(event.contacts)


@EventHandler.register((StartupEvent))
async def _(event: StartupEvent, cm: "ConversationManager", *args, **kwargs):
    cm._session_logger.info("startup", "Received startup event")
    payload = event.to_dict()["payload"]
    cm.set_details(payload)
    cm.call_manager.set_config(cm.get_call_config())
    asyncio.create_task(
        asyncio.to_thread(
            debug_logger.log_job_startup,
            job_name=cm.job_name,
            user_id=cm.user_id,
            assistant_id=cm.assistant_id,
        ),
    )

    asyncio.create_task(managers_utils.init_conv_manager(cm))
    asyncio.create_task(managers_utils.listen_to_operations(cm))


@EventHandler.register(AssistantUpdateEvent)
async def _(event: AssistantUpdateEvent, cm: "ConversationManager", *args, **kwargs):
    cm._session_logger.info("assistant_update", "Received assistant update event")
    payload = event.to_dict()["payload"]
    cm.set_details(payload)
    cm.call_manager.set_config(cm.get_call_config())

    # Update contact manager with new assistant/user details
    await managers_utils.queue_operation(
        managers_utils.update_session_contacts,
        cm,
        event.assistant_name,
        event.assistant_number,
        event.assistant_email,
        event.user_name,
        event.user_number,
        event.user_email,
    )


@EventHandler.register(GetChatHistory)
async def _(event: GetChatHistory, cm: "ConversationManager", *args, **kwargs):
    cm._session_logger.debug(
        "state_update",
        f"Received chat history ({len(event.chat_history)} messages)",
    )
    cm.chat_history = event.chat_history + cm.chat_history


@EventHandler.register(ActorHandleStarted)
async def _(event: ActorHandleStarted, cm: "ConversationManager", *args, **kwargs):
    cm.notifications_bar.push_notif(
        "Action",
        f"Action started: {event.query[:50]}{'...' if len(event.query) > 50 else ''}",
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

    cm.notifications_bar.push_notif(
        event.source,
        event.content,
        event.timestamp,
        pinned=event.pinned,
        id=event.interjection_id,
    )

    await cm.cancel_proactive_speech()
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

    cm.notifications_bar.remove_notif(event.interjection_id)


@EventHandler.register(ActorResult)
async def _(event: ActorResult, cm: "ConversationManager", *args, **kwargs):
    action_data = cm.in_flight_actions.get(event.handle_id, {})
    action_query = action_data.get("query", f"Action {event.handle_id}")
    short_desc = action_query[:30] + "..." if len(action_query) > 30 else action_query

    cm.notifications_bar.push_notif(
        "Action",
        f"Action completed: {short_desc}\nResult: {event.result}",
        event.timestamp,
    )
    cm.in_flight_actions.pop(event.handle_id, None)
    await cm.request_llm_run()


@EventHandler.register((ActorPause, ActorResume))
async def _(
    event: ActorPause | ActorResume,
    cm: "ConversationManager",
    *args,
    **kwargs,
):
    op = "pause" if isinstance(event, ActorPause) else "resume"
    cm._session_logger.info("actor_request", f"Received action {op} event")
    reason = getattr(event, "reason", "")
    affected: list[int] = []
    for hid, data in list(cm.in_flight_actions.items()):
        handle = data.get("handle")
        if handle is None:
            continue

        try:
            if op == "pause":
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
                f"Failed to {op} action {hid}: {e}",
            )

    for hid in affected:
        try:
            await cm.event_broker.publish(
                "app:actor:notification",
                ActorNotification(
                    handle_id=int(hid),
                    response=f"Action {op}d: {reason}",
                ).to_json(),
            )
        except Exception as e:
            cm._session_logger.error(
                "actor_request",
                f"Failed to publish {op} notification for action {hid}: {e}",
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
    if (
        event.medium == Medium.PHONE_CALL
        and cm.call_manager.call_exchange_id == UNASSIGNED
    ):
        cm.call_manager.call_exchange_id = event.exchange_id
    if (
        event.medium == Medium.UNIFY_MEET
        and cm.call_manager.unify_meet_exchange_id == UNASSIGNED
    ):
        cm.call_manager.unify_meet_exchange_id = event.exchange_id


@EventHandler.register(PreHireMessage)
async def _(event: PreHireMessage, cm: "ConversationManager", *args, **kwargs):
    await managers_utils.queue_operation(managers_utils.log_message, cm, event)


@EventHandler.register(SummarizeContext)
async def _(event: SummarizeContext, cm: "ConversationManager", *args, **kwargs):
    if cm.memory_manager is None:
        cm._session_logger.debug(
            "summarize",
            "SummarizeContext skipped (MemoryManager disabled)",
        )
        cm.is_summarizing = False
        cm.chat_history = []
        return

    async def summarize_task():
        # Build render data for each active conversation
        render_data = []
        for contact_id, conv_state in cm.contact_index.active_conversations.items():
            contact_info = cm.contact_index.get_contact(contact_id) or {}
            rendered = cm.prompt_renderer.render_contact(
                contact_info=contact_info,
                conv_state=conv_state,
                max_messages=25,
                last_snapshot=cm.last_snapshot,
            )
            render_data.append((contact_id, rendered))

        tasks = [
            asyncio.create_task(
                cm.memory_manager.update_contact_rolling_summary(
                    rendered,
                    contact_id=cid,
                ),
            )
            for cid, rendered in render_data
        ]
        try:
            await asyncio.gather(*tasks)
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

    if cm.mode.is_voice:
        await cm.event_broker.publish(
            "app:call:call_guidance",
            json.dumps({"content": event.content}),
        )

    contact = cm.get_active_contact()
    contact_id = contact.get("contact_id") if contact else 1
    sender_name = _get_sender_name(contact)

    medium = Medium.UNIFY_MEET if cm.mode == Mode.MEET else Medium.PHONE_CALL
    cm.contact_index.push_message(
        contact_id=contact_id,
        sender_name=sender_name,
        thread_name=medium,
        message_content=event.content,
        role="assistant",
        timestamp=event.timestamp,
    )
