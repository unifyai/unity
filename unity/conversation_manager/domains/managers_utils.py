from datetime import timedelta
import os
import asyncio
from time import perf_counter
from typing import TYPE_CHECKING

import unity

from unity.session_details import DEFAULT_ASSISTANT_ID, SESSION_DETAILS
from unity.common.async_tool_loop import SteerableToolHandle
from unity.contact_manager.types.contact import UNASSIGNED
from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.events import *
from unity.conversation_manager.events import *
from unity.events.event_bus import EVENT_BUS
from unity.memory_manager.memory_manager import MemoryManager
from unity.conductor.conductor import Conductor
from unity.conductor.manager_registry import get_class
from unity.settings import SETTINGS

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager

event_broker = get_event_broker()

# Thought: This entire file could actually be turned into a mixin class


# EVENT BUS
async def get_last_store_chat_history() -> StoreChatHistory:
    bus_events = await EVENT_BUS.search(
        filter='type == "Comms" and payload_cls == "StoreChatHistory"',
        limit=1,
    )
    if len(bus_events):
        return Event.from_bus_event(bus_events[0]).to_dict()
    return None


async def publish_bus_events(event):
    try:
        event_name = event.__class__.__name__
        bus_event = event.to_bus_event()
        bus_event.payload.pop("api_key", None)
        bus_event.payload.pop("message_id", None)
        print("Publishing bus event", event_name)
        await EVENT_BUS.publish(bus_event)
    except Exception as e:
        print(f"[ManagersWorker] Error publishing bus event: {e}")


# CONDUCTOR
async def conductor_watch_result(
    handle_id: int,
    handle: SteerableToolHandle,
) -> None:
    """Await final result and publish completion (or failure), then cleanup."""
    # await result
    try:
        result = await handle.result()
    except Exception as e:
        result = f"Error getting conductor result: {e}"
        print(f"[ManagersWorker] {result}")
    await event_broker.publish(
        "app:conductor:result",
        ConductorResult(
            handle_id=handle_id,
            success=False if "Error" in result else True,
            result=result,
        ).to_json(),
    )


async def conductor_watch_notifications(
    handle_id: int,
    handle: SteerableToolHandle,
) -> None:
    """Forward notifications as handle responses until handle completes."""
    while not handle.done():
        # await notification
        try:
            notif = await asyncio.wait_for(handle.next_notification(), timeout=30)
        except asyncio.TimeoutError:
            continue

        # get message
        msg = notif.get("message") if isinstance(notif, dict) else str(notif)

        # publish response
        await event_broker.publish(
            "app:conductor:notification",
            ConductorNotification(
                handle_id=handle_id,
                response=msg,
            ).to_json(),
        )


async def conductor_watch_clarifications(
    handle_id: int,
    handle: SteerableToolHandle,
) -> None:
    """Forward clarifications to CM until handle completes."""
    while not handle.done():
        # await clarification request
        try:
            clar = await asyncio.wait_for(handle.next_clarification(), timeout=30)
        except asyncio.TimeoutError:
            continue

        # get question and call id
        q = clar.get("question") if isinstance(clar, dict) else str(clar)
        call_id = clar.get("call_id") if isinstance(clar, dict) else None

        # publish clarification request
        await event_broker.publish(
            "app:conductor:clarification_request",
            ConductorClarificationRequest(
                handle_id=handle_id,
                query=q,
                call_id=call_id,
            ).to_json(),
        )


async def log_message(cm: "ConversationManager", event: Event) -> None:
    """Log a message via TranscriptManager."""
    event_name = event.__class__.__name__
    print("publishing transcript", event_name)
    event_name = event_name.lower()
    if "unify" in event_name or "prehire" in event_name:
        medium = "unify_call" if "call" in event_name else "unify_message"
    elif "phone" in event_name:
        medium = "phone_call"
    elif "sms" in event_name:
        medium = "sms_message"
    elif "email" in event_name:
        medium = "email"
    else:
        medium = "whatsapp_message"
    role = "Assistant" if "sent" in event_name or "assistant" in event_name else "User"
    if "prehire" in event_name:
        role = event.role.capitalize()
    if isinstance(event, (EmailSent, EmailReceived)):
        content = event.subject + "\n\n" + event.body
    else:
        content = event.content

    contact_id = None
    if isinstance(
        event,
        (
            UnifyMessageSent,
            UnifyMessageReceived,
            UnifyCallUtterance,
            AssistantUnifyCallUtterance,
            PreHireMessage,
        ),
    ):
        contact_id = 1
    elif event.contact["contact_id"] in cm.contact_index.contacts:
        contact_id = event.contact["contact_id"]
    if role == "Assistant":
        sender_id, receiver_ids = 0, [contact_id]
    else:
        sender_id, receiver_ids = contact_id, [0]

    exchange_id = getattr(event, "exchange_id", UNASSIGNED)
    if medium == "phone_call":
        exchange_id = cm.call_manager.call_exchange_id
    if medium == "unify_call":
        exchange_id = cm.call_manager.unify_call_exchange_id

    call_utterance_timestamp = ""
    call_url = ""
    # compute utterance timestamp based on active call type
    timestamp = (
        cm.call_manager.call_start_timestamp
        if medium == "phone_call"
        else (
            cm.call_manager.unify_call_start_timestamp
            if medium == "unify_call"
            else None
        )
    )
    if timestamp:
        delta = datetime.now() - timestamp
        if role == "Assistant":
            delta += timedelta(seconds=2)
        minutes, seconds = divmod(int(delta.total_seconds()), 60)
        # ToDo: Make this MM:SS once we have explicit types working
        call_utterance_timestamp = f"{minutes:02d}.{seconds:02d}"
    if DEFAULT_ASSISTANT_ID not in SESSION_DETAILS.assistant.id:
        call_url = (
            "https://storage.cloud.google.com/assistant-call-recordings/staging/"
            f"{cm.assistant_id}/{cm.call_manager.conference_name}.mp3"
        )

    # publish transcript on a separate thread
    def _publish_transcript() -> int:
        try:
            nonlocal exchange_id
            print(f"[ManagersWorker] Logging message: {event.to_dict()}")
            # call_utterance_timestamp = event.call_utterance_timestamp
            # call_url = event.call_url

            if exchange_id == UNASSIGNED:
                exchange_id = cm.transcript_manager.log_first_message_in_new_exchange(
                    {
                        "medium": medium,
                        "sender_id": sender_id,
                        "receiver_ids": receiver_ids,
                        "timestamp": event.timestamp,
                        "content": content,
                    },
                )
            else:
                metadata = getattr(event, "metadata", None)
                cm.transcript_manager.log_messages(
                    {
                        "medium": medium,
                        "sender_id": sender_id,
                        "receiver_ids": receiver_ids,
                        # not sure if this is right but that's how it is in the code in main
                        "timestamp": event.timestamp,
                        "content": content,
                        "exchange_id": exchange_id,
                        # "call_utterance_timestamp": call_utterance_timestamp,
                        # "call_url": call_url,
                        "_metadata": metadata,
                    },
                    synchronous=True,
                )

            print(
                f"[ManagersWorker] Logged message: {medium}"
                f" from {sender_id} to {receiver_ids}",
            )
            return exchange_id
        except Exception as e:
            print(f"[ManagersWorker] Error logging message: {e}")

    exchange_id = await asyncio.to_thread(_publish_transcript)

    # publish reply as event envelope
    await event_broker.publish(
        "app:logging:message_logged",
        LogMessageResponse(
            medium=medium,
            exchange_id=exchange_id,
        ).to_json(),
    )
    print(f"[ManagersWorker] Published exchange_id {exchange_id}")


# Queueing operations that need managers

_operations_queue = asyncio.Queue()


async def queue_operation(async_func: callable, *args, **kwargs) -> None:
    """
    Queue an async operation to be executed when managers are initialized.
    The operation will be processed by listen_to_operations().
    """
    await _operations_queue.put((async_func, args, kwargs))
    func_name = getattr(async_func, "__name__", str(async_func))
    print(f"[ManagersWorker] Queued operation: {func_name}")


async def wait_for_initialization(cm: "ConversationManager") -> None:
    """
    Wait for initialization to complete.
    """
    while not cm.initialized:
        await asyncio.sleep(0.1)


async def listen_to_operations(cm: "ConversationManager") -> None:
    """
    Worker loop that processes queued operations once initialized.
    Should be started as a background task alongside init_conv_manager.
    """
    # Wait for initialization to complete
    await wait_for_initialization(cm)

    print("[ManagersWorker] Operations listener started, processing queue...")

    # Process operations as they come in
    while True:
        # Wait for next operation (with timeout to allow checking for shutdown)
        try:
            async_func, args, kwargs = await asyncio.wait_for(
                _operations_queue.get(),
                timeout=1.0,
            )
        except asyncio.TimeoutError:
            continue

        # Execute the operation
        func_name = getattr(async_func, "__name__", str(async_func))
        try:
            await async_func(*args, **kwargs)
        except Exception as e:
            print(f"[ManagersWorker] Error executing {func_name}: {e}")
        finally:
            _operations_queue.task_done()


# Initialization

_init_lock = asyncio.Lock()


def _init_managers(
    cm: "ConversationManager",
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Initialize all managers in a separate thread.
    The main event loop is passed for managers that need to schedule async tasks.
    """
    start_time = perf_counter()

    # 0. Initialize unity using SESSION_DETAILS (the canonical source of session config)
    print("[ManagersWorker] Initializing unity...")
    local_start_time = perf_counter()
    if not unity.ASSISTANT:
        # When default_assistant is provided, unity.init() uses it directly
        # and ignores the assistant_id parameter entirely
        unity.init(
            default_assistant={
                "agent_id": SESSION_DETAILS.assistant.id,
                "first_name": SESSION_DETAILS.assistant.name,
                "age": SESSION_DETAILS.assistant.age,
                "nationality": SESSION_DETAILS.assistant.nationality,
                "about": SESSION_DETAILS.assistant.about,
                "phone": SESSION_DETAILS.assistant.number or None,
                "email": SESSION_DETAILS.assistant.email or None,
                "user_id": SESSION_DETAILS.user.id,
                "user_phone": SESSION_DETAILS.user.number or None,
                "user_whatsapp_number": SESSION_DETAILS.user.whatsapp_number or None,
                "assistant_whatsapp_number": SESSION_DETAILS.assistant.number or None,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "surname": "",
                "weekly_limit": None,
                "max_parallel": None,
                "profile_photo": None,
                "country": None,
                "user_last_name": "",
            },
        )
    print(
        "[ManagersWorker] Unity initialized in "
        f"{perf_counter() - local_start_time:.2f} seconds",
    )

    # Assumes UNIFY_KEY is already in environment from set_details()
    api_key = os.environ.get("UNIFY_KEY")

    # 1. Configure EventBus
    print("[ManagersWorker] Configuring EventBus...")
    local_start_time = perf_counter()
    if api_key:
        EVENT_BUS._get_logger().session.headers["Authorization"] = f"Bearer {api_key}"
    EVENT_BUS.set_window("Comms", 50)
    print(
        "[ManagersWorker] EventBus configured in "
        f"{perf_counter() - local_start_time:.2f} seconds",
    )

    # 2. Initialize ContactManager (respects UNITY_CONTACTS_IMPL setting)
    print("[ManagersWorker] Initializing ContactManager...")
    local_start_time = perf_counter()
    contacts_cls = get_class("contacts", SETTINGS.UNITY_CONTACTS_IMPL)
    if SETTINGS.UNITY_CONTACTS_IMPL == "simulated":
        cm.contact_manager = contacts_cls(description="production deployment")
    else:
        cm.contact_manager = contacts_cls()
    print(
        f"[ManagersWorker] ContactManager ({contacts_cls.__name__}) initialized in "
        f"{perf_counter() - local_start_time:.2f} seconds",
    )

    # 3. Initialize TranscriptManager (respects UNITY_TRANSCRIPTS_IMPL setting)
    print("[ManagersWorker] Initializing TranscriptManager...")
    local_start_time = perf_counter()
    transcripts_cls = get_class("transcripts", SETTINGS.UNITY_TRANSCRIPTS_IMPL)
    if SETTINGS.UNITY_TRANSCRIPTS_IMPL == "simulated":
        cm.transcript_manager = transcripts_cls(description="production deployment")
    else:
        cm.transcript_manager = transcripts_cls(contact_manager=cm.contact_manager)
    print(
        f"[ManagersWorker] TranscriptManager ({transcripts_cls.__name__}) initialized in "
        f"{perf_counter() - local_start_time:.2f} seconds",
    )

    # 4. Configure TranscriptManager logger (only for real implementation)
    if api_key and SETTINGS.UNITY_TRANSCRIPTS_IMPL != "simulated":
        cm.transcript_manager._get_logger().session.headers[
            "Authorization"
        ] = f"Bearer {api_key}"

    # 5. Initialize MemoryManager (pass loop for thread-safe task scheduling)
    print("[ManagersWorker] Initializing MemoryManager...")
    local_start_time = perf_counter()
    cm.memory_manager = MemoryManager(
        transcript_manager=cm.transcript_manager,
        contact_manager=cm.contact_manager,
        loop=loop,
    )
    print(
        "[ManagersWorker] MemoryManager initialized in "
        f"{perf_counter() - local_start_time:.2f} seconds",
    )

    # 6. Initialize ConversationManagerHandle (respects UNITY_CONVERSATION_IMPL setting)
    print("[ManagersWorker] Initializing ConversationManagerHandle...")
    local_start_time = perf_counter()
    conversation_cls = get_class("conversation", SETTINGS.UNITY_CONVERSATION_IMPL)
    if SETTINGS.UNITY_CONVERSATION_IMPL == "simulated":
        cm._conversation_manager_handle = conversation_cls(
            assistant_id=SESSION_DETAILS.assistant.id,
            contact_id="1",
            description="production deployment",
        )
    else:
        cm._conversation_manager_handle = conversation_cls(
            event_broker=cm.event_broker,
            conversation_id=SESSION_DETAILS.assistant.id,
            contact_id="1",
            transcript_manager=cm.transcript_manager,
            conversation_manager=cm,
        )
    print(
        f"[ManagersWorker] ConversationManagerHandle ({conversation_cls.__name__}) initialized in "
        f"{perf_counter() - local_start_time:.2f} seconds",
    )

    # 7. Initialize Conductor (all other managers use settings automatically)
    print("[ManagersWorker] Initializing Conductor...")
    try:
        local_start_time = perf_counter()
        cm.conductor = Conductor(
            contact_manager=cm.contact_manager,
            transcript_manager=cm.transcript_manager,
            conversation_manager=cm._conversation_manager_handle,
        )
        print(
            f"[ManagersWorker] Conductor initialized in "
            f"{perf_counter() - local_start_time:.2f} seconds",
        )
    except Exception as e:
        print(f"[ManagersWorker] Error initializing Conductor: {e}")

    print(
        "[ManagersWorker] All managers initialized in "
        f"{perf_counter() - start_time:.2f} seconds",
    )


async def init_conv_manager(cm: "ConversationManager") -> None:
    """
    Initialize all managers for the ConversationManager.
    All initialization runs in a separate thread (non-blocking).
    """
    print("[ManagersWorker] Processing startup")

    async with _init_lock:
        start_time = perf_counter()
        if cm.initialized:
            print("[ManagersWorker] Already initialized, skipping")
            return

        try:
            # Get the main event loop to pass to managers that need it
            loop = asyncio.get_running_loop()

            # Run all manager initialization in a thread (non-blocking)
            await asyncio.to_thread(_init_managers, cm, loop)

            store_chat_history = await get_last_store_chat_history()
            if store_chat_history:
                await cm.event_broker.publish(
                    "app:comms:chat_history",
                    GetChatHistory(chat_history=store_chat_history.chat_history).to_json(),
                )

            # Mark as initialized
            cm.initialized = True

            print(
                "[ManagersWorker] Initialization complete in "
                f"{perf_counter() - start_time:.2f} seconds",
            )

        except Exception as e:
            print(f"[ManagersWorker] Error during initialization: {e}")
