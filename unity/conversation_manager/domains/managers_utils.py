from datetime import timedelta
import asyncio
from time import perf_counter
from typing import TYPE_CHECKING

import unity

from unity.settings import SETTINGS
from unity.session_details import DEFAULT_ASSISTANT_ID, SESSION_DETAILS
from unity.common.async_tool_loop import SteerableToolHandle
from unity.contact_manager.types.contact import UNASSIGNED
from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.events import *
from unity.common.prompt_helpers import now as prompt_now
from unity.events.event_bus import EVENT_BUS
from unity.manager_registry import ManagerRegistry
from unity.conversation_manager.types import Medium

if TYPE_CHECKING:
    from unity.actor.base import BaseActor
    from unity.conversation_manager.conversation_manager import ConversationManager

event_broker = get_event_broker()

# Cache for pre-hire exchange ID - used to group all pre-hire messages into one exchange
_pre_hire_exchange_id: int | None = None

# Thought: This entire file could actually be turned into a mixin class


# EVENT BUS
async def get_last_store_chat_history() -> StoreChatHistory:
    bus_events = await EVENT_BUS.search(
        filter='type == "Comms" and payload_cls == "StoreChatHistory"',
        limit=1,
    )
    if len(bus_events):
        return Event.from_bus_event(bus_events[0])
    return None


async def publish_bus_events(event):
    try:
        event_name = event.__class__.__name__
        bus_event = event.to_bus_event()
        bus_event.payload.pop("api_key", None)
        bus_event.payload.pop("email_id", None)
        print("Publishing bus event", event_name)
        await EVENT_BUS.publish(bus_event)
    except Exception as e:
        print(f"[ManagersWorker] Error publishing bus event: {e}")


# ACTOR
async def actor_watch_result(
    handle_id: int,
    handle: SteerableToolHandle,
) -> None:
    """Await final result and publish completion (or failure), then cleanup."""
    # await result
    try:
        result = await handle.result()
    except Exception as e:
        result = f"Error getting actor result: {e}"
        print(f"[ManagersWorker] {result}")
    await event_broker.publish(
        "app:actor:result",
        ActorResult(
            handle_id=handle_id,
            success=False if "Error" in result else True,
            result=result,
        ).to_json(),
    )


async def actor_watch_notifications(
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

        # Extract a human-friendly message.
        #
        # Contract:
        # - Notifications may be plain strings (already display-ready), OR
        # - Structured dict payloads (recommended: include both "type" and "message").
        #
        # We keep this adapter strict and predictable: prefer "message"; otherwise
        # fall back to "type"; otherwise JSON-dump the payload.
        msg: str
        if isinstance(notif, dict):
            if notif.get("message") is not None:
                msg = str(notif.get("message"))
            elif notif.get("type") is not None:
                msg = str(notif.get("type"))
            else:
                try:
                    import json as _json

                    msg = _json.dumps(notif, ensure_ascii=False, default=str)
                except Exception:
                    msg = str(notif)
        else:
            msg = str(notif)

        # publish response
        await event_broker.publish(
            "app:actor:notification",
            ActorNotification(
                handle_id=handle_id,
                response=msg,
            ).to_json(),
        )


async def actor_watch_clarifications(
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
            "app:actor:clarification_request",
            ActorClarificationRequest(
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
        medium = Medium.UNIFY_MEET if "meet" in event_name else Medium.UNIFY_MESSAGE
    elif "phone" in event_name:
        medium = Medium.PHONE_CALL
    elif "sms" in event_name:
        medium = Medium.SMS_MESSAGE
    else:
        medium = Medium.EMAIL
    role = "Assistant" if "sent" in event_name or "assistant" in event_name else "User"
    if "prehire" in event_name:
        role = event.role.capitalize()
    if isinstance(event, (EmailSent, EmailReceived)):
        content = event.subject + "\n\n" + event.body
    else:
        content = event.content

    contact_id = None
    if isinstance(event, (PreHireMessage,)):
        # PreHireMessage is always boss context
        contact_id = 1
    elif isinstance(
        event,
        (
            UnifyMessageSent,
            UnifyMessageReceived,
            InboundUnifyMeetUtterance,
            OutboundUnifyMeetUtterance,
        ),
    ):
        # Use contact from event - contact_id must be valid, no silent fallback
        evt_contact_id = event.contact.get("contact_id")
        if cm.contact_index.get_contact(contact_id=evt_contact_id):
            contact_id = evt_contact_id
        else:
            # Log error but use the provided contact_id anyway since the event
            # already contains the full contact dict from the source
            print(
                f"Warning: contact_id {evt_contact_id} not in contact_index, "
                f"using contact from event",
            )
            contact_id = evt_contact_id
    elif cm.contact_index.get_contact(contact_id=event.contact["contact_id"]):
        contact_id = event.contact["contact_id"]
    if role == "Assistant":
        sender_id, receiver_ids = 0, [contact_id]
    else:
        sender_id, receiver_ids = contact_id, [0]

    exchange_id = getattr(event, "exchange_id", UNASSIGNED)

    # For pre-hire messages, reuse the cached exchange_id if available
    # This ensures all messages from a pre-hire chat batch go into the same exchange
    if isinstance(event, PreHireMessage):
        if _pre_hire_exchange_id is not None:
            exchange_id = _pre_hire_exchange_id
        # else: stays UNASSIGNED, will create new exchange
    elif medium == Medium.PHONE_CALL:
        exchange_id = cm.call_manager.call_exchange_id
    elif medium == Medium.UNIFY_MEET:
        exchange_id = cm.call_manager.unify_meet_exchange_id

    call_utterance_timestamp = ""
    call_url = ""
    # compute utterance timestamp based on active call type
    timestamp = (
        cm.call_manager.call_start_timestamp
        if medium == Medium.PHONE_CALL
        else (
            cm.call_manager.unify_meet_start_timestamp
            if medium == Medium.UNIFY_MEET
            else None
        )
    )
    if timestamp:
        delta = prompt_now(as_string=False) - timestamp
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
        global _pre_hire_exchange_id
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
                # Cache the exchange_id for subsequent pre-hire messages in the batch
                if isinstance(event, PreHireMessage):
                    _pre_hire_exchange_id = exchange_id
                    print(
                        f"[ManagersWorker] Cached pre-hire exchange_id: {exchange_id}",
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


# Contact updates


async def update_session_contacts(
    cm: "ConversationManager",
    assistant_name: str,
    assistant_number: str,
    assistant_email: str,
    user_name: str,
    user_number: str,
    user_email: str,
) -> None:
    """
    Update the assistant (contact_id=0) and boss (contact_id=1) contacts
    in the ContactManager when session details change.

    Called when an AssistantUpdateEvent is received.
    """
    if cm.contact_manager is None:
        print("[ManagersWorker] Cannot update contacts: contact_manager is None")
        return

    def _get_name_parts(name: str) -> tuple[str, str]:
        if " " in name:
            parts = name.split(" ", 1)
            return parts[0], parts[1]
        return name, ""

    async def _update_contact(
        contact_id: int,
        first_name: str,
        surname: str,
        phone_number: str,
        email_address: str,
    ):
        try:
            await asyncio.to_thread(
                cm.contact_manager.update_contact,
                contact_id=contact_id,
                phone_number=phone_number,
                email_address=email_address,
                first_name=first_name,
                surname=surname,
            )
            print(
                f"[ManagersWorker] Updated contact {contact_id}: {first_name} {surname}",
            )
        except Exception as e:
            print(f"[ManagersWorker] Failed to update contact {contact_id}: {e}")

    user_first_name, user_last_name = _get_name_parts(user_name)
    assistant_first_name, assistant_last_name = _get_name_parts(assistant_name)
    await _update_contact(
        0,
        assistant_first_name,
        assistant_last_name,
        assistant_number,
        assistant_email,
    )
    await _update_contact(1, user_first_name, user_last_name, user_number, user_email)


async def update_rolling_summaries(cm: "ConversationManager") -> None:
    """Update rolling summaries for all active conversations."""
    if cm.memory_manager is None:
        print("[ManagersWorker] Rolling summary skipped (MemoryManager disabled)")
        cm._session_logger.debug(
            "summarize",
            "Rolling summary skipped (MemoryManager disabled)",
        )
        cm.is_summarizing = False
        cm.chat_history = []
        return

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

    print(
        f"[ManagersWorker] Updating rolling summary for {len(render_data)} contacts: "
        f"{[cid for cid, _ in render_data]}",
    )

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
        print("[ManagersWorker] Rolling summary updated successfully")
        cm._session_logger.info("summarize", "Contact rolling summary updated")
    except Exception as e:
        print(f"[ManagersWorker] Error updating rolling summary: {e}")
        cm._session_logger.error(
            "summarize",
            f"Error updating rolling summary: {e}",
        )


# Queueing operations that need managers

_operations_queue = asyncio.Queue()


async def queue_operation(async_func: callable, *args, **kwargs) -> None:
    """
    Queue an async operation to be executed when managers are initialized.
    The operation will be processed by listen_to_operations().
    """
    await _operations_queue.put((async_func, args, kwargs))


async def wait_for_initialization(
    cm: "ConversationManager",
    timeout: float = 30.0,
) -> None:
    """
    Wait for initialization to complete.

    Args:
        cm: The ConversationManager instance to wait for.
        timeout: Maximum seconds to wait before raising an error. Default 30s.

    Raises:
        RuntimeError: If initialization does not complete within the timeout.
    """
    import time

    start = time.monotonic()
    while not cm.initialized:
        if time.monotonic() - start > timeout:
            raise RuntimeError(
                f"ConversationManager initialization did not complete within {timeout}s. "
                "Check for initialization errors above.",
            )
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
    actor: "BaseActor | None" = None,
) -> None:
    """
    Initialize all managers in a separate thread.
    The main event loop is passed for managers that need to schedule async tasks.

    Args:
        cm: The ConversationManager instance to initialize.
        loop: The main event loop for scheduling async tasks.
        actor: Optional pre-instantiated Actor. If provided, used directly instead
            of creating one via ManagerRegistry. Useful for testing with specific
            Actor implementations.
    """
    start_time = perf_counter()

    # 0. Initialize unity using SESSION_DETAILS (the canonical source of session config)
    print("[ManagersWorker] Initializing unity...")
    local_start_time = perf_counter()
    if not SESSION_DETAILS.assistant_record:
        # When default_assistant is provided, unity.init() uses it directly
        # and ignores the assistant_id parameter entirely
        unity.init(
            default_assistant={
                "agent_id": SESSION_DETAILS.assistant.id,
                "first_name": SESSION_DETAILS.assistant.name,
                "age": SESSION_DETAILS.assistant.age,
                "nationality": SESSION_DETAILS.assistant.nationality,
                "timezone": SESSION_DETAILS.assistant.timezone or None,
                "about": SESSION_DETAILS.assistant.about,
                "phone": SESSION_DETAILS.assistant.number or None,
                "email": SESSION_DETAILS.assistant.email or None,
                "user_id": SESSION_DETAILS.user.id,
                "user_phone": SESSION_DETAILS.user.number or None,
                "created_at": prompt_now(as_string=False).isoformat(),
                "updated_at": prompt_now(as_string=False).isoformat(),
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

    # Get API key from SESSION_DETAILS (set by ConversationManager on startup)
    api_key = SESSION_DETAILS.unify_key or None

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

    # 2. Initialize ContactManager (respects SETTINGS.contact.IMPL)
    print("[ManagersWorker] Initializing ContactManager...")
    local_start_time = perf_counter()
    cm.contact_manager = ManagerRegistry.get_contact_manager(
        description="production deployment",
    )
    # Wire up ContactManager to ContactIndex for always-fresh contact data
    cm.contact_index.set_contact_manager(cm.contact_manager)
    print(
        f"[ManagersWorker] ContactManager ({type(cm.contact_manager).__name__}) initialized in "
        f"{perf_counter() - local_start_time:.2f} seconds",
    )

    # 3. Initialize TranscriptManager (respects SETTINGS.transcript.IMPL)
    print("[ManagersWorker] Initializing TranscriptManager...")
    local_start_time = perf_counter()
    cm.transcript_manager = ManagerRegistry.get_transcript_manager(
        description="production deployment",
        contact_manager=cm.contact_manager,
    )
    print(
        f"[ManagersWorker] TranscriptManager ({type(cm.transcript_manager).__name__}) initialized in "
        f"{perf_counter() - local_start_time:.2f} seconds",
    )

    # 4. Configure TranscriptManager logger (only for real implementation)
    # Check hasattr instead of SETTINGS to be defensive against implementation mismatches
    if api_key and hasattr(cm.transcript_manager, "_get_logger"):
        cm.transcript_manager._get_logger().session.headers[
            "Authorization"
        ] = f"Bearer {api_key}"

    # 5. Initialize MemoryManager (optional - respects SETTINGS.memory.ENABLED and IMPL)
    if SETTINGS.memory.ENABLED:
        print("[ManagersWorker] Initializing MemoryManager...")
        local_start_time = perf_counter()
        cm.memory_manager = ManagerRegistry.get_memory_manager(
            transcript_manager=cm.transcript_manager,
            contact_manager=cm.contact_manager,
            loop=loop,
        )
        print(
            "[ManagersWorker] MemoryManager initialized in "
            f"{perf_counter() - local_start_time:.2f} seconds",
        )
    else:
        print("[ManagersWorker] MemoryManager disabled (SETTINGS.memory.ENABLED=False)")

    # 6. Initialize ConversationManagerHandle (respects SETTINGS.conversation.IMPL)
    print("[ManagersWorker] Initializing ConversationManagerHandle...")
    local_start_time = perf_counter()
    # ConversationManagerHandle has different constructor args for real vs simulated
    if SETTINGS.conversation.IMPL == "simulated":
        cm._conversation_manager_handle = (
            ManagerRegistry.get_conversation_manager_handle(
                description="production deployment",
                assistant_id=SESSION_DETAILS.assistant.id,
                contact_id="1",
            )
        )
    else:
        cm._conversation_manager_handle = (
            ManagerRegistry.get_conversation_manager_handle(
                event_broker=cm.event_broker,
                conversation_id=SESSION_DETAILS.assistant.id,
                contact_id="1",
                transcript_manager=cm.transcript_manager,
                conversation_manager=cm,
            )
        )
    print(
        f"[ManagersWorker] ConversationManagerHandle ({type(cm._conversation_manager_handle).__name__}) initialized in "
        f"{perf_counter() - local_start_time:.2f} seconds",
    )

    # 7. Initialize Actor (use provided actor or create via ManagerRegistry)
    print("[ManagersWorker] Initializing Actor...")
    try:
        local_start_time = perf_counter()
        if actor is not None:
            # Use pre-instantiated actor (e.g., for testing)
            cm.actor = actor
        else:
            # Create via ManagerRegistry (respects SETTINGS.actor.IMPL)
            cm.actor = ManagerRegistry.get_actor(
                description="production deployment",
            )
        actor_cls = type(cm.actor).__name__
        print(
            f"[ManagersWorker] Actor ({actor_cls}) initialized in "
            f"{perf_counter() - local_start_time:.2f} seconds",
        )
    except Exception as e:
        print(f"[ManagersWorker] Error initializing Actor: {e}")

    print(
        "[ManagersWorker] All managers initialized in "
        f"{perf_counter() - start_time:.2f} seconds",
    )


async def _start_file_sync() -> None:
    """Start file sync with managed VM after managers are initialized.

    This starts rclone-based file synchronization between ~/Unity (assistant)
    and /Unity (managed VM) if a desktop_url is configured in SESSION_DETAILS.

    Runs asynchronously and logs success/failure.
    """
    from unity.session_details import SESSION_DETAILS

    # Only sync when a desktop_url is configured
    if not SESSION_DETAILS.assistant.desktop_url:
        print("[ManagersWorker] No desktop_url configured, skipping file sync")
        return

    try:
        from unity.file_manager.managers.local import LocalFileManager

        # Get LocalFileManager singleton (may already exist from manager init)
        local_fm = LocalFileManager()
        adapter = local_fm._adapter

        # Check if adapter supports sync (LocalFileSystemAdapter does)
        if not hasattr(adapter, "start_sync"):
            print("[ManagersWorker] Adapter does not support file sync")
            return

        if adapter._enable_sync:
            print("[ManagersWorker] Starting file sync with managed VM...")
            success = await adapter.start_sync()
            if success:
                print("[ManagersWorker] File sync started successfully")
            else:
                print("[ManagersWorker] File sync not enabled or failed to start")
        else:
            print("[ManagersWorker] File sync disabled by configuration")

    except Exception as e:
        # File sync failure should not block manager initialization
        print(f"[ManagersWorker] Failed to start file sync: {e}")
        import traceback

        traceback.print_exc()


async def init_conv_manager(
    cm: "ConversationManager",
    *,
    actor: "BaseActor | None" = None,
) -> None:
    """
    Initialize all managers for the ConversationManager.
    All initialization runs in a separate thread (non-blocking).

    Args:
        cm: The ConversationManager instance to initialize.
        actor: Optional pre-instantiated Actor. If provided, used directly instead
            of creating one via ManagerRegistry. Useful for testing with specific
            Actor implementations (e.g., SimulatedActor).
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
            await asyncio.to_thread(_init_managers, cm, loop, actor)

            store_chat_history = await get_last_store_chat_history()
            if store_chat_history:
                await cm.event_broker.publish(
                    "app:comms:chat_history",
                    GetChatHistory(
                        chat_history=store_chat_history.chat_history,
                    ).to_json(),
                )

            # Start file sync with managed VM (non-blocking)
            await _start_file_sync()

            # Mark as initialized
            cm.initialized = True

            # Publish initialization complete event for test synchronization
            await event_broker.publish(
                "app:comms:initialization_complete",
                InitializationComplete().to_json(),
            )

            print(
                "[ManagersWorker] Initialization complete in "
                f"{perf_counter() - start_time:.2f} seconds",
            )

        except Exception as e:
            print(f"[ManagersWorker] Error during initialization: {e}")
            raise
