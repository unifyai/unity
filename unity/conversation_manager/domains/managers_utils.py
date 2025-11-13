from datetime import timedelta
import os
import asyncio
from time import perf_counter
from typing import TYPE_CHECKING

import unity

from unity.common.async_tool_loop import SteerableToolHandle
from unity.contact_manager.types.contact import UNASSIGNED
from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.new_events import *
from unity.contact_manager.contact_manager import ContactManager
from unity.conversation_manager.handle import ConversationManagerHandle
from unity.conversation_manager.new_events import *
from unity.events.event_bus import EVENT_BUS
from unity.memory_manager.memory_manager import MemoryManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.conductor.conductor import Conductor

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager

event_broker = get_event_broker()

# Thought: This entire file could actually be turned into a mixin class


# EVENT BUS
async def get_bus_events():
    bus_events = await EVENT_BUS.search(filter='type == "Comms"', limit=50)
    return [Event.from_bus_event(e).to_dict() for e in bus_events][::-1]


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
        exchange_id = cm.call_exchange_id
    if medium == "unify_call":
        exchange_id = cm.unify_call_exchange_id

    call_utterance_timestamp = ""
    call_url = ""
    # compute utterance timestamp based on active call type
    timestamp = (
        cm.call_start_timestamp
        if medium == "phone_call"
        else (cm.unify_call_start_timestamp if medium == "unify_call" else None)
    )
    if timestamp:
        delta = datetime.now() - timestamp
        if role == "Assistant":
            delta += timedelta(seconds=2)
        minutes, seconds = divmod(int(delta.total_seconds()), 60)
        # ToDo: Make this MM:SS once we have explicit types working
        call_utterance_timestamp = f"{minutes:02d}.{seconds:02d}"
    if "default-assistant" not in cm.assistant_id:
        call_url = (
            "https://storage.cloud.google.com/assistant-call-recordings/staging/"
            f"{cm.assistant_id}/{cm.conference_name}.mp3"
        )
    try:
        print(f"[ManagersWorker] Logging message: {event.to_dict()}")
        # call_utterance_timestamp = event.call_utterance_timestamp
        # call_url = event.call_url
        metadata = getattr(event, "metadata", None)

        # Log the message
        messages = cm.transcript_manager.log_messages(
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

        message = messages[0] if messages else None
        print(
            f"[ManagersWorker] Logged message: {medium}"
            f" from {sender_id} to {receiver_ids}",
        )

        # Publish reply as Event envelope
        if message:
            await event_broker.publish(
                "app:logging:message_logged",
                LogMessageResponse(
                    medium=medium,
                    exchange_id=message.exchange_id,
                ).to_json(),
            )
            print(f"[ManagersWorker] Published exchange_id {message.exchange_id}")

    except Exception as e:
        print(f"[ManagersWorker] Error logging message: {e}")


_init_lock = asyncio.Lock()
_initialized = False


# TODO: this will be blocking so might have to run it in a thread? it should be fast but its actually very slow it seems
async def init_conv_manager(cm: "ConversationManager"):
    print("[ManagersWorker] Processing startup")
    global _init_lock, _initialized

    async with _init_lock:
        start_time = perf_counter()
        if _initialized:
            print("[ManagersWorker] Already initialized, skipping")
            return

        try:
            # 0. Initialize unity
            print("[ManagersWorker] Initializing unity...")
            payload = {
                "agent_id": cm.assistant_id,
                "first_name": cm.assistant_name,
                "age": cm.assistant_age,
                "nationality": cm.assistant_nationality,
                "about": cm.assistant_about,
                "phone": cm.assistant_number,
                "email": cm.assistant_email,
                "user_phone": cm.user_number,
                "user_whatsapp_number": cm.user_whatsapp_number,
                "assistant_whatsapp_number": cm.assistant_number,
            }
            if not unity.ASSISTANT:
                unity.init(
                    assistant_id=int(
                        payload.get("agent_id", "0").replace(
                            "default-assistant-",
                            "",
                        ),
                    ),
                    default_assistant={
                        **payload,
                        "user_id": "default-user",
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                        "surname": "",
                        "weekly_limit": None,
                        "max_parallel": None,
                        "profile_photo": None,
                        "country": None,
                        "user_last_name": "",
                        "phone": payload["phone"] or None,
                        "email": payload["email"] or None,
                        "user_phone": payload["user_phone"] or None,
                        "user_whatsapp_number": payload["user_whatsapp_number"] or None,
                        "assistant_whatsapp_number": payload[
                            "assistant_whatsapp_number"
                        ]
                        or None,
                    },
                )
            print("[ManagersWorker] Unity initialized")
            # print("Clearing all events for clean testing")
            # EVENT_BUS.reset()

            # Assumes UNIFY_KEY is already in environment from set_details()
            api_key = os.environ.get("UNIFY_KEY")

            # 1. Configure EventBus
            print("[ManagersWorker] Configuring EventBus...")
            if api_key:
                EVENT_BUS._get_logger().session.headers[
                    "Authorization"
                ] = f"Bearer {api_key}"
            # event_bus auto-pinning registration
            EVENT_BUS.set_window("Comms", 50)
            EVENT_BUS.register_auto_pin(
                event_type="Comms",
                open_predicate=lambda e: e.payload.get("role", "") == "tool_use start",
                close_predicate=lambda e: e.payload.get("role", "") == "tool_use end",
                key_fn=lambda e: e.payload.get("handle_id", ""),
            )
            bus_events_task = asyncio.create_task(get_bus_events())
            EVENT_BUS.clear()
            print("[ManagersWorker] EventBus configured")

            # 2. Initialize ContactManager and get contacts
            print("[ManagersWorker] Initializing ContactManager...")
            cm.contact_manager = ContactManager()

            # contacts_task = asyncio.create_task(get_contacts())
            # await asyncio.gather(bus_events_task, contacts_task)
            await bus_events_task
            print("[ManagersWorker] ContactManager initialized")

            # 3. Initialize TranscriptManager with ContactManager
            print("[ManagersWorker] Initializing TranscriptManager...")
            cm.transcript_manager = TranscriptManager(
                contact_manager=cm.contact_manager,
            )
            print("[ManagersWorker] TranscriptManager initialized")

            # 4. Configure TranscriptManager logger with auth header
            if api_key:
                cm.transcript_manager._get_logger().session.headers[
                    "Authorization"
                ] = f"Bearer {api_key}"
                print("[ManagersWorker] TranscriptManager logger configured")

            # TODO: Initialize other managers (Conductor, etc.) here
            print("[ManagersWorker] Initializing MemoryManager...")
            cm.memory_manager = MemoryManager(
                transcript_manager=cm.transcript_manager,
                contact_manager=cm.contact_manager,
            )
            print("[ManagersWorker] MemoryManager initialized")

            # 5. Initialize ConversationManager
            print("[ManagersWorker] Initializing ConversationManagerHandle...")
            conversation_manager_handle = ConversationManagerHandle(
                event_broker=cm.event_broker,
                conversation_id=os.getenv("ASSISTANT_ID", "default-assistant"),
                contact_id="1",
                transcript_manager=cm.transcript_manager,
            )
            print("[ManagersWorker] ConversationManagerHandle initialized")

            # 6. Initialize Conductor with existing managers
            print("[ManagersWorker] Initializing Conductor...")
            try:
                cm.conductor = Conductor(
                    contact_manager=cm.contact_manager,
                    transcript_manager=cm.transcript_manager,
                    conversation_manager=conversation_manager_handle,
                )
                print("[ManagersWorker] Conductor initialized")
            except Exception as e:
                print(f"[ManagersWorker] Error initializing Conductor: {e}")

            _initialized = True
            print("[ManagersWorker] Initialization complete")

        except Exception as e:
            print(f"[ManagersWorker] Error during initialization: {e}")

        cm.initialized = True

        print(
            "[ManagersWorker] Initialization complete in "
            f"{perf_counter() - start_time:.2f} seconds",
        )
