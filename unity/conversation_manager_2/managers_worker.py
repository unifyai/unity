from __future__ import annotations

import asyncio
from time import perf_counter
from dotenv import load_dotenv
import threading
from datetime import datetime
import os
from typing import Optional, Any, Callable
import contextlib
import redis.asyncio as redis

load_dotenv()

import unity
from unity.conversation_manager_2.handle import ConversationManagerHandle
from unity.memory_manager.memory_manager import MemoryManager
from unity.contact_manager.contact_manager import ContactManager
from unity.events.event_bus import EVENT_BUS
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.conversation_manager_2.new_events import *
from unity.conductor.conductor import Conductor
from unity.common.async_tool_loop import SteerableToolHandle


class ManagersWorker:
    """
    Background worker that subscribes to Redis Pub/Sub events from ConversationManager
    and handles ContactManager, TranscriptManager, and Conductor operations.

    Runs as a separate async task parallel to the conversation manager.
    Uses Pub/Sub with an internal queue to ensure FIFO ordering.
    """

    def __init__(self, event_broker: Optional[redis.Redis] = None):
        self._event_broker = event_broker

        # Pub/Sub channels
        self._subscribe_channel = "app:managers:input"
        self._publish_channel = "app:managers:output"

        # Internal queue for ordered processing
        self._message_queue: asyncio.Queue = asyncio.Queue()

        # Managers (initialized on startup message)
        self._contact_manager: Optional[ContactManager] = None
        self._transcript_manager: Optional[TranscriptManager] = None
        self._memory_manager: Optional[MemoryManager] = None
        self._conductor: Optional[Conductor] = None

        # State flags
        self._init_lock = asyncio.Lock()
        self._stop_event = threading.Event()

        # Startup flag
        self._initialized = False

        # Conductor handle registry: incrementing int handle_id -> handle
        self._next_handle_id: int = 0
        self._handle_registry: dict[int, dict[str, Any]] = {}
        # Minimal meta for tracing (per handle_id)
        self._handle_meta: dict[int, dict] = {}

        # Per-requirement queues and consumer tasks
        self._queues: dict[str, asyncio.Queue[tuple[Event, Callable[[], bool]]]] = {
            "initialized": asyncio.Queue(),
            "contact_manager": asyncio.Queue(),
            "transcript_manager": asyncio.Queue(),
            "memory_manager": asyncio.Queue(),
            "conductor": asyncio.Queue(),
        }
        self._queue_consumers: list[asyncio.Task] = []

    # ──────────────────────────────────────────────────────────────────
    # Message handlers
    # ──────────────────────────────────────────────────────────────────

    async def _startup(self, payload: dict) -> None:
        """
        Initialize all managers and configure them.
        This is the first message processed, blocking all subsequent messages.

        Note: Environment variables are already set by ConversationManagerState.set_details()
        when the StartupEvent arrives, so we don't duplicate that logic here.
        """
        print("[ManagersWorker] Processing startup")

        async with self._init_lock:
            start_time = perf_counter()
            if self._initialized:
                print("[ManagersWorker] Already initialized, skipping")
                return

            try:
                # 0. Initialize unity
                print("[ManagersWorker] Initializing unity...")
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
                            "user_whatsapp_number": payload["user_whatsapp_number"]
                            or None,
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
                    open_predicate=lambda e: e.payload.get("role", "")
                    == "tool_use start",
                    close_predicate=lambda e: e.payload.get("role", "")
                    == "tool_use end",
                    key_fn=lambda e: e.payload.get("handle_id", ""),
                )
                bus_events_task = asyncio.create_task(self._get_bus_events())
                print("[ManagersWorker] EventBus configured")

                # 2. Initialize ContactManager and get contacts
                print("[ManagersWorker] Initializing ContactManager...")
                self._contact_manager = ContactManager()

                # clear rolling summary
                # contacts = self._contact_manager._filter_contacts().get("contacts", [])
                # print("got contacts", contacts)
                # for c in contacts:
                #     self._contact_manager._update_contact(contact_id=c.contact_id, rolling_summary="")

                contacts_task = asyncio.create_task(self._get_contacts())
                await asyncio.gather(bus_events_task, contacts_task)
                print("[ManagersWorker] ContactManager initialized")

                # 3. Initialize TranscriptManager with ContactManager
                print("[ManagersWorker] Initializing TranscriptManager...")
                self._transcript_manager = TranscriptManager(
                    contact_manager=self._contact_manager,
                )
                print("[ManagersWorker] TranscriptManager initialized")

                # 4. Configure TranscriptManager logger with auth header
                if api_key:
                    self._transcript_manager._get_logger().session.headers[
                        "Authorization"
                    ] = f"Bearer {api_key}"
                    print("[ManagersWorker] TranscriptManager logger configured")

                # TODO: Initialize other managers (Conductor, etc.) here
                print("[ManagersWorker] Initializing MemoryManager...")
                self._memory_manager = MemoryManager(
                    transcript_manager=self._transcript_manager,
                    contact_manager=self._contact_manager,
                )
                print("[ManagersWorker] MemoryManager initialized")

                # 5. Initialize ConversationManager
                print("[ManagersWorker] Initializing ConversationManagerHandle...")
                conversation_manager_handle = ConversationManagerHandle(
                    event_broker=self._event_broker,
                    conversation_id=os.getenv("ASSISTANT_ID", "default-assistant"),
                    contact_id="1",
                    transcript_manager=self._transcript_manager,
                )
                print("[ManagersWorker] ConversationManagerHandle initialized")

                # 6. Initialize Conductor with existing managers
                print("[ManagersWorker] Initializing Conductor...")
                try:
                    self._conductor = Conductor(
                        contact_manager=self._contact_manager,
                        transcript_manager=self._transcript_manager,
                        conversation_manager=conversation_manager_handle,
                    )
                    print("[ManagersWorker] Conductor initialized")
                except Exception as e:
                    print(f"[ManagersWorker] Error initializing Conductor: {e}")

                self._initialized = True
                print("[ManagersWorker] Initialization complete")

            except Exception as e:
                print(f"[ManagersWorker] Error during initialization: {e}")

            await self._event_broker.publish(
                self._publish_channel,
                ManagersStartupResponse(initialized=self._initialized).to_json(),
            )
            print(
                "[ManagersWorker] Initialization complete in "
                f"{perf_counter() - start_time:.2f} seconds"
            )

    async def _get_bus_events(self) -> None:
        """Get events from EventBus."""
        bus_events = await EVENT_BUS.search(filter='type == "Comms"', limit=50)
        await self._event_broker.publish(
            self._publish_channel,
            GetBusEventsResponse(
                events=[Event.from_bus_event(e).to_dict() for e in bus_events][::-1],
            ).to_json(),
        )

    async def _publish_bus_event(self, event: Event) -> None:
        """Publish an event to the EventBus."""
        event_dict = event.to_dict()["payload"]["event"]
        event_name = event_dict["event_name"]
        bus_event = Event.from_dict(event_dict).to_bus_event()
        bus_event.payload.pop("api_key", None)
        bus_event.payload.pop("message_id", None)
        print("Publishing bus event", event_name)
        await EVENT_BUS.publish(bus_event)

    async def _log_message(self, event: LogMessageRequest) -> None:
        """Log a message via TranscriptManager."""
        try:
            print(f"[ManagersWorker] Logging message: {event.to_dict()}")
            medium = event.medium or "unify_message"
            sender_id = int(event.sender_id)
            receiver_ids = [int(r) for r in (event.receiver_ids or [])]
            content = event.content
            timestamp = event.timestamp
            exchange_id = event.exchange_id
            # call_utterance_timestamp = event.call_utterance_timestamp
            # call_url = event.call_url
            metadata = getattr(event, "metadata", None)

            # Log the message
            messages = self._transcript_manager.log_messages(
                {
                    "medium": medium,
                    "sender_id": sender_id,
                    "receiver_ids": receiver_ids,
                    "timestamp": timestamp,
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
                await self._event_broker.publish(
                    self._publish_channel,
                    LogMessageResponse(
                        medium=medium,
                        exchange_id=message.exchange_id,
                    ).to_json(),
                )
                print(f"[ManagersWorker] Published exchange_id {message.exchange_id}")

        except Exception as e:
            print(f"[ManagersWorker] Error logging message: {e}")

    async def _get_contacts(self) -> None:
        """Fetch all contacts and publish back."""
        try:
            # Get all contacts from ContactManager and convert to dict
            rows = self._contact_manager._filter_contacts().get("contacts", [])
            contacts = [c.model_dump() for c in rows]

            # Publish reply as Event envelope
            await self._event_broker.publish(
                self._publish_channel,
                GetContactsResponse(contacts=contacts).to_json(),
            )

            print(f"[ManagersWorker] Fetched {len(contacts)} contacts")

            return contacts

        except Exception as e:
            print(f"[ManagersWorker] Error fetching contacts: {e}")

    async def _get_contact_by_id(self, contact_id: int) -> None:
        """Fetch a single contact by ID and publish back."""
        try:
            # get contact info from ContactManager
            contacts = self._contact_manager.get_contact_info([contact_id])

            if contact_id in contacts:
                # publish updated contact details
                contact = contacts[contact_id]
                await self._event_broker.publish(
                    self._publish_channel,
                    ContactInfoResponse(contact_details=contact).to_json(),
                )
                print(f"[ManagersWorker] Fetched contact {contact_id}")
            else:
                print(f"[ManagersWorker] Contact {contact_id} not found")

        except Exception as e:
            print(f"[ManagersWorker] Error fetching contact: {e}")

    async def _create_contact(self, contact: dict) -> None:
        """Create a contact in the ContactManager."""
        try:
            self._contact_manager._create_contact(
                first_name=contact["first_name"],
                surname=contact["surname"],
                phone_number=contact["phone_number"],
                email_address=contact["email_address"],
            )
            print(f"[ManagersWorker] Created contact: {contact}")

            # return back the list of updated contacts
            await self._get_contacts()
        except Exception as e:
            print(f"[ManagersWorker] Error creating contact: {e}")

    async def _update_contact(self, contact: dict) -> None:
        """Update a contact in the ContactManager."""
        try:
            self._contact_manager._update_contact(
                contact_id=contact["contact_id"],
                first_name=contact["first_name"],
                surname=contact["surname"],
                phone_number=contact["phone_number"],
                email_address=contact["email_address"],
            )
            await self._get_contacts()
        except Exception as e:
            print(f"[ManagersWorker] Error updating contact: {e}")

    async def _update_contact_rolling_summary(
        self,
        contacts_ids: list[int],
        transcripts: list[str],
    ):
        print(transcripts)
        tasks = [
            self._memory_manager.update_contact_rolling_summary(t, contact_id=cid)
            for cid, t in zip(contacts_ids, transcripts)
        ]
        try:
            await asyncio.gather(*tasks)
            print("[ManagersWorker] Contact rolling summary updated")
        except Exception as e:
            print(f"[ManagersWorker] Error updating contact rolling summary: {e}")

    async def _conductor_watch_result(
        self, handle_id: int, handle: SteerableToolHandle
    ) -> None:
        """Await final result and publish completion (or failure), then cleanup."""
        # await result
        try:
            result = await handle.result()
        except Exception as e:
            result = f"Error getting conductor result: {e}"
            print(f"[ManagersWorker] {result}")
        await self._event_broker.publish(
            self._publish_channel,
            ConductorResult(
                handle_id=handle_id,
                success=False if "Error" in result else True,
                result=result,
            ).to_json(),
        )

        # cleanup registry entry
        self._handle_registry.pop(handle_id, None)
        self._handle_meta.pop(handle_id, None)

    async def _conductor_watch_notifications(
        self, handle_id: int, handle: SteerableToolHandle
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
            await self._event_broker.publish(
                self._publish_channel,
                ConductorNotification(
                    handle_id=handle_id,
                    response=msg,
                ).to_json(),
            )

    async def _conductor_watch_clarifications(
        self, handle_id: int, handle: SteerableToolHandle
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
            await self._event_broker.publish(
                self._publish_channel,
                ConductorClarificationRequest(
                    handle_id=handle_id,
                    query=q,
                    call_id=call_id,
                ).to_json(),
            )

    async def _handle_conductor_request(self, event: ConductorRequest) -> None:
        """Start a Conductor ask/request, store handle, and publish started."""
        if event.action_name == "ask":
            handle = await self._conductor.ask(
                event.query,
                _parent_chat_context=event.parent_chat_context,
            )
        else:
            handle = await self._conductor.request(
                event.query,
                _parent_chat_context=event.parent_chat_context,
            )

        # allocate handle id and register
        handle_id = self._next_handle_id
        self._next_handle_id += 1
        self._handle_registry[handle_id] = {
            "handle": handle,
            "query": event.query,
            "handle_actions": [],
        }

        # publish started
        await self._event_broker.publish(
            self._publish_channel,
            ConductorResponse(
                handle_id=handle_id,
                action_name=event.action_name,
                query=event.query,
                response=f"Started: {event.query}",
            ).to_json(),
        )

        # spawn watchers
        asyncio.create_task(self._conductor_watch_result(handle_id, handle))
        asyncio.create_task(self._conductor_watch_notifications(handle_id, handle))
        asyncio.create_task(self._conductor_watch_clarifications(handle_id, handle))

    def _register_handle_action(
        self, handle_id: int, action_name: str, query: str
    ) -> None:
        """Register a handle action."""
        handle_data = self._handle_registry.get(handle_id)
        if not handle_data:
            print(f"[ManagersWorker] Unknown handle_id={handle_id} for action")
            return

        # record intervention
        handle_data["handle_actions"].append(
            {"action_name": action_name, "query": query}
        )
        return handle_data["handle"]

    async def _handle_conductor_clarification_response(
        self, event: ConductorClarificationResponse
    ) -> None:
        """Handle a Conductor clarification response."""
        handle: SteerableToolHandle = self._register_handle_action(
            event.handle_id, event.action_name, event.query
        )

        # perform intervention
        try:
            await handle.answer_clarification(event.call_id, event.response)
        except Exception as e:
            print(f"[ManagersWorker] Error answering clarification: {e}")
            return

    async def _handle_conductor_handle_request(
        self, event: ConductorHandleRequest
    ) -> None:
        handle: SteerableToolHandle = self._register_handle_action(
            event.handle_id, event.action_name, event.query
        )

        # perform intervention
        result = ""
        try:
            match event.action_name:
                case "ask":
                    ask_handle = await handle.ask(
                        event.query,
                        parent_chat_context_cont=event.parent_chat_context,
                    )
                    result = await ask_handle.result()
                case "interject":
                    await handle.interject(
                        event.query,
                        parent_chat_context_cont=event.parent_chat_context,
                    )
                    result = "Handle Interjected"
                case "stop":
                    handle.stop(reason=event.query)
                    result = "Handle Stopped"
                case "pause":
                    handle.pause()
                    result = "Handle Paused"
                case "resume":
                    handle.resume()
                    result = "Handle Resumed"
                case "done":
                    done_result = handle.done()
                    result = "Handle Done" if done_result else "Handle Not Done"
                case _:
                    print(
                        f"[ManagersWorker] Unknown action_name={event.action_name} for intervention"
                    )
                    return
        except Exception as e:
            result = f"Error in conductor handle request: {e}"
            print(f"[ManagersWorker] {result}")

        # publish response
        await self._event_broker.publish(
            self._publish_channel,
            ConductorHandleResponse(
                handle_id=event.handle_id,
                action_name=event.action_name,
                query=event.query,
                response=f"Intervened: {event.action_name} {result}",
            ).to_json(),
        )

    # ──────────────────────────────────────────────────────────────────
    # Message Routing Workers (ingress/egress)
    # ──────────────────────────────────────────────────────────────────

    # Single-level routing: return (primary queue key, readiness checker)
    def _route_for(self, event: Event) -> tuple[str, Callable[[], bool]]:
        if isinstance(event, ManagersStartupRequest):
            return ("initialized", lambda: True)
        if isinstance(
            event,
            (
                GetContactsRequest,
                ContactInfoRequest,
                CreateContactRequest,
                UpdateContactRequest,
            ),
        ):
            return ("contact_manager", lambda: self._contact_manager is not None)
        if isinstance(event, LogMessageRequest):
            return ("transcript_manager", lambda: self._transcript_manager is not None)
        if isinstance(event, UpdateContactRollingSummaryRequest):
            return ("memory_manager", lambda: self._memory_manager is not None)
        if isinstance(
            event,
            (ConductorRequest, ConductorHandleRequest, ConductorClarificationResponse),
        ):
            return ("conductor", lambda: self._conductor is not None)
        if isinstance(event, (GetBusEventsRequest, PublishBusEventRequest)):
            return ("initialized", lambda: self._initialized)
        return ("initialized", lambda: self._initialized)

    async def _egress_worker(self, req_key: str) -> None:
        print(f"[ManagersWorker] Egress worker for {req_key} starting")
        q = self._queues[req_key]
        while not self._stop_event.is_set():
            try:
                ev, ready = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                waited = 0
                while not ready() and not self._stop_event.is_set():
                    if waited % 5 == 0:
                        print(
                            f"[ManagersWorker] Waiting for {req_key} readiness... {waited}s",
                        )
                    await asyncio.sleep(1)
                    waited += 1
                # Inline dispatch here to reduce call depth
                match ev:
                    case ManagersStartupRequest():
                        asyncio.create_task(self._startup(ev.to_dict()["payload"]))
                    case GetBusEventsRequest():
                        asyncio.create_task(self._get_bus_events())
                    case PublishBusEventRequest():
                        asyncio.create_task(self._publish_bus_event(ev))
                    case LogMessageRequest():
                        asyncio.create_task(self._log_message(ev))
                    case GetContactsRequest():
                        asyncio.create_task(self._get_contacts())
                    case ContactInfoRequest():
                        asyncio.create_task(self._get_contact_by_id(ev.contact_id))
                    case CreateContactRequest():
                        asyncio.create_task(
                            self._create_contact(ev.to_dict()["payload"])
                        )
                    case UpdateContactRequest():
                        asyncio.create_task(
                            self._update_contact(ev.to_dict()["payload"])
                        )
                    case UpdateContactRollingSummaryRequest():
                        asyncio.create_task(
                            self._update_contact_rolling_summary(
                                ev.contacts_ids, ev.transcripts
                            )
                        )
                    case ConductorRequest():
                        asyncio.create_task(self._handle_conductor_request(ev))
                    case ConductorHandleRequest():
                        asyncio.create_task(self._handle_conductor_handle_request(ev))
                    case ConductorClarificationResponse():
                        asyncio.create_task(
                            self._handle_conductor_clarification_response(ev)
                        )
                    case _:
                        print(
                            f"[ManagersWorker] Unknown event: {ev.to_dict()['event_name']}"
                        )
            except Exception as e:
                print(f"[ManagersWorker] Error dispatching {req_key} event: {e}")
            finally:
                with contextlib.suppress(Exception):
                    q.task_done()

    async def _ingress_router(self) -> None:
        """Worker task that processes messages from Redis and routes into per-requirement queues."""
        print("[ManagersWorker] Ingress router started")

        while not self._stop_event.is_set():
            try:
                # Wait for message with timeout to check stop event
                try:
                    msg = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Route event into the appropriate per-requirement queue; workers will drain
                try:
                    key, ready = self._route_for(msg)
                    print(
                        f"[ManagersWorker] Routed {msg.to_dict()['event_name']} to {key} queue",
                    )
                    self._queues[key].put_nowait((msg, ready))
                except Exception as e:
                    print(f"[ManagersWorker] Failed to enqueue event: {e}")

            except Exception as e:
                print(f"[ManagersWorker] Error in ingress router: {e}")

        print("[ManagersWorker] Ingress router stopped")

    async def wait_for_events(self) -> None:
        """
        Subscribe to Redis Pub/Sub and enqueue messages.
        A separate task processes the queue to ensure ordering.
        """
        if self._event_broker is None:
            raise RuntimeError(
                "[ManagersWorker] _event_broker must be set before wait_for_events()",
            )
        print("[ManagersWorker] Flag", self._initialized)
        print("[ManagersWorker] Starting to wait for events")
        print(f"[ManagersWorker] Subscribe channel: {self._subscribe_channel}")
        print(f"[ManagersWorker] Publish channel: {self._publish_channel}")

        # Start ingress router (Redis → internal queues) and per-requirement workers
        ingress_task = asyncio.create_task(self._ingress_router())
        for key in self._queues.keys():
            self._queue_consumers.append(asyncio.create_task(self._egress_worker(key)))

        try:
            async with self._event_broker.pubsub() as pubsub:
                await pubsub.subscribe(self._subscribe_channel)
                print(f"[ManagersWorker] Subscribed to {self._subscribe_channel}")

                while not self._stop_event.is_set():
                    try:
                        msg = await pubsub.get_message(
                            timeout=2,
                            ignore_subscribe_messages=True,
                        )

                        # Parse Event from JSON envelope and enqueue
                        if msg is not None:
                            try:
                                event = Event.from_json(msg["data"])  # type: ignore[arg-type]
                                print(
                                    f"[ManagersWorker] Enqueued event: {event.to_dict()['event_name']}",
                                )
                                await self._message_queue.put(event)
                            except Exception as parse_err:
                                print(
                                    f"[ManagersWorker] Failed to parse Event from message: {parse_err}",
                                )

                    except Exception as e:
                        print(f"[ManagersWorker] Error receiving message: {e}")
                        await asyncio.sleep(1)

        finally:
            # Stop processor
            self._stop_event.set()
            await self._event_broker.aclose()
            await ingress_task
            # Cancel consumer tasks and wait for them to exit
            for t in self._queue_consumers:
                t.cancel()
            for t in self._queue_consumers:
                with contextlib.suppress(Exception):
                    await t
            print("[ManagersWorker] Worker stopped")

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._stop_event.set()
