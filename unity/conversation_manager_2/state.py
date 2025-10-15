from datetime import datetime
from dataclasses import dataclass
import os
import logging
from typing import Literal, Optional
from collections import deque

from pydantic import Field
from unity.conversation_manager_2.actions import build_dynamic_response_models
from unity.conversation_manager_2.new_events import *
from unity.contact_manager.types.contact import Contact as ContactType
from unity.transcript_manager.types.message import UNASSIGNED
from unity.conversation_manager_2.event_broker import get_event_broker

logger = logging.getLogger(__name__)


class Contact(ContactType):
    is_boss: bool = False
    threads: dict[str, deque] = Field(
        default_factory=lambda: {
            "sms": deque(maxlen=50),
            "email": deque(maxlen=50),
            "phone": deque(maxlen=50),
            "unify_message": deque(maxlen=50),
        },
    )

    @property
    def full_name(self):
        name = self.first_name + " " + self.surname if self.surname else ""
        return name.strip()


@dataclass
class Message:
    name: str
    content: str
    timestamp: datetime


@dataclass
class EmailMessage:
    name: str
    subject: str
    body: str
    timestamp: datetime


@dataclass
class Notification:
    type: str
    content: str
    timestamp: datetime
    pinned: bool = False
    interjection_id: Optional[str] = None


class ConversationManagerState:
    def __init__(
        self,
        job_name: str,
        user_id: str,
        assistant_id: str,
        assistant_name: str,
        assistant_age: str,
        assistant_region: str,
        assistant_about: str,
        voice_provider: str,
        voice_id: str,
        assistant_number: str,
        assistant_email: str,
        user_name: str,
        user_number: str,
        user_email: str,
        user_whatsapp_number: str,
    ):

        # These should not be harcoded
        self.phone_contacts_map = {
            # "+12697784020": Contact(
            #     "1",
            #     "Yasser",
            #     "Ahmed",
            #     True,
            #     "+12697784020",
            #     "yasser@unify.ai",
            # )
        }
        self.email_contacts_map = {
            # "yasser@unify.ai": Contact(
            #     "1",
            #     "Yasser",
            #     "Ahmed",
            #     True,
            #     "+12697784020",
            #     "yasser@unify.ai",
            # )
        }

        self.inverted_contacts_map = {
            v.contact_id: v for v in self.phone_contacts_map.values()
        }

        self.active_conversations: dict[str, Contact] = {}

        self.notifs: list[Notification] = []

        self.mode: Literal["text", "call", "gmeet"] = "text"
        self.events = []
        self.last_snapshot_time = datetime.now()
        self.phone_contact: Optional[Contact] = None

        # call details
        self.call_exchange_id = UNASSIGNED
        self.call_start_timestamp = None
        self.conference_name = ""

        # assistant details
        self.job_name = job_name
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.assistant_name = assistant_name
        self.assistant_age = assistant_age
        self.assistant_region = assistant_region
        self.assistant_about = assistant_about
        self.voice_provider = voice_provider
        self.voice_id = voice_id

        # contact data
        self.assistant_number = assistant_number
        self.assistant_email = assistant_email
        self.user_name = user_name
        self.user_number = user_number
        self.user_email = user_email
        self.user_whatsapp_number = user_whatsapp_number

        # initialization state
        self.initialized: bool = False
        self.chat_history = []
        self.event_broker = get_event_broker()

        self.summarizing = False

        # dynamic response models
        self.dynamic_response_models = None
        print("assistant_id", self.assistant_id)
        if self.assistant_id:
            self.build_response_model()

    def update_state(self, event: Event):
        # log the event if it's loggable
        if event.__class__.loggable:
            self.events.append(event)
        match event:
            # startup events
            case ManagersStartupOutput() as e:
                if not e.initialized:
                    raise Exception("Managers failed to initialize")
                self.initialized = bool(e.initialized)
            case StartupEvent() as e:
                payload = e.to_dict()["payload"]
                self.set_details(payload)
            case AssistantUpdateEvent() as e:
                payload = e.to_dict()["payload"]
                self.set_details(payload)
                self.update_or_create_new_contact(
                    contact_id=0,
                    first_name=payload["assistant_name"],
                    surname="",
                    email_address=payload["assistant_email"],
                    phone_number=payload["assistant_number"],
                )

            case GetBusEventsOutput() as e:
                # TODO: should also grab the latest messages ~50 messages
                # and populate their contacts in the active conversations
                for ev in reversed(e.events):
                    if ev["event_name"] == "LLMInput":
                        print("found history")
                        self.chat_history = ev["payload"]["chat_history"]
                        break

            case UpdateContactRollingSummaryResponse() as e:
                print("clearing context...")
                for cid, rolling_summary in e.rolling_summaries:
                    self.active_conversations[cid].rolling_summary = rolling_summary
                self.chat_history = []
                self.summarizing = False

            # Handle steering notifications from external handles
            case NotificationInjectedEvent() as e:
                # Only process if it's for this conversation
                if e.target_conversation_id == self.assistant_id:
                    self.push_notif(
                        Notification(
                            type=e.source,
                            content=e.content,
                            timestamp=e.timestamp,
                            pinned=e.pinned,
                            interjection_id=e.interjection_id,
                        ),
                    )

            case NotificationUnpinnedEvent() as e:
                # Only process if it's for this conversation
                if e.target_conversation_id == self.assistant_id:
                    # Find and unpin the notification
                    for notif in self.notifs:
                        if notif.interjection_id == e.interjection_id:
                            notif.pinned = False
                            break

            case PhoneCallRecieved() as e:
                self.conference_name = e.conference_name

                # contact should always exist here.
                contact = self.get_contact(phone_number=e.contact)
                self.push_message(
                    contact,
                    "phone",
                    Message(contact.full_name, "<Phone call sent...>", e.timestamp),
                )
                self.push_notif(
                    Notification(
                        "comms",
                        f"Phone Call recieved from '{contact.full_name}'",
                        e.timestamp,
                    ),
                )

            # made by assistant
            case PhoneCallSent() as e:
                contact = self.get_contact(phone_number=e.contact)
                self.push_message(
                    contact,
                    "phone",
                    Message("You", "<Phone call sent...>", e.timestamp),
                )
                self.push_notif(
                    Notification(
                        "comms",
                        f"Phone Call Sent to '{contact.full_name}'",
                        e.timestamp,
                    ),
                )
            case PhoneCallStarted() as e:
                self.call_start_timestamp = e.timestamp
                self.mode = "call"
                contact = self.get_contact(phone_number=e.contact)
                self.phone_contact = contact
                self.push_message(
                    contact,
                    "phone",
                    Message(contact.full_name, "<Phone call started...>", e.timestamp),
                )
                self.push_notif(
                    Notification(
                        "comms",
                        f"Phone Call started with '{contact.full_name}'",
                        e.timestamp,
                    ),
                )

            case PhoneUtterance() as e:
                contact = self.get_contact(phone_number=e.contact)
                self.push_message(
                    contact,
                    "phone",
                    Message(contact.full_name, e.content, e.timestamp),
                )
                self.push_notif(
                    Notification(
                        "comms",
                        f"Phone Call Utterance recieved from '{contact.full_name}'",
                        e.timestamp,
                    ),
                )
            # made by assistant
            case AssistantPhoneUtterance() as e:
                contact = self.get_contact(phone_number=e.contact)
                self.push_message(
                    contact,
                    "phone",
                    Message("You", e.content, e.timestamp),
                )
                self.push_notif(
                    Notification(
                        "comms",
                        f"Phone Call Utterance sent to '{contact.full_name}'",
                        e.timestamp,
                    ),
                )
            case PhoneCallEnded() as e:
                contact = self.get_contact(phone_number=e.contact)
                self.push_message(
                    contact,
                    "phone",
                    Message(contact.full_name, "<Phone Call Ended...>", e.timestamp),
                )
                self.push_notif(
                    Notification(
                        "comms",
                        f"Phone Call Ended with '{contact.full_name}'",
                        e.timestamp,
                    ),
                )
                self.phone_contact = None

            case SMSRecieved() as e:
                contact = self.get_contact(phone_number=e.contact)
                self.push_message(
                    contact,
                    "sms",
                    Message(contact.full_name, e.content, e.timestamp),
                )
                self.push_notif(
                    Notification(
                        "comms",
                        f"SMS recieved from '{contact.full_name}'",
                        e.timestamp,
                    ),
                )
            # made by assistant
            case SMSSent() as e:
                contact = self.get_contact(phone_number=e.contact)
                self.push_message(
                    contact,
                    "sms",
                    Message("You", e.content, e.timestamp),
                )
                self.push_notif(
                    Notification(
                        "comms",
                        f"SMS sent to '{contact.full_name}'",
                        e.timestamp,
                    ),
                )
            case UnifyMessageRecieved() as e:
                contact = self.get_contact(contact_id=int(e.contact))
                self.push_message(
                    contact,
                    "unify_message",
                    Message(contact.full_name, e.content, e.timestamp),
                )
                self.push_notif(
                    Notification(
                        "comms",
                        f"Unify message recieved from '{contact.full_name}'",
                        e.timestamp,
                    ),
                )
            case UnifyMessageSent() as e:
                contact = self.get_contact(contact_id=int(e.contact))
                self.push_message(
                    contact,
                    "unify_message",
                    Message("You", e.content, e.timestamp),
                )
                self.push_notif(
                    Notification(
                        "comms",
                        f"Unify message sent to '{contact.full_name}'",
                        e.timestamp,
                    ),
                )
            case EmailRecieved() as e:
                contact = self.get_contact(email_address=e.contact)
                self.push_message(
                    contact,
                    "email",
                    EmailMessage(contact.full_name, e.subject, e.body, e.timestamp),
                )
                self.push_notif(
                    Notification(
                        "comms",
                        f"Email recieved from '{contact.full_name}'",
                        e.timestamp,
                    ),
                )
            # made by assistant
            case EmailSent() as e:
                contact = self.get_contact(email_address=e.contact)
                self.push_message(
                    contact,
                    "email",
                    EmailMessage("You", e.subject, e.body, e.timestamp),
                )
                self.push_notif(
                    Notification(
                        "comms",
                        f"Email sent to '{contact.full_name}'",
                        e.timestamp,
                    ),
                )
            # made by assistant
            case Error() as e:
                self.push_notif(Notification("error", e.message, e.timestamp))

            case GetContactsOutput() as e:
                for c in e.contacts:
                    self.create_new_contact(**c)

            case LogMessageOutput() as e:
                # ToDo: Get this working for email and whatsapp as well
                # Email: Replying to the same thread
                # Whatsapp: Managing different kinds of chat such as groups, etc.
                if e.medium == "phone_call" and self.call_exchange_id == UNASSIGNED:
                    self.call_exchange_id = e.exchange_id

    def snapshot(self):
        self._current_snapshot_time = datetime.now()

    def commit(self):
        """marks that an llm run with done successfully"""
        self.last_snapshot_time = self._current_snapshot_time

    def get_state_for_llm(self) -> list[dict]:
        active_convs = "\n\n".join(
            self._render_contact(c) for c in self.active_conversations.values()
        )
        notif = self._render_notifs()
        state = f"<notifications>\n{self._add_spaces(notif)}\n</notifications>\n<active_conversations>\n{self._add_spaces(active_convs)}\n</active_conversations>"

        return state

    def set_details(self, payload: dict):
        """Populate assistant/user/voice details and update environment variables."""
        self.user_id = payload["user_id"]
        self.assistant_id = payload["assistant_id"]
        self.assistant_name = payload["assistant_name"]
        self.assistant_age = payload["assistant_age"]
        self.assistant_region = payload["assistant_region"]
        self.assistant_about = payload["assistant_about"]
        self.assistant_number = payload["assistant_number"]
        self.assistant_email = payload["assistant_email"]
        self.user_name = payload["user_name"]
        self.user_number = payload["user_number"]
        self.user_whatsapp_number = payload["user_whatsapp_number"]
        self.user_email = payload["user_email"]
        self.current_user = {
            "user_name": self.user_name,
            "user_number": self.user_number,
            "user_whatsapp_number": self.user_whatsapp_number,
            "user_email": self.user_email,
        }
        self.voice_provider = payload["voice_provider"]
        self.voice_id = payload["voice_id"]
        self.build_response_model()
        if payload.pop("api_key", None):
            os.environ["UNIFY_KEY"] = payload.pop("api_key")
        os.environ["USER_ID"] = self.user_id
        os.environ["USER_NAME"] = self.user_name
        os.environ["USER_NUMBER"] = self.user_number
        os.environ["USER_WHATSAPP_NUMBER"] = self.user_whatsapp_number
        os.environ["USER_EMAIL"] = self.user_email
        os.environ["ASSISTANT_NAME"] = self.assistant_name
        os.environ["ASSISTANT_NUMBER"] = self.assistant_number
        os.environ["ASSISTANT_EMAIL"] = self.assistant_email
        os.environ["VOICE_PROVIDER"] = self.voice_provider
        os.environ["VOICE_ID"] = self.voice_id

    def build_response_model(self):
        """Build dynamic response models based on available actions."""
        self.dynamic_response_models = build_dynamic_response_models(
            include_email=self.assistant_email not in [None, ""],
            include_sms=self.assistant_number not in [None, ""],
            include_call=self.assistant_number not in [None, ""],
        )
        available_actions = list(
            self.dynamic_response_models["call"].model_json_schema()["$defs"].keys(),
        )
        print("Dynamic response models built.")
        print(f"Available actions: {available_actions}")

    def get_details(self) -> dict:
        return {
            "job_name": self.job_name,
            "user_id": self.user_id,
            "assistant_id": self.assistant_id,
            "user_name": self.user_name,
            "assistant_name": self.assistant_name,
            "user_number": self.user_number,
            "user_whatsapp_number": self.user_whatsapp_number,
            "assistant_number": self.assistant_number,
            "user_email": self.user_email,
            "assistant_email": self.assistant_email,
        }

    def push_message(
        self,
        contact: Contact,
        thread: Literal["sms", "email", "phone", "unify_message"],
        message: Message | EmailMessage,
    ):
        if contact.contact_id not in self.active_conversations:
            self.active_conversations[contact.contact_id] = contact
        self.active_conversations[contact.contact_id].threads[thread].append(message)
        # hardcoded for now
        # if self.get_total_messages(contact) >= MAX_MSGS_PER_CONTACT:
        #     # asyncio.create_task(...)
        #     print("SHOULD UPDATE ROLLING SUMMARY!!!")
        #     event = UpdateContactRollingSummary(
        #         contact_id=int(contact.contact_id),
        #         transcripts=self._render_contact_threads(contact)
        #     )
        #     asyncio.create_task(self.event_broker.publish("app:managers:input", event.to_json()))

    def get_total_messages(self, contact: Contact):
        return sum(len(t) for t in contact.threads.values())

    def push_notif(self, notif: Notification):
        self.notifs.append(notif)

    # contacts helpers
    def get_contact(
        self,
        contact_id: Optional[int] = None,
        phone_number: Optional[str] = None,
        email_address: Optional[str] = None,
    ) -> Contact:
        """returns the new contact and whether they were newly created or not."""
        if not (contact_id or phone_number or email_address):
            raise Exception(
                "at least one contact_id, phone number or email_address must be provided to infer contact",
            )
        if contact_id:
            contact = self.inverted_contacts_map.get(contact_id)
            if contact:
                return contact

        if phone_number:
            contact = self.phone_contacts_map.get(phone_number)
            if contact:
                return contact
        else:
            contact = self.email_contacts_map.get(email_address)
            if contact:
                return contact

    def create_new_contact(
        self,
        contact_id: int,
        first_name: str,
        surname: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        whatsapp_number: Optional[str] = None,
        bio: Optional[str] = None,
        rolling_summary: Optional[str] = None,
        respond_to: bool = False,
        response_policy: Optional[str] = None,
    ):
        contact = Contact(
            contact_id=contact_id,
            first_name=first_name,
            surname=surname,
            email_address=email_address,
            phone_number=phone_number,
            whatsapp_number=whatsapp_number,
            bio=bio,
            rolling_summary=rolling_summary,
            respond_to=respond_to,
            response_policy=response_policy,
            is_boss=str(contact_id) == "1",
        )
        self.inverted_contacts_map[contact_id] = contact
        if email_address:
            self.email_contacts_map[email_address] = contact
        if phone_number:
            self.phone_contacts_map[phone_number] = contact
        return contact

    def update_or_create_new_contact(
        self,
        contact_id: int,
        first_name: str,
        surname: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        whatsapp_number: Optional[str] = None,
        bio: Optional[str] = None,
        rolling_summary: Optional[str] = None,
        respond_to: bool = False,
        response_policy: Optional[str] = None,
    ):
        contact = None
        print("curernt contacts", self.inverted_contacts_map)
        if contact_id != -1:  # update branch
            contact = self.get_contact(
                contact_id,
                phone_number,
                email_address=email_address,
            )
            if contact:
                # TODO: pop old emails or phone numbers if they exist
                if phone_number and (contact.phone_number != phone_number):
                    contact.phone_number = phone_number
                    self.phone_contacts_map[phone_number] = contact

                if email_address and (contact.email_address != email_address):
                    contact.email_address = email_address
                    self.email_contacts_map[email_address] = contact
            else:
                self.create_new_contact(
                    contact_id,
                    first_name,
                    surname,
                    email_address,
                    phone_number,
                    whatsapp_number,
                    bio,
                    rolling_summary,
                    respond_to,
                    response_policy,
                )
        else:
            new_contact = self.create_new_contact(
                # must make sure ids are aligned with the database here actually..
                str(len(self.phone_contacts_map) + 1),
                first_name,
                surname,
                email_address,
                phone_number,
                whatsapp_number,
                bio,
                rolling_summary,
                respond_to,
                response_policy,
            )
            self.active_conversations[new_contact.contact_id] = new_contact
            contact = new_contact
        # self.push_message("comms", f"Adding {contact.full_name} to active conversations.")
        return contact

    # rendering methods
    def _render_contact(self, contact: Contact):
        threads = []
        on_phone = self.phone_contact is contact
        details = f"""
<bio>
{self._add_spaces(contact.bio or "No bio set")}
</bio>

<response_policy>
{self._add_spaces(contact.response_policy or "No response poilcy set")}
</response_policy>

<rolling_summary>
{self._add_spaces(contact.rolling_summary or "No rolling summary yet")}
</rolling_summary>""".strip()
        for t_name, t in contact.threads.items():
            if t:
                threads.append(self._render_thread(t_name, t))
        threads = "\n\n".join(threads)
        return f"""
<contact contact_id="{contact.contact_id}" first_name="{contact.first_name}" surname="{contact.surname}" is_boss="{contact.is_boss}" phone_number="{contact.phone_number or ""}" email_address="{contact.email_address or ""}" on_phone="{on_phone}">
    <contact_details>
{self._add_spaces(details, 8)}
    </contact_details>

    <threads>
{self._add_spaces(threads, 8)}
    </threads>
</contact>""".strip()

    def _render_thread_message(self, message: Message | EmailMessage):
        is_new = self.last_snapshot_time < message.timestamp

        if isinstance(message, EmailMessage):
            return f"""{'**NEW**' if is_new else ""} [{message.name} @ {message.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}]:
Subject: {message.subject}
Body:
{message.body}
"""
        return f"""{'**NEW**' if is_new else ""} [{message.name} @ {message.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}]: {message.content}"""

    def _render_thread(self, thread_name, thread: list[dict]):
        thread_content = "\n".join(
            self._render_thread_message(m) for m in list(thread)[-5:]
        )
        # thread_content = thread_content.strip()
        return f"""
<{thread_name}>
{self._add_spaces(thread_content)}
</{thread_name}>""".strip()

    def _render_notifs(self):
        # Count notifications for debugging
        pinned_notifs = [n for n in self.notifs if n.pinned]
        regular_notifs = [n for n in self.notifs if not n.pinned and n.timestamp > self.last_snapshot_time]
        
        logger.debug(
            f"📋 Rendering notifications: {len(pinned_notifs)} pinned (always visible), "
            f"{len(regular_notifs)} regular (new since last commit)"
        )
        
        # Render pinned notifications (always visible)
        pinned = "\n".join([
            f"""[PINNED {n.type.title()} @ {n.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}] {n.content}"""
            for n in pinned_notifs
        ])
        
        # Render regular notifications (only new ones)
        regular = "\n".join([
            f"""[{n.type.title()} Notification @ {n.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}] {n.content}"""
            for n in regular_notifs
        ])
        
        # Debug log what's being shown to the LLM
        if pinned:
            for n in pinned_notifs:
                logger.debug(f"  📌 PINNED (ID: {n.interjection_id}): {n.content[:60]}...")
        if regular:
            for n in regular_notifs:
                logger.debug(f"  📝 REGULAR (ID: {n.interjection_id}): {n.content[:60]}...")
        if not pinned and not regular:
            logger.debug("  (No notifications to render)")
        
        # Combine with separator if both exist
        if pinned and regular:
            return f"{pinned}\n\n{regular}"
        return pinned or regular

    def _render_contact_threads(self, contact: Contact):
        threads = []
        for t_name, t in contact.threads.items():
            if t:
                threads.append(self._render_thread(t_name, t))
        threads = "\n\n".join(threads)
        return threads

    def _add_spaces(self, string: str, num_spaces: int = 4):
        ls = string.split("\n")
        return "\n".join(num_spaces * " " + l for l in ls)
