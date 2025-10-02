from datetime import datetime
from dataclasses import dataclass, field
from typing import Literal, Optional
from collections import deque

from unity.conversation_manager_2.new_events import *


@dataclass
class Contact:
    id: str = "-1"
    first_name: str = None
    last_name: str = None
    is_boss: bool = False
    phone_number: str = None
    email: str = None

    threads: dict[str, deque] = field(
        default_factory=lambda: {
            "sms": deque(maxlen=5),
            "email": deque(maxlen=5),
            "phone": deque(maxlen=5),
        }
    )

    @property
    def full_name(self):
        name = self.first_name + " " + self.last_name if self.last_name else ""
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


class ConversationManagerState:
    def __init__(self):
        self.phone_contacts_map = {
            "+12697784020": Contact(
                "1",
                "Yasser",
                "Ahmed",
                True,
                "+12697784020",
                "yasser@unify.ai",
            )
        }
        self.email_contacts_map = {
            "yasser@unify.ai": Contact(
                "1",
                "Yasser",
                "Ahmed",
                True,
                "+12697784020",
                "yasser@unify.ai",
            )
        }

        self.inverted_contacts_map = {v.id: v for v in self.phone_contacts_map.values()}

        self.active_conversations: dict[str, Contact] = {}

        self.notifs: list[Notification] = []

        self.mode: Literal["text", "call", "gmeet"] = "text"
        self.events = []
        self.last_snapshot_time = datetime.now()
        self.phone_contact: Optional[Contact] = None

    def update_state(self, event: Event):
        self.events.append(event)
        match event:
            case PhoneCallRecieved() as e:
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
                    )
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
                    )
                )
            case PhoneCallStarted() as e:
                self.mode = "call"
                contact = self.get_contact(phone_number=e.contact)
                self.phone_contact = contact
                self.push_message(
                    contact,
                    "phone",
                    Message(contact.full_name, "<Phone call started...>"),
                )
                self.push_notif(
                    Notification(
                        "comms",
                        f"Phone Call started with '{contact.full_name}'",
                        e.timestamp,
                    )
                )

            case PhoneUtterance() as e:
                contact = self.get_contact(phone_number=e.contact)
                self.push_message(
                    contact, "phone", Message(contact.full_name, e.content, e.timestamp)
                )
                self.push_notif(
                    Notification(
                        "comms",
                        f"Phone Call Utterance recieved from '{contact.full_name}'",
                        e.timestamp,
                    )
                )
            # made by assistant
            case AssistantPhoneUtterance() as e:
                contact = self.get_contact(phone_number=e.contact)
                self.push_message(
                    contact, "phone", Message("You", e.content, e.timestamp)
                )
                self.push_notif(
                    Notification(
                        "comms",
                        f"Phone Call Utterance sent to '{contact.full_name}'",
                        e.timestamp,
                    )
                )
            case SMSRecieved() as e:
                contact = self.get_contact(phone_number=e.contact)
                self.push_message(
                    contact, "sms", Message(contact.full_name, e.content, e.timestamp)
                )
                self.push_notif(
                    Notification(
                        "comms", f"SMS recieved from '{contact.full_name}'", e.timestamp
                    )
                )
            # made by assistant
            case SMSSent() as e:
                contact = self.get_contact(phone_number=e.contact)
                self.push_message(
                    contact, "sms", Message("You", e.content, e.timestamp)
                )
                self.push_notif(
                    Notification(
                        "comms", f"SMS sent to '{contact.full_name}'", e.timestamp
                    )
                )
            case EmailRecieved() as e:
                contact = self.get_contact(email=e.contact)
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
                    )
                )
            # made by assistant
            case EmailSent() as e:
                contact = self.get_contact(email=e.contact)
                self.push_message(
                    contact,
                    "email",
                    EmailMessage("You", e.subject, e.body, e.timestamp),
                )
                self.push_notif(
                    Notification(
                        "comms", f"Email sent to '{contact.full_name}'", e.timestamp
                    )
                )
            # made by assistant
            case Error() as e:
                self.push_notif(Notification("error", e.message, e.timestamp))

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

    def push_message(
        self,
        contact: Contact,
        thread: Literal["sms", "email", "phone"],
        message: Message | EmailMessage,
    ):
        if contact.id not in self.active_conversations:
            self.active_conversations[contact.id] = contact
        self.active_conversations[contact.id].threads[thread].append(message)

    def push_notif(self, notif: Notification):
        self.notifs.append(notif)

    # contacts helpers
    def get_contact(
        self,
        id: Optional[str] = None,
        phone_number: Optional[str] = None,
        email: Optional[str] = None,
    ):
        """returns the new contact and whether they were newly created or not."""
        if not (id or phone_number or email):
            raise Exception(
                "at least one id, phone number or email must be provided to infer contact"
            )
        if id:
            contact = self.inverted_contacts_map.get(id)
        if phone_number:
            contact = self.phone_contacts_map.get(phone_number)
        else:
            contact = self.email_contacts_map.get(email)
        return contact

    def update_or_create_new_contact(
        self,
        id: str,
        first_name: str,
        last_name: Optional[str] = None,
        email: Optional[str] = None,
        phone_number: Optional[str] = None,
    ):
        contact = None
        if id != "-1":  # update branch
            contact = self.get_contact(id, phone_number, email)
            if phone_number and contact.phone_number != phone_number:
                contact.phone_number = phone_number
            if email and contact.email != email:
                contact.email = email
            contact = contact
        else:
            new_contact = Contact(
                str(len(self.phone_contacts_map) + 1),
                first_name,
                last_name,
                email,
                phone_number,
            )

            if phone_number:
                self.phone_contacts_map[phone_number] = new_contact
            if email:
                self.email_contacts_map[email] = new_contact
            self.inverted_contacts_map[new_contact.id] = new_contact

            self.active_conversations[new_contact.id] = new_contact
            contact = new_contact
        # self.push_message("comms", f"Adding {contact.full_name} to active conversations.")
        return contact

    # rendering methods
    def _render_contact(self, contact: Contact):
        threads = []
        for t_name, t in contact.threads.items():
            if t:
                threads.append(self._render_thread(t_name, t))
        threads = "\n\n".join(threads)
        return f"""
<contact id="{contact.id}" first_name="{contact.first_name}" last_name={contact.last_name} is_boss="{contact.is_boss}" phone_number="{contact.phone_number or ""}" email="{contact.email or ""}">
{self._add_spaces(threads)}
</contact>""".strip()

    def _render_thread_message(self, message: Message | EmailMessage):
        is_new = self.last_snapshot_time < message.timestamp

        if isinstance(message, EmailMessage) == "email":
            return f"""{'**NEW**' if is_new else ""} [{message.name} @ {message.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}]:
Subject: {message.subject}
Body:
{message.body}
"""
        return f"""{'**NEW**' if is_new else ""} [{message.name} @ {message.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}]: {message.content}"""

    def _render_thread(self, thread_name, thread: list[dict]):
        thread_content = "\n".join(self._render_thread_message(m) for m in thread)
        # thread_content = thread_content.strip()
        return f"""
<{thread_name}>
{self._add_spaces(thread_content)}
</{thread_name}>""".strip()

    def _render_notifs(self):
        return "\n".join(
            [
                f"""[{n.type.title()} Notification @ {n.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}] {n.content}"""
                for n in self.notifs
                if n.timestamp > self.last_snapshot_time
            ]
        )

    def _add_spaces(self, string: str, num_spaces: int = 4):
        ls = string.split("\n")
        return "\n".join(num_spaces * " " + l for l in ls)
