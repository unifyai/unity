from collections import deque
from datetime import datetime
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from unity.contact_manager.types.contact import Contact as ContactType
from unity.common.prompt_helpers import now as prompt_now


class Contact(ContactType):
    is_boss: bool = False
    on_call: bool = False
    global_thread: deque = Field(default_factory=lambda: deque(maxlen=50))
    threads: dict[str, deque] = Field(
        default_factory=lambda: {
            "sms": deque(maxlen=25),
            "email": deque(maxlen=25),
            "voice": deque(maxlen=25),
            "unify_message": deque(maxlen=25),
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
    email_id: str | None
    timestamp: datetime


class ContactIndex:
    def __init__(self):
        self.active_conversations: dict[str, Contact] = {}
        self.contacts: dict[int, Contact] = {}

    @property
    def boss_contact(self):
        # this will have empty threads
        return self.contacts.get(1)

    def set_contacts(self, contacts: list[dict]):
        print(f"Setting contacts: {contacts}")
        for c in contacts:
            self.contacts[c["contact_id"]] = Contact(**c, is_boss=c["contact_id"] == 1)

        # only retain the -1 contact if it's different from the boss contact
        c_neg1 = self.contacts.get(-1)
        c_boss = self.contacts.get(1)
        if c_neg1 and c_boss and c_neg1.first_name == c_boss.first_name:
            self.contacts.pop(-1, None)

    def clear_conversations(self):
        """Clear all active conversations for test isolation."""
        self.active_conversations.clear()

    # is this supposed to fail for any reason?
    def push_message(
        self,
        contact: dict,
        thread_name,
        message_content=None,
        subject=None,
        body=None,
        email_id=None,
        timestamp=None,
        role: Literal["user", "assistant"] = "user",
    ):
        if not timestamp:
            timestamp = prompt_now(as_string=False)
        contact_id = contact["contact_id"]
        if contact_id not in self.active_conversations:
            self.active_conversations[contact_id] = Contact(**contact)
        contact = self.active_conversations[contact_id]
        if thread_name == "email":
            message = EmailMessage(
                (
                    contact.full_name
                    if role == "user"
                    else "You" if role == "assistant" else role
                ),
                subject,
                body,
                email_id,
                timestamp,
            )
        else:
            message = Message(
                (
                    contact.full_name
                    if role == "user"
                    else "You" if role == "assistant" else role
                ),
                message_content,
                timestamp,
            )
        contact.threads[thread_name].append(message)
        contact.global_thread.append(message)

    # should check if the contact exists
    def get_contact(
        self,
        contact_id: str = None,
        phone_number=None,
        email=None,
    ) -> dict | None:
        c = None
        if contact_id:
            c = self.contacts.get(contact_id)
        elif phone_number:
            c = next(
                (c for c in self.contacts.values() if c.phone_number == phone_number),
                None,
            )
        elif email:
            c = next(
                (c for c in self.contacts.values() if c.email_address == email),
                None,
            )
        return c.model_dump(exclude={"threads", "global_thread"}) if c else None
