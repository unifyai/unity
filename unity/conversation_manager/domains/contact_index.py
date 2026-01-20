from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, TYPE_CHECKING

from pydantic import Field

from unity.contact_manager.types.contact import Contact as ContactType
from unity.common.prompt_helpers import now as prompt_now

if TYPE_CHECKING:
    from unity.contact_manager.base import BaseContactManager


class Contact(ContactType):
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
    # List of attachment filenames (actual files are saved to Downloads/).
    attachments: list[str] = field(default_factory=list)


class ContactIndex:
    def __init__(self):
        self.active_conversations: dict[str, Contact] = {}
        self.contacts: dict[int, Contact] = {}
        self._contact_manager: "BaseContactManager | None" = None

    def set_contact_manager(self, contact_manager: "BaseContactManager") -> None:
        """Set the ContactManager to use as the source of truth for contact data.

        When set, get_contact() will always query ContactManager first to ensure
        up-to-date data, falling back to the local cache only if ContactManager
        is unavailable.
        """
        self._contact_manager = contact_manager

    @property
    def boss_contact(self):
        # this will have empty threads
        return self.contacts.get(1)

    def set_contacts(self, contacts: list[dict]):
        print(f"Setting contacts: {contacts}")
        for c in contacts:
            self.contacts[c["contact_id"]] = Contact(**c)

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
        attachments=None,
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
                attachments or [],
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

    def get_contact(
        self,
        contact_id: int = None,
        phone_number: str = None,
        email: str = None,
    ) -> dict | None:
        """Get contact information, always querying ContactManager for fresh data.

        When a ContactManager is configured (via set_contact_manager), this method
        queries it first to ensure up-to-date data. The ContactManager maintains
        an auto-syncing cache backed by the database, so any updates made by other
        components (e.g., Actor creating contacts) are immediately reflected.

        Falls back to the local cache only if ContactManager is unavailable or
        the query fails.

        Args:
            contact_id: The contact's unique ID.
            phone_number: The contact's phone number (used if contact_id not provided).
            email: The contact's email address (used if contact_id and phone_number not provided).

        Returns:
            Contact data as a dict, or None if not found.
        """
        # Always prefer ContactManager (source of truth with auto-syncing cache)
        if self._contact_manager is not None:
            try:
                if contact_id is not None:
                    result = self._contact_manager.get_contact_info(contact_id)
                    if contact_id in result:
                        return result[contact_id]
                elif phone_number is not None:
                    # Use filter_contacts to search by phone number
                    result = self._contact_manager.filter_contacts(
                        filter=f"phone_number == '{phone_number}'",
                        limit=1,
                    )
                    # filter_contacts returns {"contacts": [Contact(...)]}
                    contacts = result.get("contacts", [])
                    if contacts:
                        c = contacts[0]
                        return c.model_dump() if hasattr(c, "model_dump") else c
                elif email is not None:
                    # Use filter_contacts to search by email
                    result = self._contact_manager.filter_contacts(
                        filter=f"email_address == '{email}'",
                        limit=1,
                    )
                    # filter_contacts returns {"contacts": [Contact(...)]}
                    contacts = result.get("contacts", [])
                    if contacts:
                        c = contacts[0]
                        return c.model_dump() if hasattr(c, "model_dump") else c
            except Exception:
                # Fall through to local cache on any error
                pass

        # Fallback to local cache (may be stale, but better than nothing)
        c = None
        if contact_id is not None:
            c = self.contacts.get(contact_id)
        elif phone_number is not None:
            c = next(
                (c for c in self.contacts.values() if c.phone_number == phone_number),
                None,
            )
        elif email is not None:
            c = next(
                (c for c in self.contacts.values() if c.email_address == email),
                None,
            )
        return c.model_dump(exclude={"threads", "global_thread"}) if c else None
