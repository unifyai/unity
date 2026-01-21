"""
ContactIndex: Conversation state management for ConversationManager.

This module stores ONLY conversation state (threads, on_call status).
All contact information (name, email, phone, response_policy, etc.) is
fetched from ContactManager, which is the single source of truth.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from unity.common.prompt_helpers import now as prompt_now
from unity.transcript_manager.types.medium import Medium

if TYPE_CHECKING:
    from unity.contact_manager.base import BaseContactManager


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
    attachments: list[str] = field(default_factory=list)


@dataclass
class UnifyMessage:
    """A message from the Unify console chat interface, optionally with attachments."""

    name: str
    content: str
    timestamp: datetime
    attachments: list[str] = field(default_factory=list)


@dataclass
class ConversationState:
    """
    Conversation state for a single contact.

    This class stores ONLY conversation-related data (threads, call status).
    Contact information (name, email, etc.) is fetched from ContactManager.
    """

    contact_id: int
    on_call: bool = False
    global_thread: deque = field(default_factory=lambda: deque(maxlen=50))
    threads: dict[Medium, deque] = field(
        default_factory=lambda: {
            Medium.SMS_MESSAGE: deque(maxlen=25),
            Medium.EMAIL: deque(maxlen=25),
            Medium.PHONE_CALL: deque(maxlen=25),
            Medium.UNIFY_MEET: deque(maxlen=25),
            Medium.UNIFY_MESSAGE: deque(maxlen=25),
        },
    )


class ContactIndex:
    """
    Manages conversation state for active contacts.

    Contact information (name, email, phone, response_policy, etc.) is ALWAYS
    fetched from ContactManager - the single source of truth with DataStore-backed
    caching. This class only stores conversation state (message threads, call status).
    """

    def __init__(self):
        self.active_conversations: dict[int, ConversationState] = {}
        self._contact_manager: "BaseContactManager | None" = None

    def set_contact_manager(self, contact_manager: "BaseContactManager") -> None:
        """Set the ContactManager to use as the source of truth for contact data."""
        self._contact_manager = contact_manager

    @property
    def contact_manager(self) -> "BaseContactManager":
        """Get the ContactManager. Raises if not set."""
        if self._contact_manager is None:
            raise RuntimeError("ContactManager not set on ContactIndex")
        return self._contact_manager

    def clear_conversations(self):
        """Clear all active conversations for test isolation."""
        self.active_conversations.clear()

    def get_conversation_state(self, contact_id: int) -> ConversationState | None:
        """Get conversation state for a contact, or None if no active conversation."""
        return self.active_conversations.get(contact_id)

    def get_or_create_conversation(self, contact_id: int) -> ConversationState:
        """Get or create conversation state for a contact."""
        if contact_id not in self.active_conversations:
            self.active_conversations[contact_id] = ConversationState(
                contact_id=contact_id,
            )
        return self.active_conversations[contact_id]

    def get_contact(
        self,
        contact_id: int | None = None,
        phone_number: str | None = None,
        email: str | None = None,
    ) -> dict | None:
        """
        Get contact information from ContactManager.

        This is the ONLY way to get contact data - always queries ContactManager
        which maintains a DataStore-backed cache synced with the backend.

        Args:
            contact_id: Contact ID (preferred).
            phone_number: Phone number to search by.
            email: Email address to search by.

        Returns:
            Contact dict or None if not found.
        """
        if self._contact_manager is None:
            return None
        try:
            if contact_id is not None:
                result = self._contact_manager.get_contact_info(contact_id)
                return result.get(contact_id)
            elif phone_number is not None:
                result = self._contact_manager.filter_contacts(
                    filter=f"phone_number == '{phone_number}'",
                    limit=1,
                )
                contacts = result.get("contacts", [])
                if contacts:
                    c = contacts[0]
                    return c.model_dump() if hasattr(c, "model_dump") else c
            elif email is not None:
                result = self._contact_manager.filter_contacts(
                    filter=f"email_address == '{email}'",
                    limit=1,
                )
                contacts = result.get("contacts", [])
                if contacts:
                    c = contacts[0]
                    return c.model_dump() if hasattr(c, "model_dump") else c
        except Exception:
            return None
        return None

    def push_message(
        self,
        contact_id: int,
        sender_name: str,
        thread_name: Medium,
        message_content: str | None = None,
        subject: str | None = None,
        body: str | None = None,
        email_id: str | None = None,
        attachments: list[str] | None = None,
        timestamp: datetime | None = None,
        role: str = "user",
    ):
        """
        Push a message to a contact's conversation thread.

        Args:
            contact_id: The contact's ID.
            sender_name: Display name for the message sender.
            thread_name: Which thread to push to (Medium.SMS_MESSAGE, Medium.EMAIL, etc.).
            message_content: Message text (for SMS, voice).
            subject: Email subject (for email).
            body: Email body (for email).
            email_id: Email ID (for email).
            attachments: List of attachment filenames.
            timestamp: Message timestamp (defaults to now).
            role: "user" or "assistant".
        """
        if not timestamp:
            timestamp = prompt_now(as_string=False)

        conversation = self.get_or_create_conversation(contact_id)

        # Determine display name
        name = sender_name if role == "user" else "You" if role == "assistant" else role

        # Create appropriate message type
        if thread_name == Medium.EMAIL:
            message = EmailMessage(
                name=name,
                subject=subject or "",
                body=body or "",
                email_id=email_id,
                timestamp=timestamp,
                attachments=attachments or [],
            )
        elif thread_name == Medium.UNIFY_MESSAGE:
            message = UnifyMessage(
                name=name,
                content=message_content or "",
                timestamp=timestamp,
                attachments=attachments or [],
            )
        else:
            message = Message(
                name=name,
                content=message_content or "",
                timestamp=timestamp,
            )

        conversation.threads[thread_name].append(message)
        conversation.global_thread.append(message)
