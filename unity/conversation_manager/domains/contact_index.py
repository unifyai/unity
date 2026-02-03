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
from unity.conversation_manager.types import Medium

if TYPE_CHECKING:
    from unity.contact_manager.base import BaseContactManager


class CommsMessage:
    """Base class for actual communications with contacts.

    All message types representing real user<->assistant communications inherit
    from this class. Use isinstance(msg, CommsMessage) to distinguish actual
    communications from internal orchestration messages (like GuidanceMessage).
    """


@dataclass
class Message(CommsMessage):
    """Simple text message (SMS, voice utterances)."""

    name: str
    content: str
    timestamp: datetime
    role: str  # "user" or "assistant"


@dataclass
class EmailMessage(CommsMessage):
    """Email message with subject, body, and optional attachments."""

    name: str
    subject: str
    body: str
    email_id: str | None
    timestamp: datetime
    role: str  # "user" or "assistant"
    attachments: list[str] = field(default_factory=list)
    # Recipients (for reply-all functionality)
    to: list[str] = field(default_factory=list)
    cc: list[str] = field(default_factory=list)
    bcc: list[str] = field(default_factory=list)
    # Contact's role in this email: "sender", "to", "cc", or "bcc"
    # Used to clarify the contact's relationship to the email when rendered
    # in their contact-specific thread (emails appear in threads for ALL
    # contacts involved, not just the primary contact)
    contact_role: str | None = None


@dataclass
class UnifyMessage(CommsMessage):
    """A message from the Unify console chat interface, optionally with attachments."""

    name: str
    content: str
    timestamp: datetime
    role: str  # "user" or "assistant"
    attachments: list[str] = field(default_factory=list)


@dataclass
class GuidanceMessage:
    """Internal orchestration message (not an actual communication).

    Used for internal guidance between components (e.g., CallGuidance from the
    main ConversationManager brain to the voice agent). These should NOT appear
    in transcripts shown to external systems or used for communication context.
    """

    name: str
    content: str
    timestamp: datetime


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

    Fallback Mechanism:
    -------------------
    Before ContactManager is initialized, inbound messages may arrive with contact
    data. These contacts are cached in `_fallback_contacts` so get_contact() can
    return them. Once ContactManager is set, the fallback cache is cleared and all
    lookups go through ContactManager.
    """

    def __init__(self):
        self.active_conversations: dict[int, ConversationState] = {}
        self._contact_manager: "BaseContactManager | None" = None
        # Fallback cache for contacts before ContactManager is initialized
        self._fallback_contacts: dict[int, dict] = {}

    def set_contact_manager(self, contact_manager: "BaseContactManager") -> None:
        """Set the ContactManager to use as the source of truth for contact data.

        Note: We do NOT clear the fallback cache here. Contacts cached from inbounds
        that arrived before initialization should remain available until they can be
        looked up in ContactManager. The fallback cache is checked first in get_contact().
        """
        self._contact_manager = contact_manager

    @property
    def is_contact_manager_initialized(self) -> bool:
        """Check if ContactManager has been set."""
        return self._contact_manager is not None

    @property
    def contact_manager(self) -> "BaseContactManager":
        """Get the ContactManager. Raises if not set."""
        if self._contact_manager is None:
            raise RuntimeError("ContactManager not set on ContactIndex")
        return self._contact_manager

    def set_fallback_contacts(self, contacts: list[dict]) -> None:
        """
        Cache contacts from inbound messages.

        This is called when inbound messages arrive with contact data. These
        contacts are checked first in get_contact() before ContactManager,
        ensuring contacts from recent inbounds are always available even if
        ContactManager hasn't synced them yet.

        Args:
            contacts: List of contact dicts from inbound message events.
        """
        for contact in contacts:
            contact_id = contact.get("contact_id")
            if contact_id is not None:
                self._fallback_contacts[contact_id] = contact

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
        Get contact information from fallback cache or ContactManager.

        Checks the local fallback cache first (populated from inbound message
        events). If not found, falls back to ContactManager.

        Args:
            contact_id: Contact ID (preferred).
            phone_number: Phone number to search by.
            email: Email address to search by.

        Returns:
            Contact dict or None if not found.
        """
        if self._contact_manager is None:
            # Check fallback cache first (contacts from inbound messages)
            if contact_id is not None:
                if contact_id in self._fallback_contacts:
                    return self._fallback_contacts[contact_id]
            elif phone_number is not None:
                for c in self._fallback_contacts.values():
                    if c.get("phone_number") == phone_number:
                        return c
            elif email is not None:
                for c in self._fallback_contacts.values():
                    if c.get("email_address") == email:
                        return c
        else:
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
        to: list[str] | None = None,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        contact_role: str | None = None,
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
            to: List of recipient email addresses (for email).
            cc: List of CC email addresses (for email).
            bcc: List of BCC email addresses (for email).
            contact_role: Contact's role in this email ("sender", "to", "cc", "bcc").
        """
        if not timestamp:
            timestamp = prompt_now(as_string=False)

        conversation = self.get_or_create_conversation(contact_id)

        # Determine display name (for rendering to brain)
        name = sender_name if role == "user" else "You" if role == "assistant" else role

        # Non-comms roles (e.g., "guidance") get a GuidanceMessage
        if role not in ("user", "assistant"):
            message = GuidanceMessage(
                name=name,
                content=message_content or "",
                timestamp=timestamp,
            )
        # Create appropriate comms message type based on medium
        elif thread_name == Medium.EMAIL:
            message = EmailMessage(
                name=name,
                subject=subject or "",
                body=body or "",
                email_id=email_id,
                timestamp=timestamp,
                role=role,
                attachments=attachments or [],
                to=to or [],
                cc=cc or [],
                bcc=bcc or [],
                contact_role=contact_role,
            )
        elif thread_name == Medium.UNIFY_MESSAGE:
            message = UnifyMessage(
                name=name,
                content=message_content or "",
                timestamp=timestamp,
                role=role,
                attachments=attachments or [],
            )
        else:
            message = Message(
                name=name,
                content=message_content or "",
                timestamp=timestamp,
                role=role,
            )

        conversation.threads[thread_name].append(message)
        conversation.global_thread.append(message)
