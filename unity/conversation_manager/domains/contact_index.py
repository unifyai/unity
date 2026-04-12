"""
ContactIndex: Conversation state management for ConversationManager.

All messages are stored in a single shared global deque. Per-contact and
per-medium views are derived on demand. Contact information (name, email,
phone, response_policy, etc.) is fetched from ContactManager, which is the
single source of truth.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from unity.common.prompt_helpers import now as prompt_now
from unity.conversation_manager.cm_types import Medium

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
    local_message_id: int = field(default=0, compare=False)
    screenshots: list[str] = field(default_factory=list, compare=False)
    image_ids: list[int] = field(default_factory=list, compare=False)


@dataclass
class EmailMessage(CommsMessage):
    """Email message with subject, body, and optional attachments.

    Each attachment is a dict with keys: id, filename (and optionally filepath).
    """

    name: str
    subject: str
    body: str
    email_id: str | None
    timestamp: datetime
    role: str  # "user" or "assistant"
    attachments: list[dict] = field(default_factory=list)
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
    """A message from the Unify console chat interface, optionally with attachments.

    Each attachment is a dict with keys: id, filename, gs_url, content_type, size_bytes.
    """

    name: str
    content: str
    timestamp: datetime
    role: str  # "user" or "assistant"
    attachments: list[dict] = field(default_factory=list)


@dataclass
class WhatsAppMessage(CommsMessage):
    """A WhatsApp message, optionally with attachments.

    Each attachment is a dict with keys: id, filename, gs_url, content_type, size_bytes.
    """

    name: str
    content: str
    timestamp: datetime
    role: str  # "user" or "assistant"
    attachments: list[dict] = field(default_factory=list)


@dataclass
class ApiMessage(CommsMessage):
    """A programmatic API message, optionally with attachments and developer-supplied tags.

    Each attachment is a dict with keys: id, filename, gs_url, content_type, size_bytes.
    Tags are opaque strings chosen by the developer for routing and context.
    """

    name: str
    content: str
    timestamp: datetime
    role: str  # "user" or "assistant"
    attachments: list[dict] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class GuidanceMessage:
    """Internal orchestration message (not an actual communication).

    Used for internal notifications between components (e.g., FastBrainNotification
    from the ConversationManager to the voice agent). These should NOT appear in
    transcripts shown to external systems or used for communication context.
    """

    name: str
    content: str
    timestamp: datetime


# Message type -> Medium mapping for deriving per-medium views from the global deque.
_MESSAGE_TYPE_TO_MEDIUM: dict[type, Medium] = {
    EmailMessage: Medium.EMAIL,
    UnifyMessage: Medium.UNIFY_MESSAGE,
    ApiMessage: Medium.API_MESSAGE,
}


@dataclass
class GlobalThreadEntry:
    """An entry in the shared global thread.

    Wraps a message with its contact associations and medium, enabling
    per-contact and per-medium views to be derived from the single deque.
    """

    message: Message | EmailMessage | UnifyMessage | GuidanceMessage
    medium: Medium
    # For most messages, a single contact. For emails, all involved contacts
    # with their roles (sender, to, cc, bcc). role is None for non-email.
    contact_roles: dict[int, str | None]


@dataclass
class ConversationState:
    """Per-contact conversation metadata (not message storage).

    Messages live in the shared global deque on ContactIndex. This class
    stores only non-message state like call status.
    """

    contact_id: int
    on_call: bool = False


class ContactIndex:
    """
    Manages conversation state for active contacts.

    All messages are stored in a single shared global deque. Per-contact and
    per-medium views are derived on demand via helper methods.

    Contact information (name, email, phone, response_policy, etc.) is ALWAYS
    fetched from ContactManager - the single source of truth with DataStore-backed
    caching.

    Fallback Mechanism:
    -------------------
    Before ContactManager is initialized, inbound messages may arrive with contact
    data. These contacts are cached in `_fallback_contacts` so get_contact() can
    return them. Once ContactManager is set, the fallback cache is cleared and all
    lookups go through ContactManager.
    """

    DEFAULT_GLOBAL_THREAD_SIZE = 100

    def __init__(self, global_thread_size: int = DEFAULT_GLOBAL_THREAD_SIZE):
        self.active_conversations: dict[int, ConversationState] = {}
        self.global_thread: deque[GlobalThreadEntry] = deque(
            maxlen=global_thread_size,
        )
        self._contact_manager: "BaseContactManager | None" = None
        # Fallback cache for contacts before ContactManager is initialized
        self._fallback_contacts: dict[int, dict] = {}
        self._next_local_message_id: int = 0

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
        self.global_thread.clear()

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
        whatsapp_number: str | None = None,
    ) -> dict | None:
        """
        Get contact information from fallback cache or ContactManager.

        Checks the local fallback cache first (populated from inbound message
        events). If not found, falls back to ContactManager.

        Args:
            contact_id: Contact ID (preferred).
            phone_number: Phone number to search by.
            email: Email address to search by.
            whatsapp_number: WhatsApp number to search by.

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
            elif whatsapp_number is not None:
                for c in self._fallback_contacts.values():
                    if c.get("whatsapp_number") == whatsapp_number:
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
                elif whatsapp_number is not None:
                    result = self._contact_manager.filter_contacts(
                        filter=f"whatsapp_number == '{whatsapp_number}'",
                        limit=1,
                    )
                    contacts = result.get("contacts", [])
                    if contacts:
                        c = contacts[0]
                        return c.model_dump() if hasattr(c, "model_dump") else c
            except Exception:
                return None
        return None

    # =========================================================================
    # Message query helpers — derive views from the shared global deque
    # =========================================================================

    def get_messages_for_contact(
        self,
        contact_id: int,
        medium: Medium | None = None,
    ) -> list:
        """Get messages for a contact, optionally filtered by medium.

        Args:
            contact_id: The contact to filter for.
            medium: If provided, only return messages of this medium.

        Returns:
            List of messages (in chronological order) for this contact.
        """
        results = []
        for entry in self.global_thread:
            if contact_id not in entry.contact_roles:
                continue
            if medium is not None and entry.medium != medium:
                continue
            results.append(entry.message)
        return results

    def get_active_contact_ids(self) -> set[int]:
        """Return the set of contact_ids present in the global thread."""
        ids: set[int] = set()
        for entry in self.global_thread:
            ids.update(entry.contact_roles.keys())
        return ids

    def get_messages_grouped_by_contact(
        self,
    ) -> dict[int, list[GlobalThreadEntry]]:
        """Group all global thread entries by contact_id.

        Returns a dict mapping contact_id to a list of GlobalThreadEntry
        in chronological order. An entry appears under every contact_id
        in its contact_roles.
        """
        groups: dict[int, list[GlobalThreadEntry]] = {}
        for entry in self.global_thread:
            for cid in entry.contact_roles:
                if cid not in groups:
                    groups[cid] = []
                groups[cid].append(entry)
        return groups

    # =========================================================================
    # Message push
    # =========================================================================

    def build_message(
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
        tags: list[str] | None = None,
    ) -> "GlobalThreadEntry":
        """
        Build a GlobalThreadEntry without appending it to the global thread.

        Accepts the same arguments as push_message. Also ensures that
        conversation state exists for the contact.
        """
        if not timestamp:
            timestamp = prompt_now(as_string=False)

        # Ensure conversation state exists for this contact
        self.get_or_create_conversation(contact_id)

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
        elif thread_name == Medium.WHATSAPP_MESSAGE:
            message = WhatsAppMessage(
                name=name,
                content=message_content or "",
                timestamp=timestamp,
                role=role,
                attachments=attachments,
            )
        elif thread_name == Medium.API_MESSAGE:
            message = ApiMessage(
                name=name,
                content=message_content or "",
                timestamp=timestamp,
                role=role,
                attachments=attachments or [],
                tags=tags or [],
            )
        else:
            self._next_local_message_id += 1
            message = Message(
                name=name,
                content=message_content or "",
                timestamp=timestamp,
                role=role,
                local_message_id=self._next_local_message_id,
            )

        return GlobalThreadEntry(
            message=message,
            medium=thread_name,
            contact_roles={contact_id: contact_role},
        )

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
        tags: list[str] | None = None,
    ) -> int:
        """
        Build a message and append it to the shared global thread.

        Returns the message_id assigned to the new message (0 for non-Message types).
        """
        entry = self.build_message(
            contact_id=contact_id,
            sender_name=sender_name,
            thread_name=thread_name,
            message_content=message_content,
            subject=subject,
            body=body,
            email_id=email_id,
            attachments=attachments,
            timestamp=timestamp,
            role=role,
            to=to,
            cc=cc,
            bcc=bcc,
            contact_role=contact_role,
            tags=tags,
        )
        self.global_thread.append(entry)
        msg = entry.message
        return msg.local_message_id if isinstance(msg, Message) else 0

    def prepend_entries(self, entries: list) -> None:
        """Prepend entries to the front of the global thread.

        Used by hydration to insert historical messages before any messages
        that arrived during initialization. Respects the deque maxlen by
        keeping the most recent entries when the combined size exceeds it.
        """
        if not entries:
            return
        existing = list(self.global_thread)
        self.global_thread.clear()
        # extend respects maxlen, dropping oldest (leftmost) if over capacity
        self.global_thread.extend(entries + existing)
