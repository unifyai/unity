"""
Brain action tools for ConversationManager.

All contact information is fetched from ContactManager (source of truth).
No local caching of contact data.

Context Propagation:
- When `act` is called, the current state snapshot is passed to Actor via _parent_chat_context
- For `interject` operations, only the incremental diff from the initial snapshot is sent
  via _parent_chat_context_cont, avoiding duplication of unchanged state
"""

from __future__ import annotations

import asyncio
import inspect
import re
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel as _BaseModel
from pydantic import create_model as _create_model

from unity.common.prompt_helpers import now as prompt_now
from unity.logger import LOGGER
from unity.common.hierarchical_logger import DEFAULT_ICON, ICONS

from unity.conversation_manager.domains import comms_utils
from unity.conversation_manager.domains import managers_utils
from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.events import (
    SMSSent,
    UnifyMessageSent,
    EmailSent,
    PhoneCallSent,
    ActorHandleStarted,
    ActorHandleResponse,
    Error,
)
from unity.common._async_tool.dynamic_tools_factory import DynamicToolFactory
from unity.common._async_tool.utils import get_handle_paused_state
from unity.conversation_manager.types import Medium
from unity.conversation_manager.task_actions import (
    STEERING_OPERATIONS,
    OPERATION_MAP,
    derive_short_name,
    build_action_name,
    safe_call_id_suffix,
)
from unity.conversation_manager.domains.renderer import (
    SnapshotState,
    compute_snapshot_diff,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from unity.conversation_manager.conversation_manager import ConversationManager


# ─────────────────────────────────────────────────────────────────────────────
# Schema dict → Pydantic model conversion
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA_TYPE_MAP: dict[str, type] = {
    "string": str,
    "str": str,
    "integer": int,
    "int": int,
    "number": float,
    "float": float,
    "boolean": bool,
    "bool": bool,
}


def _resolve_schema_type(schema: Any, name_hint: str) -> type:
    """Resolve a single schema value to a Python type.

    Handles:
    - String type names (``"string"``, ``"integer"``, …)
    - Nested dicts (recursively creates a child Pydantic model)
    - Lists where the first element defines the item schema
    """
    if isinstance(schema, str):
        return _SCHEMA_TYPE_MAP.get(schema.lower(), str)
    if isinstance(schema, dict):
        return schema_dict_to_pydantic(schema, name_hint)
    if isinstance(schema, list) and len(schema) > 0:
        item_type = _resolve_schema_type(schema[0], f"{name_hint}Item")
        return list[item_type]  # type: ignore[valid-type]
    return str  # fallback for unrecognised shapes


def schema_dict_to_pydantic(
    schema: dict,
    model_name: str = "ResponseFormat",
) -> type[_BaseModel]:
    """Convert a simplified schema dict to a dynamic Pydantic model.

    The schema uses a concise, LLM-friendly format:

    - **String values** are type names: ``"string"``, ``"integer"``,
      ``"number"``, ``"boolean"`` (shorthand ``"str"``, ``"int"``, etc.
      also accepted).
    - **Dict values** define nested object schemas (recursively converted).
    - **List values** define array types; the first element is the item
      schema.

    Examples::

        # Flat
        {"email": "string", "age": "integer"}

        # Nested with array
        {"contacts": [{"name": "string", "phone": "string"}], "total": "integer"}
    """
    fields: dict[str, tuple[type, ...]] = {}
    for field_name, field_schema in schema.items():
        field_type = _resolve_schema_type(
            field_schema,
            f"{model_name}{field_name.title()}",
        )
        fields[field_name] = (field_type, ...)
    return _create_model(model_name, **fields)


def _coerce_contact_id(v: Any) -> int:
    """Coerce a contact_id value to int, handling string-encoded integers."""
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        try:
            return int(v)
        except (ValueError, TypeError):
            pass
    raise TypeError(f"contact_id must be an integer, got {type(v).__name__}: {v!r}")


# Global handle ID counter for action tracking
_next_handle_id = 0


def _get_contact_display_name(contact: dict | None) -> str:
    """Get a display name for a contact for error messages."""
    if not contact:
        return "unknown contact"
    first = contact.get("first_name") or ""
    last = contact.get("surname") or ""
    name = f"{first} {last}".strip()
    if not name:
        name = f"contact_id={contact.get('contact_id', 'unknown')}"
    return name


def _check_outbound_allowed(contact: dict | None) -> str | None:
    """Check if outbound communication is allowed for a contact."""
    if not contact:
        return "Contact not found"
    should_respond = contact.get("should_respond", False)
    if not should_respond:
        contact_name = _get_contact_display_name(contact)
        return (
            f"Cannot send outbound communication to {contact_name}: "
            f"should_respond is False for this contact. "
            f"Check the contact's response_policy for details or ask your boss for guidance."
        )
    return None


# Pattern matching <in_flight_actions>...</in_flight_actions> sections.
# These contain CM-level steering tools that should not be exposed to the Actor.
_IN_FLIGHT_ACTIONS_PATTERN = re.compile(
    r"<in_flight_actions>.*?</in_flight_actions>\s*",
    re.DOTALL,
)


def _filter_cm_state_for_actor(state_snapshot: dict) -> dict:
    """Filter CM state snapshot before passing to Actor as parent context.

    The CM state snapshot contains <in_flight_actions> with <steering_tools>
    listing CM-level tools (stop_, pause_, interject_, ask_) for each action.
    These are CM brain tools that exist only in the CM's tool surface.

    If passed verbatim to the Actor, the Actor LLM may interpret these tool
    names as callable functions and generate code like:
        await stop_search_the_web_for__1()
    This causes NameError since these tools don't exist in the Actor's scope.

    This function strips the <in_flight_actions> section while preserving
    other useful context (notifications, active_conversations).

    Args:
        state_snapshot: The CM state snapshot dict with "content" key.

    Returns:
        A filtered copy of the snapshot with in_flight_actions removed.
    """
    if not state_snapshot:
        return state_snapshot

    content = state_snapshot.get("content", "")
    if not content:
        return state_snapshot

    # When screenshots are attached, content is a list of multimodal parts
    # rather than a plain string. Apply the regex to each text part.
    if isinstance(content, list):
        filtered_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                filtered_text = _IN_FLIGHT_ACTIONS_PATTERN.sub("", part["text"])
                filtered_parts.append({**part, "text": filtered_text})
            else:
                filtered_parts.append(part)
        return {**state_snapshot, "content": filtered_parts}

    filtered_content = _IN_FLIGHT_ACTIONS_PATTERN.sub("", content)
    return {**state_snapshot, "content": filtered_content}


def _check_contact_has_address(
    contact: dict | None,
    address_field: str,
    communication_type: str,
) -> str | None:
    """Check if a contact has the required address for a communication type."""
    if not contact:
        return f"Contact not found for {communication_type}"
    address = contact.get(address_field)
    if not address:
        contact_name = _get_contact_display_name(contact)
        field_display = address_field.replace("_", " ")
        return (
            f"Cannot send {communication_type} to {contact_name}: "
            f"this contact does not have an {field_display} on file."
        )
    return None


def _resolve_or_attach_detail(
    contact: dict | None,
    contact_id: int,
    address_field: str,
    inline_value: str | None,
    communication_type: str,
    contact_index: Any,
) -> tuple[str | None, dict | None]:
    """Resolve a contact's address field, optionally attaching an inline value.

    Supports implicit contact-detail creation for contacts that are already in
    the active conversation.  The rules are:

    1. Contact has the field (no inline, or inline matches) -> happy path.
    2. Contact has the field but inline is *different* -> error (use ``act``).
    3. Contact missing the field, no inline provided -> error (missing detail).
    4. Contact missing the field, inline provided -> attach, re-fetch, proceed.

    Args:
        contact: The contact dict (or ``None`` if not found).
        contact_id: The contact's ID.
        address_field: Field to check (e.g. ``"phone_number"``, ``"email_address"``).
        inline_value: Optional inline value provided by the caller.
        communication_type: Human-readable type for error messages (e.g. ``"SMS"``).
        contact_index: :class:`ContactIndex` used for the update + re-fetch.

    Returns:
        ``(error_message, updated_contact)`` -- if *error_message* is not
        ``None`` the operation should be aborted.
    """
    if not contact:
        return (f"Contact not found for {communication_type}", None)

    existing_value = contact.get(address_field)
    field_display = address_field.replace("_", " ")
    contact_name = _get_contact_display_name(contact)

    if existing_value:
        # Contact already has this field
        if inline_value and inline_value != existing_value:
            return (
                f"Cannot send {communication_type} to {contact_name}: "
                f"this contact already has {field_display} '{existing_value}' on file, "
                f"but you provided '{inline_value}'. "
                f"Use `act` to update the contact's {field_display} if needed, "
                f"then retry the {communication_type}.",
                None,
            )
        # No inline value, or same value -- proceed with existing
        return (None, contact)

    # Contact is missing this field
    if not inline_value:
        return (
            f"Cannot send {communication_type} to {contact_name}: "
            f"this contact does not have a {field_display} on file. "
            f"Provide the {field_display} inline or use `act` to update "
            f"the contact first.",
            None,
        )

    # Attach the inline value to the existing contact
    contact_index.contact_manager.update_contact(
        contact_id=contact_id,
        **{address_field: inline_value},
    )
    # Re-fetch to get fresh data after the update
    updated_contact = contact_index.get_contact(contact_id)
    return (None, updated_contact)


class _DesktopActionHandle:
    """Lightweight handle wrapping a single desktop primitive call.

    Provides the minimal interface consumed by the CM watcher infrastructure
    (``actor_watch_result``) so desktop fast-path actions participate in the
    same in-flight lifecycle as ``act`` and the contact/transcript fast paths.

    Steering operations (pause/resume/interject/ask) are no-ops because these
    are atomic single-step actions with no inner loop to steer.
    """

    def __init__(self, task: asyncio.Task):
        self._task = task
        self._notification_q: asyncio.Queue = asyncio.Queue()

    def done(self) -> bool:
        return self._task.done()

    async def result(self) -> str:
        return await self._task

    async def next_notification(self) -> dict:
        while not self._task.done():
            try:
                return await asyncio.wait_for(self._notification_q.get(), timeout=30)
            except asyncio.TimeoutError:
                continue
        raise asyncio.CancelledError

    async def next_clarification(self) -> dict:
        while not self._task.done():
            await asyncio.sleep(30)
        raise asyncio.CancelledError

    async def stop(self, reason=None, **kwargs):
        self._task.cancel()

    async def interject(self, message, **kwargs):
        pass

    async def ask(self, question, **kwargs):
        pass

    async def pause(self):
        pass

    async def resume(self):
        pass


class ConversationManagerBrainActionTools:
    """
    Side-effecting tools for the Main CM Brain.

    All contact data is fetched from ContactManager - no local caching.
    """

    def __init__(self, cm: "ConversationManager"):
        self._cm = cm
        self._event_broker = get_event_broker()

    async def _surface_comms_error(
        self,
        error_msg: str,
        topic: str,
        *,
        contact_id: int | None = None,
        medium: str | None = None,
    ) -> dict[str, Any]:
        """Push a comms error into the conversation thread and publish the Error event.

        Ensures the brain sees the failure in ``active_conversations`` on the
        next turn and can reason about recovery (retry with missing info,
        fall back to a different channel, etc.).

        Args:
            error_msg: Human-readable error description.
            topic: Event broker topic (e.g. ``"app:comms:sms_sent"``).
            contact_id: Target contact (when known) — the error is pushed into
                their conversation thread.
            medium: Communication medium for the thread entry (e.g.
                ``Medium.SMS_MESSAGE``).

        Returns:
            ``{"status": "error", "error": <error_msg>}`` for the tool return.
        """
        if contact_id is not None and medium is not None:
            self._cm.contact_index.push_message(
                contact_id=contact_id,
                sender_name="System",
                thread_name=medium,
                message_content=f"[Send Failed] {error_msg}",
                role="system",
                timestamp=prompt_now(as_string=False),
            )
        event = Error(error_msg)
        await self._event_broker.publish(topic, event.to_json())
        return {"status": "error", "error": error_msg}

    async def send_sms(
        self,
        *,
        contact_id: int | str,
        content: str,
        phone_number: str | None = None,
    ) -> dict[str, Any]:
        """
        Send an SMS message to an existing contact.

        The contact must already exist in the system.

        - If the contact **already has** a phone number on file (visible in
          active_conversations), omit ``phone_number`` -- it is not needed.
        - If the contact **does not have** a phone number on file but you
          know it (e.g. the boss provided it), pass it via ``phone_number``.
          It will be saved to the contact record automatically and the SMS
          will be sent in one step.
        - **Do not** pass a ``phone_number`` that differs from the one
          already on file -- this will be rejected.  Use ``act`` to update
          the contact's phone number first, then retry.

        Args:
            contact_id: The contact_id of the recipient (from
                active_conversations or returned by ``find_contacts`` /
                ``create_contact``).
            content: The text content of the SMS message to send.
            phone_number: The recipient's phone number.  Required when the
                contact does not yet have a phone number on file; omit when
                the contact already has one.
        """
        contact_id = _coerce_contact_id(contact_id)
        contact = self._cm.contact_index.get_contact(contact_id)

        outbound_error = _check_outbound_allowed(contact)
        if outbound_error:
            return await self._surface_comms_error(
                outbound_error,
                "app:comms:sms_sent",
                contact_id=contact_id,
                medium=Medium.SMS_MESSAGE,
            )

        detail_error, contact = _resolve_or_attach_detail(
            contact,
            contact_id,
            "phone_number",
            phone_number,
            "SMS",
            self._cm.contact_index,
        )
        if detail_error:
            return await self._surface_comms_error(
                detail_error,
                "app:comms:sms_sent",
                contact_id=contact_id,
                medium=Medium.SMS_MESSAGE,
            )

        to_number = contact.get("phone_number")
        response = await comms_utils.send_sms_message_via_number(
            to_number=to_number,
            content=content,
        )

        if response["success"]:
            # Re-fetch contact from ContactManager to ensure fresh data
            fresh_contact = (
                self._cm.contact_index.get_contact(phone_number=to_number) or contact
            )
            event = SMSSent(contact=fresh_contact, content=content)
            await self._event_broker.publish("app:comms:sms_sent", event.to_json())
            return {"status": "ok"}

        if not self._cm.assistant_number:
            error_msg = "You don't have a number, please provision one."
        else:
            error_msg = f"Failed to send sms to {to_number}"
        return await self._surface_comms_error(
            error_msg,
            "app:comms:sms_sent",
            contact_id=contact_id,
            medium=Medium.SMS_MESSAGE,
        )

    async def send_unify_message(
        self,
        *,
        content: str,
        contact_id: int | str,
        attachment_filepath: str | None = None,
    ) -> dict[str, Any]:
        """
        Send a Unify message to a contact via the Unify platform.

        Args:
            content: Message content to send.
            contact_id: Target contact_id (integer) from active conversations.
            attachment_filepath: Optional filepath to attach.
        """
        contact_id = _coerce_contact_id(contact_id)
        import os

        contact = self._cm.contact_index.get_contact(contact_id=contact_id)

        _unify_topic = "app:comms:unify_message_sent"
        _unify_err = dict(contact_id=contact_id, medium=Medium.UNIFY_MESSAGE)

        if contact:
            outbound_error = _check_outbound_allowed(contact)
            if outbound_error:
                return await self._surface_comms_error(
                    outbound_error,
                    _unify_topic,
                    **_unify_err,
                )

        # Handle attachment
        attachment = None
        attachment_filename = None
        if attachment_filepath:
            try:
                from unity.file_manager.filesystem_adapters.local_adapter import (
                    LocalFileSystemAdapter,
                )

                adapter = LocalFileSystemAdapter()
                file_ref = adapter.get_file(attachment_filepath)
                abs_path = adapter._abspath(attachment_filepath)
                with open(abs_path, "rb") as f:
                    file_contents = f.read()

                max_size_mb = 25
                file_size_mb = len(file_contents) / (1024 * 1024)
                if file_size_mb > max_size_mb:
                    return await self._surface_comms_error(
                        f"File too large: {file_size_mb:.1f}MB exceeds {max_size_mb}MB limit",
                        _unify_topic,
                        **_unify_err,
                    )

                attachment_filename = os.path.basename(attachment_filepath)
                upload_result = await comms_utils.upload_unify_attachment(
                    file_content=file_contents,
                    filename=attachment_filename,
                )

                if "error" in upload_result:
                    return await self._surface_comms_error(
                        f"Failed to upload attachment: {upload_result['error']}",
                        _unify_topic,
                        **_unify_err,
                    )

                attachment = upload_result

            except FileNotFoundError:
                return await self._surface_comms_error(
                    f"File not found: {attachment_filepath}",
                    _unify_topic,
                    **_unify_err,
                )
            except Exception as e:
                return await self._surface_comms_error(
                    f"Failed to read file: {e}",
                    _unify_topic,
                    **_unify_err,
                )

        response = await comms_utils.send_unify_message(
            content=content,
            contact_id=contact_id,
            attachment=attachment,
        )
        if response["success"]:
            fresh_contact = (
                self._cm.contact_index.get_contact(contact_id=contact_id)
                or contact
                or {}
            )
            # Use full attachment metadata if available, otherwise empty list
            attachments_for_event = [attachment] if attachment else []
            event = UnifyMessageSent(
                contact=fresh_contact,
                content=content,
                attachments=attachments_for_event,
            )
            await self._event_broker.publish(_unify_topic, event.to_json())
            return {"status": "ok"}

        return await self._surface_comms_error(
            "Failed to send unify message",
            _unify_topic,
            **_unify_err,
        )

    async def send_email(
        self,
        *,
        to: list[int | dict] | None = None,
        cc: list[int | dict] | None = None,
        bcc: list[int | dict] | None = None,
        subject: str,
        body: str,
        reply_all: bool = False,
        email_id_to_reply_to: str | None = None,
        attachment_filepath: str | None = None,
    ) -> dict[str, Any]:
        """
        Send an email to existing contacts.

        Each contact must already exist in the system.  Each recipient in
        ``to``, ``cc``, and ``bcc`` is specified in one of two ways:

        - **Contact already has an email on file** -- pass the bare
          ``contact_id`` (integer).  Example: ``to=[5]``.
        - **Contact does NOT have an email on file** but you know it
          (e.g. the boss provided it) -- pass a dict with both fields:
          ``{"contact_id": 5, "email_address": "alice@example.com"}``.
          The email will be saved to the contact record automatically and
          the email will be sent in one step.

        You can mix both forms in the same list.

        **Do not** use the dict form to supply an email that differs from
        the one already on file -- this will be rejected.  Use ``act`` to
        update the contact's email address first, then retry.

        Duplicates are automatically collapsed.

        Args:
            to: Primary recipients.  Each element is either a
                ``contact_id`` (int) when the contact already has an email
                on file, or ``{"contact_id": int, "email_address": str}``
                when you need to provide the email address.
            cc: CC recipients (same format as ``to``).
            bcc: BCC recipients (same format as ``to``).
            subject: Email subject.
            body: Email body.
            reply_all: If True, automatically populate to/cc from the email being
                replied to. Mutually exclusive with to/cc/bcc - fails if both are set.
            email_id_to_reply_to: Email ID (RFC Message-ID) to reply to for threading.
                Required for reply_all, or auto-inferred from most recent inbound email.
            attachment_filepath: Optional filepath to attach.
        """
        import base64
        import os

        from unity.session_details import SESSION_DETAILS

        # Coerce each recipient item to (contact_id, optional_inline_email).
        # Accepts: int, str (string-encoded int), or dict with contact_id + email_address.
        def _coerce_recipients(
            items: list | None,
        ) -> list[tuple[int, str | None]] | None:
            if items is None:
                return None
            result: list[tuple[int, str | None]] = []
            for item in items:
                if isinstance(item, dict):
                    cid = item.get("contact_id")
                    if cid is None:
                        raise TypeError(
                            f"Email recipient dict must include 'contact_id', got: {item!r}",
                        )
                    result.append((_coerce_contact_id(cid), item.get("email_address")))
                else:
                    result.append((_coerce_contact_id(item), None))
            return result

        to = _coerce_recipients(to)
        cc = _coerce_recipients(cc)
        bcc = _coerce_recipients(bcc)

        _email_topic = "app:comms:email_sent"
        # Best-effort contact_id for thread placement of email errors.
        _email_cid = (
            (to[0][0] if to else None)
            or (cc[0][0] if cc else None)
            or (bcc[0][0] if bcc else None)
        )
        _email_err = dict(contact_id=_email_cid, medium=Medium.EMAIL)

        # --- Validation: reply_all is mutually exclusive with to/cc/bcc ---
        if reply_all and (to or cc or bcc):
            error_msg = (
                "reply_all=True is mutually exclusive with to/cc/bcc. "
                "Either use reply_all to auto-populate recipients from the thread, "
                "or specify recipients explicitly."
            )
            return await self._surface_comms_error(
                error_msg,
                _email_topic,
                **_email_err,
            )

        # --- Helper: resolve recipients to unique (email, contact) pairs ---
        def _resolve_recipients(
            recipients: list[tuple[int, str | None]] | None,
        ) -> tuple[str | None, list[tuple[str, dict]]]:
            """Resolve recipients to (email_address, contact_dict) pairs.

            Each item is ``(contact_id, optional_inline_email)``.  Uses
            :func:`_resolve_or_attach_detail` for implicit detail creation.

            Returns ``(error, resolved)`` -- if *error* is not ``None`` the
            whole send should be aborted.
            """
            if not recipients:
                return (None, [])
            results: dict[str, dict] = {}  # email -> contact, for deduplication
            for cid, inline_email in recipients:
                contact = self._cm.contact_index.get_contact(cid)
                err, resolved = _resolve_or_attach_detail(
                    contact,
                    cid,
                    "email_address",
                    inline_email,
                    "email",
                    self._cm.contact_index,
                )
                if err:
                    return (err, [])
                if resolved:
                    email = resolved.get("email_address")
                    if email and email not in results:
                        results[email] = resolved
            return (None, [(e, c) for e, c in results.items()])

        # --- Handle reply_all: populate to/cc from the email being replied to ---
        final_to: list[str] = []
        final_cc: list[str] = []
        final_bcc: list[str] = []
        reply_email_id = email_id_to_reply_to
        primary_contact: dict | None = None  # For EmailSent event

        if reply_all:
            # Find the email to reply to
            original_email = None
            # Search the global thread for the email with this ID
            all_emails = [
                e.message
                for e in self._cm.contact_index.global_thread
                if e.medium == Medium.EMAIL
            ]
            if reply_email_id:
                for m in all_emails:
                    if getattr(m, "email_id", None) == reply_email_id:
                        original_email = m
                        break
            else:
                # Auto-infer: find the most recent inbound email with matching subject
                for m in reversed(all_emails):
                    if getattr(m, "name", None) != "You" and getattr(
                        m,
                        "email_id",
                        None,
                    ):
                        # Check subject match (strip "Re: " prefix for comparison)
                        m_subject = getattr(m, "subject", "") or ""
                        clean_subject = subject.removeprefix("Re: ").strip()
                        clean_m_subject = m_subject.removeprefix("Re: ").strip()
                        if clean_subject == clean_m_subject or not clean_subject:
                            original_email = m
                            reply_email_id = m.email_id
                            break

            if not original_email:
                return await self._surface_comms_error(
                    "reply_all=True but no email found to reply to. "
                    "Either provide email_id_to_reply_to or ensure there's a matching "
                    "inbound email in the thread.",
                    _email_topic,
                    **_email_err,
                )

            # Standard reply-all behavior:
            # - Original sender -> to
            # - Original to + cc (minus self) -> cc
            assistant_email = SESSION_DETAILS.assistant.email
            original_to = getattr(original_email, "to", []) or []
            original_cc = getattr(original_email, "cc", []) or []

            # The sender goes to "to" - we need to find the sender email
            # For inbound emails, the sender is in the contact associated with the email
            # We can find it from the conversation state's contact
            sender_email = None
            for entry in self._cm.contact_index.global_thread:
                if entry.message is original_email:
                    # Find the contact who was the sender
                    for cid, role in entry.contact_roles.items():
                        if role == "sender":
                            contact = self._cm.contact_index.get_contact(cid)
                            if contact:
                                sender_email = contact.get("email_address")
                                primary_contact = contact
                            break
                    break

            if sender_email:
                final_to = [sender_email]

            # Original to + cc (minus self) go to cc
            all_original_recipients = set(original_to) | set(original_cc)
            if assistant_email:
                all_original_recipients.discard(assistant_email)
            if sender_email:
                all_original_recipients.discard(sender_email)
            final_cc = list(all_original_recipients)

        else:
            # --- Resolve explicit recipients to email addresses ---
            to_err, to_resolved = _resolve_recipients(to)
            if to_err:
                return await self._surface_comms_error(
                    to_err,
                    _email_topic,
                    **_email_err,
                )

            cc_err, cc_resolved = _resolve_recipients(cc)
            if cc_err:
                return await self._surface_comms_error(
                    cc_err,
                    _email_topic,
                    **_email_err,
                )

            bcc_err, bcc_resolved = _resolve_recipients(bcc)
            if bcc_err:
                return await self._surface_comms_error(
                    bcc_err,
                    _email_topic,
                    **_email_err,
                )

            # Extract just the email addresses for sending
            final_to = [email for email, _ in to_resolved]
            final_cc = [email for email, _ in cc_resolved]
            final_bcc = [email for email, _ in bcc_resolved]

            # Keep track of primary contact for the event
            primary_contact = None
            if to_resolved:
                primary_contact = to_resolved[0][1]
            elif cc_resolved:
                primary_contact = cc_resolved[0][1]
            elif bcc_resolved:
                primary_contact = bcc_resolved[0][1]

            # --- Validation: at least one recipient required ---
            if not final_to and not final_cc and not final_bcc:
                return await self._surface_comms_error(
                    "At least one recipient is required. "
                    "Provide to, cc, or bcc, or use reply_all=True.",
                    _email_topic,
                    **_email_err,
                )

            # --- Infer reply ID from email thread if not provided ---
            if not reply_email_id:
                try:
                    all_emails = [
                        e.message
                        for e in self._cm.contact_index.global_thread
                        if e.medium == Medium.EMAIL
                    ]
                    for m in reversed(all_emails):
                        if (
                            getattr(m, "name", None) != "You"
                            and getattr(m, "subject", None) == subject
                            and getattr(m, "email_id", None)
                        ):
                            reply_email_id = m.email_id
                            break
                except Exception:
                    pass

        # --- Handle subject prefix for replies ---
        final_subject = subject
        if reply_email_id and not subject.startswith("Re: "):
            final_subject = f"Re: {subject}"

        # --- Handle attachment ---
        attachment = None
        attachment_filename = None
        if attachment_filepath:
            try:
                from unity.file_manager.filesystem_adapters.local_adapter import (
                    LocalFileSystemAdapter,
                )

                adapter = LocalFileSystemAdapter()
                file_ref = adapter.get_file(attachment_filepath)
                abs_path = adapter._abspath(attachment_filepath)
                with open(abs_path, "rb") as f:
                    file_contents = f.read()

                max_size_mb = 25
                file_size_mb = len(file_contents) / (1024 * 1024)
                if file_size_mb > max_size_mb:
                    return await self._surface_comms_error(
                        f"File too large: {file_size_mb:.1f}MB exceeds {max_size_mb}MB limit",
                        _email_topic,
                        **_email_err,
                    )

                attachment_filename = os.path.basename(attachment_filepath)
                attachment = {
                    "filename": attachment_filename,
                    "content_base64": base64.b64encode(file_contents).decode("utf-8"),
                }
            except FileNotFoundError:
                return await self._surface_comms_error(
                    f"File not found: {attachment_filepath}",
                    _email_topic,
                    **_email_err,
                )
            except Exception as e:
                return await self._surface_comms_error(
                    f"Failed to read file: {e}",
                    _email_topic,
                    **_email_err,
                )

        # --- Send the email ---
        response = await comms_utils.send_email_via_address(
            to=final_to,
            subject=final_subject,
            body=body,
            cc=final_cc if final_cc else None,
            bcc=final_bcc if final_bcc else None,
            email_id=reply_email_id,
            attachment=attachment,
        )

        if response["success"]:
            # Use the primary contact we resolved earlier (or empty dict for reply_all fallback)
            event = EmailSent(
                contact=primary_contact or {},
                body=body,
                subject=final_subject,
                email_id_replied_to=reply_email_id,
                attachments=[attachment_filename] if attachment_filename else [],
                to=final_to,
                cc=final_cc,
                bcc=final_bcc,
            )
            await self._event_broker.publish(_email_topic, event.to_json())
            return {"status": "ok"}

        if not self._cm.assistant_email:
            error_msg = "You don't have an email address, please provision one."
        else:
            recipients = final_to + final_cc + final_bcc
            error_msg = response.get(
                "error",
                f"Failed to send email to {recipients}",
            )
        return await self._surface_comms_error(
            error_msg,
            _email_topic,
            **_email_err,
        )

    async def make_call(
        self,
        *,
        contact_id: int | str,
        context: str,
        phone_number: str | None = None,
    ) -> dict[str, Any]:
        """
        Start an outbound phone call to an existing contact.

        The contact must already exist in the system.

        - If the contact **already has** a phone number on file (visible in
          active_conversations), omit ``phone_number`` -- it is not needed.
        - If the contact **does not have** a phone number on file but you
          know it (e.g. the boss provided it), pass it via ``phone_number``.
          It will be saved to the contact record automatically and the call
          will be placed in one step.
        - **Do not** pass a ``phone_number`` that differs from the one
          already on file -- this will be rejected.  Use ``act`` to update
          the contact's phone number first, then retry.

        Args:
            contact_id: The contact_id of the person to call (from
                active_conversations or returned by ``find_contacts`` /
                ``create_contact``).
            context: **Mission briefing for the voice agent.** This is the
                voice agent's sole source of context about what to do on the
                call. Once the call connects, the voice agent speaks first
                and will not receive any further guidance from you until the
                other person responds or an external event arrives. Everything
                the voice agent needs to open and conduct the conversation
                must be in this string.

                Include:
                - **Purpose**: Why are we calling? What is the goal?
                - **Key information**: Specific facts, names, dates, or
                  details the voice agent needs (e.g. "the meeting is
                  Thursday at 3pm at the downtown office").
                - **Questions to ask**: What specific information do we need
                  from the other person?
                - **Tone / relationship**: How does the boss know this person?
                  Any relevant social context (e.g. "this is a close friend"
                  vs "this is a new business contact").
                - **Constraints**: Anything to avoid saying, sensitive topics,
                  or fallback behavior if the person is unavailable or
                  confused.

                Be thorough — a well-briefed voice agent produces a natural,
                purposeful conversation. A vague context produces an awkward
                opening.

                Example: "Call to confirm the Thursday 3pm meeting with the
                design team at the downtown office. Ask if Sarah has any
                dietary preferences for the team lunch we're ordering. She's
                a senior designer and long-time colleague — friendly and
                informal tone is fine. If she can't make Thursday, ask what
                times work Friday instead."
            phone_number: The recipient's phone number.  Required when the
                contact does not yet have a phone number on file; omit when
                the contact already has one.
        """
        contact_id = _coerce_contact_id(contact_id)
        contact = self._cm.contact_index.get_contact(contact_id)

        outbound_error = _check_outbound_allowed(contact)
        if outbound_error:
            return await self._surface_comms_error(
                outbound_error,
                "app:comms:make_call",
                contact_id=contact_id,
                medium=Medium.PHONE_CALL,
            )

        detail_error, contact = _resolve_or_attach_detail(
            contact,
            contact_id,
            "phone_number",
            phone_number,
            "phone call",
            self._cm.contact_index,
        )
        if detail_error:
            return await self._surface_comms_error(
                detail_error,
                "app:comms:make_call",
                contact_id=contact_id,
                medium=Medium.PHONE_CALL,
            )

        to_number = contact.get("phone_number")
        LOGGER.debug(
            f"{DEFAULT_ICON} [make_call] context: {context}, to_number: {to_number}",
        )
        # Store initial notification so CallManager can publish it to the fast
        # brain after the subprocess spawns (before the recipient picks up).
        if context:
            self._cm.call_manager.initial_notification = context
        response = await comms_utils.start_call(to_number=to_number)
        if response["success"]:
            fresh_contact = (
                self._cm.contact_index.get_contact(phone_number=to_number)
                or contact
                or {}
            )
            event = PhoneCallSent(contact=fresh_contact)
            await self._event_broker.publish("app:comms:make_call", event.to_json())
            return {"status": "ok"}

        if not self._cm.assistant_number:
            error_msg = "You don't have a number, please provision one."
        else:
            error_msg = f"Failed to send call to {to_number}"
        return await self._surface_comms_error(
            error_msg,
            "app:comms:make_call",
            contact_id=contact_id,
            medium=Medium.PHONE_CALL,
        )

    async def act(
        self,
        *,
        query: str,
        response_format: Optional[dict] = None,
        persist: bool = False,
        include_conversation_context: bool = True,
    ) -> dict[str, Any]:
        """
        Engage with knowledge, resources, and the world beyond immediate conversations.

        This is the all-purpose method for any work that requires searching, retrieving,
        manipulating, or acting on information. Use ``act`` liberally — if it cannot
        help, it will simply report back. There is no penalty for speculative delegation.

        **Capabilities include:**

        - **Retrieval**: Search contact records, query knowledge bases, look up past
          conversations, find calendar events, search the web, retrieve files
        - **Action**: Update records, modify spreadsheets, control the desktop/web interface,
          schedule tasks, create reminders
        - **Combined**: Find information and act on it (e.g., "find David's email")

        **When uncertain, call ``act``**: If you need information you don't have (like
        a contact's email address), call ``act`` to search for it. If ``act`` can't find
        it, it will tell you, and you can then ask the user.

        Args:
            query: Natural language request specifying what to do or find.
            response_format: An optional structured schema describing the shape of
                the result you need back.  When provided, the action is required to
                return a JSON object conforming to this schema (via a dedicated
                ``final_response`` tool) instead of free-form text.

                The schema uses a concise format where keys are field names and
                values describe their types:

                - Type strings: ``"string"``, ``"integer"``, ``"number"``,
                  ``"boolean"`` (shorthand ``"str"``, ``"int"``, etc. also work).
                - Nested objects: use a dict value, e.g.
                  ``{"address": {"city": "string", "zip": "string"}}``.
                - Arrays: use a single-element list whose element defines the item
                  schema, e.g. ``[{"name": "string", "email": "string"}]``.

                **Examples:**

                - Simple flat fields::

                      {"email": "string", "phone": "string"}

                - Nested with array::

                      {"contacts": [{"name": "string", "email": "string"}],
                       "total_count": "integer"}

                When omitted (the default), the action returns free-form text and
                the result is whatever the actor decides to report.
            persist: If True, the action runs as a **persistent session** that does
                not self-complete.  The actor stays alive after each response and
                waits for the next ``interject`` before continuing.  Use this for
                long-running interactive sessions (e.g. guided onboarding, live
                screen-sharing walkthroughs, multi-step workflows with a tight
                feedback loop between conversation and action).

                **Key differences from the default (persist=False):**

                - The action will **never** complete on its own.  You must
                  explicitly call ``stop_*`` to end the session.
                - Intermediate responses from the actor appear as **response**
                  events in the action's history (marked ``awaiting_input``).
                  Each response means the actor has finished its current turn
                  and is waiting for your next instruction via ``interject_*``.
                - Progress updates (notifications) may still arrive while the
                  actor is working, before it sends a response.

                The default (False) is a one-shot task: the actor works until
                done and the result arrives as an ``ActorResult``.
            include_conversation_context: Whether to pass the current conversation
                state to the action. When ``true`` (default), the action receives
                the full rendered conversation snapshot — messages, notifications,
                and in-flight actions — helping it understand the broader context.
                Set ``false`` when the action is self-contained and the query
                alone provides all necessary information (e.g. simple lookups,
                web searches, or factual questions). Subsequent steering calls
                (interject, ask) on this action will also skip context forwarding.
        """
        global _next_handle_id

        import time as _bat_time

        _bat_t0 = _bat_time.perf_counter()

        def _bat_ms() -> str:
            return f"{(_bat_time.perf_counter() - _bat_t0) * 1000:.0f}ms"

        import logging as _bat_logging

        _bat_log = _bat_logging.getLogger("unity")
        _bat_log.debug(f"⏱️ [CM.act tool +{_bat_ms()}] entered")

        # Pass the fresh rendered state snapshot as context for the Actor,
        # unless the LLM opted out.
        parent_context = None
        if include_conversation_context:
            parent_context = (
                [_filter_cm_state_for_actor(self._cm._current_state_snapshot)]
                if self._cm._current_state_snapshot
                else None
            )
        _bat_log.debug(f"⏱️ [CM.act tool +{_bat_ms()}] parent context built")

        # Convert the LLM-provided schema dict into a Pydantic model that the
        # Actor's async tool loop uses for structured output validation.
        pydantic_response_format = None
        if response_format is not None:
            pydantic_response_format = schema_dict_to_pydantic(response_format)

        # Invoke the actor. If managers are already initialized, run
        # directly so in_flight_actions is populated before we return.
        # Otherwise queue via listen_to_operations() which defers until
        # initialization completes.
        cm = self._cm

        async def _invoke_actor():
            _bat_log.debug(f"⏱️ [CM.act tool +{_bat_ms()}] calling cm.actor.act()")
            handle = await cm.actor.act(
                query,
                _parent_chat_context=parent_context,
                response_format=pydantic_response_format,
                persist=persist,
            )
            _bat_log.debug(
                f"⏱️ [CM.act tool +{_bat_ms()}] cm.actor.act() returned handle",
            )

            # Capture the snapshot state for incremental diff computation.
            # This is used when interjecting to send only changed state, avoiding duplication.
            initial_snapshot_state: SnapshotState | None = None
            if hasattr(self._cm, "_current_snapshot_state"):
                initial_snapshot_state = self._cm._current_snapshot_state

            self._cm.in_flight_actions[handle_id] = {
                "handle": handle,
                "query": query,
                "persist": persist,
                "action_type": "act",
                "handle_actions": [
                    {
                        "action_name": "act_started",
                        "query": query,
                        "timestamp": prompt_now(),
                    },
                ],
                "initial_snapshot_state": initial_snapshot_state,
                "context_opted_in": include_conversation_context,
            }
            asyncio.create_task(managers_utils.actor_watch_result(handle_id, handle))
            asyncio.create_task(
                managers_utils.actor_watch_notifications(handle_id, handle),
            )
            asyncio.create_task(
                managers_utils.actor_watch_clarifications(handle_id, handle),
            )
            _bat_log.debug(f"⏱️ [CM.act tool +{_bat_ms()}] watchers started")

        handle_id = _next_handle_id
        _next_handle_id += 1

        if cm.initialized:
            await _invoke_actor()
        else:
            await managers_utils.queue_operation(_invoke_actor)

        _bat_log.debug(f"⏱️ [CM.act tool +{_bat_ms()}] publishing ActorHandleStarted")
        await self._event_broker.publish(
            f"app:actor:actor_started_handle_{handle_id}",
            ActorHandleStarted(
                handle_id=handle_id,
                action_name="act",
                query=query,
                response_format=response_format,
            ).to_json(),
        )
        _bat_log.debug(f"⏱️ [CM.act tool +{_bat_ms()}] done, returning")

        return {"status": "acting", "query": query}

    async def _invoke_manager_action(
        self,
        *,
        manager: Any,
        method_name: str,
        text: str,
        action_type: str,
        response_format: Optional[dict] = None,
        include_conversation_context: bool = True,
    ) -> dict[str, Any]:
        """Shared lifecycle for direct manager tools (contact and transcript actions).

        Follows the same pattern as ``act``: queue invocation, store handle in
        ``in_flight_actions``, spawn watcher tasks, publish started event.
        """
        global _next_handle_id
        LOGGER.info(
            f"{ICONS['fast_path']} [FastPath] {action_type}: {text}",
        )

        parent_context = None
        if include_conversation_context:
            parent_context = (
                [_filter_cm_state_for_actor(self._cm._current_state_snapshot)]
                if self._cm._current_state_snapshot
                else None
            )

        pydantic_response_format = None
        if response_format is not None:
            pydantic_response_format = schema_dict_to_pydantic(response_format)

        cm = self._cm

        async def _invoke():
            method = getattr(manager, method_name)
            handle = await method(
                text,
                response_format=pydantic_response_format,
                _parent_chat_context=parent_context,
            )

            initial_snapshot_state: SnapshotState | None = None
            if hasattr(cm, "_current_snapshot_state"):
                initial_snapshot_state = cm._current_snapshot_state

            cm.in_flight_actions[handle_id] = {
                "handle": handle,
                "query": text,
                "persist": False,
                "action_type": action_type,
                "handle_actions": [
                    {
                        "action_name": f"{action_type}_started",
                        "query": text,
                        "timestamp": prompt_now(),
                    },
                ],
                "initial_snapshot_state": initial_snapshot_state,
                "context_opted_in": include_conversation_context,
            }
            asyncio.create_task(
                managers_utils.actor_watch_result(handle_id, handle),
            )
            asyncio.create_task(
                managers_utils.actor_watch_notifications(handle_id, handle),
            )
            asyncio.create_task(
                managers_utils.actor_watch_clarifications(handle_id, handle),
            )

        handle_id = _next_handle_id
        _next_handle_id += 1

        if cm.initialized:
            await _invoke()
        else:
            await managers_utils.queue_operation(_invoke)

        await self._event_broker.publish(
            f"app:actor:actor_started_handle_{handle_id}",
            ActorHandleStarted(
                handle_id=handle_id,
                action_name=action_type,
                query=text,
                response_format=response_format,
            ).to_json(),
        )

        return {"status": "acting", "query": text}

    async def ask_about_contacts(
        self,
        *,
        text: str,
        response_format: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Query contact records directly — names, emails, phone numbers, roles,
        relationships, and any other stored contact attributes.

        This is a **direct channel** to the contact management system, bypassing
        the general ``act`` pathway. Use it for any purely contact-related
        questions:

        - Looking up a specific contact's details
        - Finding contacts by attribute (role, location, company, etc.)
        - Checking if a contact exists
        - Listing or filtering contacts
        - Comparing contact records

        **Route here instead of ``act`` when the question is purely about
        contact data.** If the question also involves non-contact information
        (tasks, knowledge, transcripts, web, files, etc.) or requires
        cross-domain reasoning, use ``act`` instead.

        Args:
            text: Natural language question about contacts
                (e.g. "What is Sarah's email address?").
            response_format: Optional structured schema describing the shape of
                the result you need back. Same format as ``act``'s
                ``response_format`` — keys are field names, values are type
                strings (``"string"``, ``"integer"``, etc.), nested dicts, or
                single-element lists for arrays. When omitted, a free-form text
                answer is returned.
        """
        return await self._invoke_manager_action(
            manager=self._cm.contact_manager,
            method_name="ask",
            text=text,
            action_type="ask_about_contacts",
            response_format=response_format,
        )

    async def update_contacts(
        self,
        *,
        text: str,
        response_format: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Create, edit, delete, or merge contact records directly.

        This is a **direct channel** to the contact management system, bypassing
        the general ``act`` pathway. Use it for any purely contact-related
        mutations:

        - Creating new contacts
        - Updating contact details (phone, email, address, role, bio, etc.)
        - Deleting contacts
        - Merging duplicate contacts

        **Route here instead of ``act`` when the request is purely about
        modifying contacts.** If the request also involves non-contact work
        or cross-domain operations, use ``act`` instead.

        Args:
            text: Natural language description of the contact change
                (e.g. "Add a new contact for John Smith, email john@acme.com").
            response_format: Optional structured schema describing the shape of
                the result you need back. Same format as ``act``'s
                ``response_format``. When omitted, a free-form text summary of
                the mutation is returned.
        """
        return await self._invoke_manager_action(
            manager=self._cm.contact_manager,
            method_name="update",
            text=text,
            action_type="update_contacts",
            response_format=response_format,
        )

    async def query_past_transcripts(
        self,
        *,
        text: str,
        response_format: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Search and analyse past messages and conversation history directly.

        This is a **direct channel** to the transcript store, bypassing the
        general ``act`` pathway. Use it for any purely transcript-related
        questions:

        - Retrieving recent messages from a specific contact or channel
        - Searching past conversations for a keyword or topic
        - Summarising what was discussed in a previous exchange
        - Checking what someone said or when they last messaged
        - Comparing or filtering messages by date, medium, or sender

        **Route here instead of ``act`` when the question is purely about
        past messages or conversation history.** If the question also involves
        non-transcript information (contacts, knowledge, tasks, web, files,
        etc.) or requires cross-domain reasoning, use ``act`` instead.

        Args:
            text: Natural language question about past transcripts
                (e.g. "What did Bob say about the deadline yesterday?").
            response_format: Optional structured schema describing the shape of
                the result you need back. Same format as ``act``'s
                ``response_format`` — keys are field names, values are type
                strings (``"string"``, ``"integer"``, etc.), nested dicts, or
                single-element lists for arrays. When omitted, a free-form text
                answer is returned.
        """
        return await self._invoke_manager_action(
            manager=self._cm.transcript_manager,
            method_name="ask",
            text=text,
            action_type="query_past_transcripts",
            response_format=response_format,
        )

    # ── Desktop fast-path tools ───────────────────────────────────────────

    async def _silent_interject_desktop_act_sessions(
        self,
        message: str,
    ) -> None:
        """Send a silent interjection to every in-flight ``act`` session,
        keeping the Actor informed without triggering an immediate LLM turn."""
        for hid, data in list(self._cm.in_flight_actions.items()):
            if data.get("action_type") != "act":
                continue
            handle = data.get("handle")
            if handle and not handle.done():
                try:
                    await handle.interject(
                        message,
                        trigger_immediate_llm_turn=False,
                    )
                except TypeError:
                    await handle.interject(message)

    async def _invoke_desktop_action(
        self,
        *,
        coro,
        text: str,
        action_type: str,
    ) -> dict[str, Any]:
        """Shared lifecycle for desktop fast-path tools.

        Runs the desktop primitive call in the background and registers it as
        an in-flight action with the same lifecycle as ``act`` and the contact/
        transcript fast paths: ``ActorHandleStarted`` event, watcher tasks for
        result delivery, and automatic cleanup on completion.

        Interjects in-flight Actor desktop sessions twice:
        1. Immediately when the request is made (so the Actor knows what's happening)
        2. After the primitive completes with the result
        """
        global _next_handle_id
        LOGGER.info(
            f"{ICONS['fast_path']} [FastPath] {action_type}: {text}",
        )

        cm = self._cm
        handle_id = _next_handle_id
        _next_handle_id += 1

        async def _run():
            result = await coro
            summary = str(result) if result is not None else "done"
            LOGGER.info(
                f"{ICONS['fast_path']} [FastPath] {action_type} completed:\n"
                f"{summary}",
            )
            await self._silent_interject_desktop_act_sessions(
                f"[FYI — already done] The outer process executed "
                f'{action_type}("{text}") via a direct fast path. '
                f"Result: {result}\n"
                f"No action needed from you — this is for awareness only. "
                f"If you have relevant context (e.g. a stored skill that "
                f"should be used instead, or a reason to adjust your own "
                f"desktop plan), you may act on it; otherwise treat as a "
                f"no-op.",
            )
            return str(result) if result is not None else "done"

        handle = _DesktopActionHandle(asyncio.create_task(_run()))

        cm.in_flight_actions[handle_id] = {
            "handle": handle,
            "query": text,
            "persist": False,
            "action_type": action_type,
            "handle_actions": [
                {
                    "action_name": f"{action_type}_started",
                    "query": text,
                    "timestamp": prompt_now(),
                },
            ],
            "initial_snapshot_state": getattr(cm, "_current_snapshot_state", None),
            "context_opted_in": False,
        }
        asyncio.create_task(managers_utils.actor_watch_result(handle_id, handle))

        await self._event_broker.publish(
            f"app:actor:actor_started_handle_{handle_id}",
            ActorHandleStarted(
                handle_id=handle_id,
                action_name=action_type,
                query=text,
            ).to_json(),
        )

        await self._silent_interject_desktop_act_sessions(
            f"[FYI — being handled] The outer process is executing "
            f'{action_type}("{text}") via a direct fast path. '
            f"Do NOT replicate this action — it is already in progress. "
            f"You will receive the result shortly. If this conflicts with "
            f"your current plan, you may adjust accordingly.",
        )

        return {"status": "acting", "query": text}

    async def desktop_act(
        self,
        *,
        instruction: str,
    ) -> dict[str, Any]:
        """Execute a single atomic action on the assistant's desktop.

        This is a **direct shortcut** to the desktop agent, bypassing the
        general ``act`` pathway.  The action runs in the background and its
        result is delivered asynchronously (same lifecycle as ``act``).

        Use for **single atomic actions** where the user has explicitly
        described both the action and the target:

        - "Click the blue Submit button"
        - "Type 'hello world' into the search box"
        - "Scroll down"
        - "Press Enter"

        **Route through ``act`` instead** when the request requires reasoning
        about *what* to do, involves multiple steps, or benefits from guidance
        and compositional functions.

        Args:
            instruction: A concrete desktop action to perform
                (e.g. "Click the Submit button").
        """
        cp = self._cm.computer_primitives
        return await self._invoke_desktop_action(
            coro=cp.desktop.act(instruction),
            text=instruction,
            action_type="desktop_act",
        )

    # ── Web fast-path tools ────────────────────────────────────────────

    def _resolve_or_create_web_session(self, session_id: int | None):
        """Return (handle, is_new) for an existing or freshly-created session."""
        cp = self._cm.computer_primitives

        async def _resolve():
            if session_id is not None:
                for h in cp.web.list_sessions():
                    if h.session_id == session_id and h.active:
                        return h, False
                raise ValueError(
                    f"No active web session with id {session_id}. "
                    f"Check <active_web_sessions> for valid IDs.",
                )
            handle = await cp.web.new_session(visible=True)
            return handle, True

        return _resolve()

    async def web_act(
        self,
        *,
        request: str,
        session_id: int | None = None,
    ) -> dict[str, Any]:
        """Execute a request in a visible web browser session.

        This is a **direct shortcut** for browser-only work — searching the
        web, navigating sites, filling web forms, reading web pages, or
        extracting web content.  It bypasses the general ``act`` pathway and
        runs directly against a Chromium browser session visible on the
        desktop.

        A new browser session is created automatically when ``session_id``
        is omitted.  Pass a numeric ``session_id`` from
        ``<active_web_sessions>`` to continue working in an existing session.

        **Use ``desktop_act`` instead** for native desktop actions that
        cannot be done inside a browser (clicking desktop UI, opening native
        apps, terminal commands, file manager operations).

        **Use ``act`` instead** for complex multi-step work, cross-domain
        reasoning, or anything requiring guidance / functions / knowledge.

        Args:
            request: Natural language description of the browser task
                (e.g. "Search Google for 'best CRM software 2025'").
            session_id: Optional numeric ID of an existing active web session
                to reuse.  When omitted a new visible session is created.
        """
        handle, is_new = await self._resolve_or_create_web_session(session_id)
        used_id = handle.session_id
        label = f"[session={used_id}, new={is_new}]"
        return await self._invoke_desktop_action(
            coro=handle.act(request),
            text=f"{request} {label}",
            action_type="web_act",
        )

    async def close_web_session(
        self,
        *,
        session_id: int,
    ) -> dict[str, Any]:
        """Close a visible web browser session to free resources.

        Use after completing browser work to clean up.  Check
        ``<active_web_sessions>`` in the current state for valid IDs.

        Args:
            session_id: The numeric ID of the web session to close.
        """
        cp = self._cm.computer_primitives
        for h in cp.web.list_sessions():
            if h.session_id == session_id and h.active:
                await h.stop()
                return {"status": "closed", "session_id": session_id}
        return {
            "status": "not_found",
            "session_id": session_id,
            "error": "No active web session with that ID.",
        }

    async def set_boss_details(
        self,
        *,
        first_name: str | None = None,
        surname: str | None = None,
        phone_number: str | None = None,
        email_address: str | None = None,
    ) -> dict[str, Any]:
        """
        Update the boss contact's details (contact_id=1).

        Use this when you learn the boss's name, phone number, or email
        address during conversation. Only provided fields are updated;
        omitted fields are left unchanged.

        Updating the boss's email address is especially important — once
        their email is on file and they create an account at unify.ai,
        the assistant will be automatically linked to their account.

        Args:
            first_name: The boss's first name.
            surname: The boss's surname / last name.
            phone_number: The boss's phone number.
            email_address: The boss's email address.
        """
        updates = {
            k: v
            for k, v in {
                "first_name": first_name,
                "surname": surname,
                "phone_number": phone_number,
                "email_address": email_address,
            }.items()
            if v is not None
        }
        if not updates:
            return {"status": "error", "error": "No fields provided to update."}

        self._cm.contact_index.contact_manager.update_contact(
            contact_id=1,
            **updates,
        )
        return {"status": "updated", "updates": updates}

    async def wait(
        self,
        delay: int | None = None,
    ) -> dict[str, Any]:
        """
        Wait for more input without taking any action.

        PREFER THIS TOOL over sending messages in most situations. Call this tool:
        - After completing a request (let the user respond first)
        - When there are no NEW messages requiring response
        - When unsure whether to speak (when in doubt, wait)
        - To let the conversation end naturally

        The user should usually have the last word. Do not send follow-up
        messages, additional information, or "anything else?" prompts unless
        the user explicitly asks for more.

        Parameters
        ----------
        delay : int | None
            Seconds to wait before automatically waking up for another thinking
            turn.  When ``None`` (the default), wait indefinitely until the next
            external event (new message, action completion, etc.).  When set to a
            positive integer, the system schedules a follow-up thinking turn after
            that many seconds — useful for probing a long-running action or
            revisiting a situation after a reasonable interval.
        """
        self._cm._outbound_suppress_gen = self._cm._llm_gen
        return {"status": "waiting", "delay": delay}

    async def guide_voice_agent(
        self,
        *,
        content: str,
        should_speak: bool = False,
        response_text: str = "",
    ) -> dict[str, Any]:
        """
        Relay information to the Voice Agent during a live call.

        Call this tool **in parallel** with your action tool (``wait``, ``act``,
        ``send_sms``, etc.) to send guidance to the Voice Agent. If you have
        nothing to relay, simply omit this tool call entirely.

        **Modes:**

        1. **NOTIFY** (default) — Provide context for the Voice Agent to decide
           how to phrase. The Voice Agent receives this as background context and
           gets an LLM turn to decide whether and how to speak. Use for progress
           updates, supplementary context, or information the Voice Agent can
           articulate better with its conversational context.

        2. **SPEAK** — Provide exact text to speak aloud immediately via TTS,
           bypassing the Voice Agent's LLM. Set ``should_speak=True`` and provide
           ``response_text``. Use when you can write a concise, natural sentence
           the user should hear now (concrete data, completion confirmations).

        3. **BLOCK** — Do not call this tool at all.

        Write ``content`` in the language currently spoken on the call so the
        Voice Agent can relay it without translating.

        Do NOT use this tool to steer conversation style, suggest specific
        dialogue, or micromanage the Voice Agent's approach. Provide data,
        status, and progress — not conversational direction.

        Args:
            content: The guidance content to relay. Examples:
                ``"I found 9 backend engineer openings at OpenAI"``,
                ``"The meeting is confirmed for 3pm Thursday in the downtown office."``
            should_speak: When True, ``response_text`` is spoken aloud via TTS,
                bypassing the fast brain's LLM. Use for concrete data answers,
                completion confirmations, or notifications the user should hear
                immediately. When False (default), the guidance is injected as
                silent context and the fast brain decides whether and how to speak.
            response_text: Exact text to speak aloud when ``should_speak`` is
                True. Must be concise (1-2 sentences), natural, and in the Voice
                Agent's persona (first person, conversational). Examples:
                ``"Your flight's at 6am out of Terminal 2, gate B14."``,
                ``"Done — I've sent the email to Sarah."``.
                Leave empty when ``should_speak`` is False.
        """
        return {"status": "guidance_noted"}

    def as_tools(self) -> dict[str, "Callable[..., Any]"]:
        """Return the static tools dict for start_async_tool_loop."""
        from unity.settings import SETTINGS

        tools: dict[str, Callable[..., Any]] = {
            "send_unify_message": self.send_unify_message,
            "wait": self.wait,
        }
        if self._cm.assistant_number:
            tools["send_sms"] = self.send_sms
            call_in_progress = (
                self._cm.mode.is_voice or self._cm.call_manager._call_proc is not None
            )
            if not call_in_progress:
                tools["make_call"] = self.make_call
        if self._cm.assistant_email:
            tools["send_email"] = self.send_email
        if getattr(self._cm.mode, "is_voice", False):
            tools["guide_voice_agent"] = self.guide_voice_agent
        if SETTINGS.DEMO_MODE:
            tools["set_boss_details"] = self.set_boss_details
        else:
            tools["act"] = self.act
            tools["ask_about_contacts"] = self.ask_about_contacts
            tools["update_contacts"] = self.update_contacts
            tools["query_past_transcripts"] = self.query_past_transcripts
        return tools

    def build_action_steering_tools(self) -> dict[str, "Callable[..., Any]"]:
        """Build dynamic tools for steering in-flight actions.

        Conditionally generates pause/resume tools based on current state:
        - If action is paused: only generate resume_* (skip pause_*)
        - If action is running: only generate pause_* (skip resume_*)
        - If state unknown: only generate pause_* (default to running)
        """
        tools: dict[str, Callable[..., Any]] = {}

        for handle_id, handle_data in (self._cm.in_flight_actions or {}).items():
            query = handle_data.get("query", "")
            short_name = derive_short_name(query)
            handle = handle_data.get("handle")
            handle_actions = handle_data.get("handle_actions", [])

            # Check pause state to conditionally generate pause/resume tools
            is_paused = get_handle_paused_state(handle)

            pending_clarifications = [
                a
                for a in handle_actions
                if a.get("action_name") == "clarification_request"
                and not a.get("response")
            ]

            for op in STEERING_OPERATIONS:
                # Conditionally skip pause/resume based on current state
                # is_paused=True: skip pause, only offer resume
                # is_paused=False or None: skip resume, only offer pause (default to running)
                if op.name == "pause" and is_paused is True:
                    continue  # Already paused, don't offer pause
                if op.name == "resume" and is_paused is not True:
                    continue  # Not paused (running or unknown), don't offer resume

                if op.requires_clarification:
                    for clar in pending_clarifications:
                        call_id = clar.get("call_id", "")
                        suffix = safe_call_id_suffix(call_id)
                        tool_name = build_action_name(
                            op.name,
                            short_name,
                            handle_id,
                            suffix,
                        )
                        tool_fn = self._make_steering_tool(
                            handle_id,
                            handle,
                            op.name,
                            op.param_name,
                            op.get_docstring(),
                            query,
                            call_id,
                        )
                        tools[tool_name] = tool_fn
                else:
                    tool_name = build_action_name(op.name, short_name, handle_id)
                    tool_fn = self._make_steering_tool(
                        handle_id,
                        handle,
                        op.name,
                        op.param_name,
                        op.get_docstring(),
                        query,
                    )
                    tools[tool_name] = tool_fn

        return tools

    def build_completed_action_tools(self) -> dict[str, "Callable[..., Any]"]:
        """Build ask tools for completed actions.

        Completed actions preserve their trajectory and remain available
        for `ask` queries about their execution and results.
        """
        tools: dict[str, Callable[..., Any]] = {}

        for handle_id, handle_data in (self._cm.completed_actions or {}).items():
            query = handle_data.get("query", "")
            short_name = derive_short_name(query)
            handle = handle_data.get("handle")

            # ask tool — query the completed action's trajectory/results
            ask_op = OPERATION_MAP["ask"]
            tool_name = build_action_name(ask_op.name, short_name, handle_id)
            tool_fn = self._make_completed_action_ask_tool(
                handle_id,
                handle,
                ask_op.param_name,
                ask_op.get_docstring(),
                query,
            )
            tools[tool_name] = tool_fn

        return tools

    @staticmethod
    def _extract_tool_param_value(
        *,
        kwargs: dict[str, Any],
        primary_name: str,
        aliases: tuple[str, ...] = (),
    ) -> Any:
        """Extract a tool parameter value from kwargs using primary name then aliases."""
        if not primary_name:
            return ""
        for name in (primary_name, *aliases):
            if name in kwargs:
                return kwargs.get(name, "")
        return ""

    def _make_completed_action_ask_tool(
        self,
        handle_id: int,
        handle: Any,
        param_name: str,
        docstring: str,
        query: str,
    ) -> "Callable[..., Any]":
        """Create an ask tool closure for a completed action."""

        cm = self._cm
        event_broker = cm.event_broker
        ask_param_aliases = tuple(
            name for name in ("question", "query") if name != param_name
        )

        async def ask_completed_action(
            **kwargs: Any,
        ) -> dict[str, Any]:
            param_value = self._extract_tool_param_value(
                kwargs=kwargs,
                primary_name=param_name,
                aliases=ask_param_aliases,
            )

            # Get handle_data from completed_actions
            handle_data = cm.completed_actions.get(handle_id)

            # Record action with pending status
            if handle_data:
                handle_data["handle_actions"].append(
                    {
                        "action_name": f"ask_{handle_id}",
                        "query": param_value,
                        "status": "pending",
                        "timestamp": prompt_now(),
                    },
                )

            _handle = handle
            _param_value = param_value
            _handle_id = handle_id
            _parent_context = (
                [cm._current_state_snapshot] if cm._current_state_snapshot else None
            )

            async def _perform_ask_and_emit():
                try:
                    ask_handle = await _handle.ask(
                        _param_value,
                        _parent_chat_context=_parent_context,
                    )
                    ask_result = await ask_handle.result()
                except Exception as e:
                    ask_result = f"Error: {e}"
                await event_broker.publish(
                    f"app:actor:handle_response_{_handle_id}",
                    ActorHandleResponse(
                        handle_id=_handle_id,
                        action_name="ask",
                        query=_param_value,
                        response=ask_result,
                        call_id="",
                    ).to_json(),
                )

            task = asyncio.create_task(_perform_ask_and_emit())
            cm._pending_steering_tasks.add(task)
            task.add_done_callback(cm._pending_steering_tasks.discard)

            return {
                "status": "ok",
                "operation": "ask",
                "result": (
                    "Query submitted. You will receive another turn "
                    "when the answer is ready."
                ),
            }

        # Build signature with proper parameter name
        if param_name:
            params = [
                inspect.Parameter(
                    param_name,
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=str,
                ),
            ]
        else:
            params = []

        ask_completed_action.__signature__ = inspect.Signature(params)
        base_doc = docstring or "Ask about this completed action."
        ask_completed_action.__doc__ = f"{base_doc}\n\nFor action: {query}"
        return ask_completed_action

    def _make_steering_tool(
        self,
        handle_id: int,
        handle: Any,
        operation: str,
        param_name: str,
        docstring: str,
        query: str,
        call_id: str | None = None,
    ) -> "Callable[..., Any]":
        """Create a closure for an action steering operation."""

        cm = self._cm
        # Use cm.event_broker to ensure the same broker is used throughout
        # (important for test patching)
        event_broker = cm.event_broker

        async def steering_tool(
            **kwargs: Any,
        ) -> dict[str, Any]:
            param_aliases: tuple[str, ...] = ()
            if operation == "ask":
                param_aliases = tuple(
                    name for name in ("question", "query") if name != param_name
                )
            param_value = self._extract_tool_param_value(
                kwargs=kwargs,
                primary_name=param_name,
                aliases=param_aliases,
            )

            handle_data = cm.in_flight_actions.get(handle_id)

            result = ""
            try:
                match operation:
                    case "ask":
                        # Record action with pending status - result will arrive async
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"ask_{handle_id}",
                                    "query": param_value,
                                    "status": "pending",
                                    "timestamp": prompt_now(),
                                },
                            )

                        # Capture values for the closure.
                        # Use the fresh rendered state snapshot (set by _run_llm before tools execute).
                        # Only pass context if the original action opted in.
                        _handle = handle
                        _param_value = param_value
                        _handle_id = handle_id
                        _ctx_opted_in = (
                            handle_data.get("context_opted_in", True)
                            if handle_data
                            else True
                        )
                        _parent_context = None
                        if _ctx_opted_in and cm._current_state_snapshot:
                            _parent_context = [cm._current_state_snapshot]

                        # Spawn background task to perform ask and emit result
                        async def _perform_ask_and_emit():
                            try:
                                # Start the ask operation (does the LLM roundtrip)
                                ask_handle = await _handle.ask(
                                    _param_value,
                                    _parent_chat_context=_parent_context,
                                )
                                # Await the result
                                ask_result = await ask_handle.result()
                            except Exception as e:
                                ask_result = f"Error: {e}"
                            # Emit ActorHandleResponse event to wake brain
                            await event_broker.publish(
                                f"app:actor:handle_response_{_handle_id}",
                                ActorHandleResponse(
                                    handle_id=_handle_id,
                                    action_name="ask",
                                    query=_param_value,
                                    response=ask_result,
                                    call_id="",
                                ).to_json(),
                            )

                        task = asyncio.create_task(_perform_ask_and_emit())
                        cm._pending_steering_tasks.add(task)
                        task.add_done_callback(
                            cm._pending_steering_tasks.discard,
                        )

                        # Return immediately - brain will be woken when result arrives
                        return {
                            "status": "ok",
                            "operation": "ask",
                            "result": (
                                "Query submitted. You will receive another turn "
                                "when the answer is ready."
                            ),
                        }

                    case "interject":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                    "timestamp": prompt_now(),
                                },
                            )

                        # Only compute and send context diffs if the original
                        # action opted into conversation context.
                        parent_context_cont = None
                        _interject_ctx_opted_in = (
                            handle_data.get("context_opted_in", True)
                            if handle_data
                            else True
                        )

                        if _interject_ctx_opted_in:
                            initial_snapshot = (
                                handle_data.get("initial_snapshot_state")
                                if handle_data
                                else None
                            )
                            current_snapshot = getattr(
                                cm,
                                "_current_snapshot_state",
                                None,
                            )

                            if current_snapshot is not None:
                                diff_content = compute_snapshot_diff(
                                    initial_snapshot,
                                    current_snapshot,
                                )
                                if diff_content:
                                    parent_context_cont = [
                                        {
                                            "role": "user",
                                            "content": diff_content,
                                            "_cm_context_diff": True,
                                        },
                                    ]
                            elif cm._current_state_snapshot:
                                parent_context_cont = [cm._current_state_snapshot]

                        await handle.interject(
                            param_value,
                            _parent_chat_context_cont=parent_context_cont,
                        )
                        result = "Interjected successfully"
                    case "stop":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                    "timestamp": prompt_now(),
                                },
                            )
                        await handle.stop(reason=param_value or None)
                        stopped = cm.in_flight_actions.pop(handle_id, None)
                        if stopped:
                            cm.completed_actions[handle_id] = stopped
                        result = "Action stopped"
                    case "pause":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                    "timestamp": prompt_now(),
                                },
                            )
                        await handle.pause()
                        result = "Action paused"
                    case "resume":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                    "timestamp": prompt_now(),
                                },
                            )
                        await handle.resume()
                        result = "Action resumed"
                    case "answer_clarification":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                    "timestamp": prompt_now(),
                                },
                            )
                        if call_id:
                            await handle.answer_clarification(call_id, param_value)
                            result = "Clarification answered"
                        else:
                            result = "No clarification call_id available"
                    case _:
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                    "timestamp": prompt_now(),
                                },
                            )
                        result = f"Unknown operation: {operation}"
            except Exception as e:
                result = f"Error: {e}"

            return {"status": "ok", "operation": operation, "result": result}

        # Copy signature + docstring from the handle's method. Parameters
        # starting with _ are automatically hidden by method_to_schema.
        if handle is not None and hasattr(handle, operation):
            DynamicToolFactory._adopt_signature_and_annotations(
                getattr(handle, operation),
                steering_tool,
            )

        # Append action context so the CM knows which action this tool steers.
        # Preserve the docstring set by _adopt_signature_and_annotations (or
        # fall back to the docstring passed in from SteeringOperation).
        base_doc = inspect.getdoc(steering_tool) or docstring
        steering_tool.__doc__ = f"{base_doc}\n\nFor action: {query}"

        return steering_tool
