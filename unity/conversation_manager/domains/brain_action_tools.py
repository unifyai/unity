"""
Brain action tools for ConversationManager.

All contact information is fetched from ContactManager (source of truth).
No local caching of contact data.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from unity.contact_manager.types import ContactDetailsEmail, ContactDetailsPhone
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
    derive_short_name,
    build_action_name,
    safe_call_id_suffix,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from unity.conversation_manager.conversation_manager import ConversationManager


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


async def _get_or_create_contact(
    cm: "ConversationManager",
    contact_id: int | None = None,
    details: ContactDetailsPhone | ContactDetailsEmail | None = None,
) -> dict | None:
    """
    Get an existing contact or create a new one via ContactManager.

    All contact operations go through ContactManager - the source of truth.
    """
    if not contact_id and not details:
        raise ValueError("Either contact_id or details must be provided")

    # Convert Pydantic model to dict
    details_dict: dict | None = None
    if details is not None:
        if isinstance(details, dict):
            details_dict = {k: v for k, v in details.items() if v is not None}
        else:
            details_dict = details.model_dump(exclude_none=True)

    # Get by contact_id
    if contact_id:
        contact = cm.contact_index.get_contact(contact_id)
        if contact:
            return contact

    # Search by phone/email
    if details_dict:
        phone_number = details_dict.get("phone_number")
        email_address = details_dict.get("email_address")

        if phone_number and cm.contact_manager:
            result = cm.contact_manager.filter_contacts(
                filter=f"phone_number == '{phone_number}'",
                limit=1,
            )
            contacts = result.get("contacts", [])
            if contacts:
                c = contacts[0]
                return c.model_dump() if hasattr(c, "model_dump") else c

        if email_address and cm.contact_manager:
            result = cm.contact_manager.filter_contacts(
                filter=f"email_address == '{email_address}'",
                limit=1,
            )
            contacts = result.get("contacts", [])
            if contacts:
                c = contacts[0]
                return c.model_dump() if hasattr(c, "model_dump") else c

        # Create new contact via ContactManager
        if cm.contact_manager:
            tool_outcome = await asyncio.to_thread(
                cm.contact_manager._create_contact,
                **details_dict,
            )
            new_contact_id = tool_outcome["details"]["contact_id"]
            new_contact = await asyncio.to_thread(
                cm.contact_manager.get_contact_info,
                new_contact_id,
            )
            return new_contact.get(new_contact_id)

    return None


class ConversationManagerBrainActionTools:
    """
    Side-effecting tools for the Main CM Brain.

    All contact data is fetched from ContactManager - no local caching.
    """

    def __init__(self, cm: "ConversationManager"):
        self._cm = cm
        self._event_broker = get_event_broker()

    async def send_sms(
        self,
        *,
        contact_id: int | None = None,
        contact_details: ContactDetailsPhone | None = None,
        content: str,
    ) -> dict[str, Any]:
        """
        Send an SMS message to a contact.

        Args:
            contact_id: Target contact_id when known (preferred).
            contact_details: Target identity details when contact_id is unknown.
            content: SMS body to send.
        """
        contact = await _get_or_create_contact(self._cm, contact_id, contact_details)

        outbound_error = _check_outbound_allowed(contact)
        if outbound_error:
            event = Error(outbound_error)
            await self._event_broker.publish("app:comms:sms_sent", event.to_json())
            return {"status": "error", "error": outbound_error}

        address_error = _check_contact_has_address(contact, "phone_number", "SMS")
        if address_error:
            event = Error(address_error)
            await self._event_broker.publish("app:comms:sms_sent", event.to_json())
            return {"status": "error", "error": address_error}

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
        else:
            if not self._cm.assistant_number:
                error_msg = "You don't have a number, please provision one."
            else:
                error_msg = f"Failed to send sms to {to_number}"
            event = Error(error_msg)
        await self._event_broker.publish("app:comms:sms_sent", event.to_json())
        return {"status": "ok"}

    async def send_unify_message(
        self,
        *,
        content: str,
        contact_id: int,
        attachment_filepath: str | None = None,
    ) -> dict[str, Any]:
        """
        Send a Unify message to a contact via the Unify platform.

        Args:
            content: Message content to send.
            contact_id: Target contact_id from active conversations.
            attachment_filepath: Optional filepath to attach.
        """
        import os

        contact = self._cm.contact_index.get_contact(contact_id=contact_id)

        if contact:
            outbound_error = _check_outbound_allowed(contact)
            if outbound_error:
                event = Error(outbound_error)
                await self._event_broker.publish(
                    "app:comms:unify_message_sent",
                    event.to_json(),
                )
                return {"status": "error", "error": outbound_error}

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
                    error_msg = f"File too large: {file_size_mb:.1f}MB exceeds {max_size_mb}MB limit"
                    event = Error(error_msg)
                    await self._event_broker.publish(
                        "app:comms:unify_message_sent",
                        event.to_json(),
                    )
                    return {"status": "error", "error": error_msg}

                attachment_filename = os.path.basename(attachment_filepath)
                upload_result = await comms_utils.upload_unify_attachment(
                    file_content=file_contents,
                    filename=attachment_filename,
                )

                if "error" in upload_result:
                    error_msg = f"Failed to upload attachment: {upload_result['error']}"
                    event = Error(error_msg)
                    await self._event_broker.publish(
                        "app:comms:unify_message_sent",
                        event.to_json(),
                    )
                    return {"status": "error", "error": error_msg}

                attachment = upload_result

            except FileNotFoundError:
                error_msg = f"File not found: {attachment_filepath}"
                event = Error(error_msg)
                await self._event_broker.publish(
                    "app:comms:unify_message_sent",
                    event.to_json(),
                )
                return {"status": "error", "error": error_msg}
            except Exception as e:
                error_msg = f"Failed to read file: {e}"
                event = Error(error_msg)
                await self._event_broker.publish(
                    "app:comms:unify_message_sent",
                    event.to_json(),
                )
                return {"status": "error", "error": error_msg}

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
            event = UnifyMessageSent(
                contact=fresh_contact,
                content=content,
                attachments=[attachment_filename] if attachment_filename else [],
            )
        else:
            event = Error("Failed to send unify message")
        await self._event_broker.publish(
            "app:comms:unify_message_sent",
            event.to_json(),
        )
        return {"status": "ok"}

    async def send_email(
        self,
        *,
        to: list[int | str] | None = None,
        cc: list[int | str] | None = None,
        bcc: list[int | str] | None = None,
        subject: str,
        body: str,
        reply_all: bool = False,
        email_id_to_reply_to: str | None = None,
        attachment_filepath: str | None = None,
    ) -> dict[str, Any]:
        """
        Send an email with flexible recipient specification.

        Recipients can be specified as contact_ids (int) or email addresses (str).
        Duplicates are automatically collapsed (e.g., if you provide both a contact_id
        and the same contact's email address, only one recipient is sent).

        Args:
            to: List of recipients (contact_ids or email addresses).
            cc: List of CC recipients (contact_ids or email addresses).
            bcc: List of BCC recipients (contact_ids or email addresses).
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

        # --- Validation: reply_all is mutually exclusive with to/cc/bcc ---
        if reply_all and (to or cc or bcc):
            error_msg = (
                "reply_all=True is mutually exclusive with to/cc/bcc. "
                "Either use reply_all to auto-populate recipients from the thread, "
                "or specify recipients explicitly."
            )
            event = Error(error_msg)
            await self._event_broker.publish("app:comms:email_sent", event.to_json())
            return {"status": "error", "error": error_msg}

        # --- Helper: resolve a contact_id or email to an email address ---
        def _resolve_to_email(recipient: int | str) -> str | None:
            if isinstance(recipient, str):
                return recipient
            # It's a contact_id - look up the email
            contact = self._cm.contact_index.get_contact(recipient)
            if contact:
                return contact.get("email_address")
            return None

        # --- Helper: resolve a list of recipients to unique email addresses ---
        def _resolve_recipients(recipients: list[int | str] | None) -> list[str]:
            if not recipients:
                return []
            emails = set()
            for r in recipients:
                email = _resolve_to_email(r)
                if email:
                    emails.add(email)
            return list(emails)

        # --- Handle reply_all: populate to/cc from the email being replied to ---
        final_to: list[str] = []
        final_cc: list[str] = []
        final_bcc: list[str] = []
        reply_email_id = email_id_to_reply_to

        if reply_all:
            # Find the email to reply to
            original_email = None
            # Search all conversations for the email with this ID
            if reply_email_id:
                for conv_state in self._cm.contact_index.active_conversations.values():
                    thread = conv_state.threads.get(Medium.EMAIL)
                    if thread:
                        for m in thread:
                            if getattr(m, "email_id", None) == reply_email_id:
                                original_email = m
                                break
                    if original_email:
                        break
            else:
                # Auto-infer: find the most recent inbound email with matching subject
                for conv_state in self._cm.contact_index.active_conversations.values():
                    thread = conv_state.threads.get(Medium.EMAIL)
                    if thread:
                        for m in reversed(list(thread)):
                            if getattr(m, "name", None) != "You" and getattr(
                                m,
                                "email_id",
                                None,
                            ):
                                # Check subject match (strip "Re: " prefix for comparison)
                                m_subject = getattr(m, "subject", "") or ""
                                clean_subject = subject.removeprefix("Re: ").strip()
                                clean_m_subject = m_subject.removeprefix("Re: ").strip()
                                if (
                                    clean_subject == clean_m_subject
                                    or not clean_subject
                                ):
                                    original_email = m
                                    reply_email_id = m.email_id
                                    break
                    if original_email:
                        break

            if not original_email:
                error_msg = (
                    "reply_all=True but no email found to reply to. "
                    "Either provide email_id_to_reply_to or ensure there's a matching "
                    "inbound email in the thread."
                )
                event = Error(error_msg)
                await self._event_broker.publish(
                    "app:comms:email_sent",
                    event.to_json(),
                )
                return {"status": "error", "error": error_msg}

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
            for cid, conv_state in self._cm.contact_index.active_conversations.items():
                thread = conv_state.threads.get(Medium.EMAIL)
                if thread and original_email in thread:
                    contact = self._cm.contact_index.get_contact(cid)
                    if contact:
                        sender_email = contact.get("email_address")
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
            # --- Resolve explicit recipients ---
            final_to = _resolve_recipients(to)
            final_cc = _resolve_recipients(cc)
            final_bcc = _resolve_recipients(bcc)

            # --- Validation: at least one recipient required ---
            if not final_to and not final_cc and not final_bcc:
                error_msg = (
                    "At least one recipient is required. "
                    "Provide to, cc, or bcc, or use reply_all=True."
                )
                event = Error(error_msg)
                await self._event_broker.publish(
                    "app:comms:email_sent",
                    event.to_json(),
                )
                return {"status": "error", "error": error_msg}

            # --- Infer reply ID from email thread if not provided ---
            if not reply_email_id:
                try:
                    # Look for a matching inbound email in any conversation
                    for (
                        conv_state
                    ) in self._cm.contact_index.active_conversations.values():
                        thread = conv_state.threads.get(Medium.EMAIL)
                        if thread:
                            for m in reversed(list(thread)):
                                if (
                                    getattr(m, "name", None) != "You"
                                    and getattr(m, "subject", None) == subject
                                    and getattr(m, "email_id", None)
                                ):
                                    reply_email_id = m.email_id
                                    break
                        if reply_email_id:
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
                    error_msg = f"File too large: {file_size_mb:.1f}MB exceeds {max_size_mb}MB limit"
                    event = Error(error_msg)
                    await self._event_broker.publish(
                        "app:comms:email_sent",
                        event.to_json(),
                    )
                    return {"status": "error", "error": error_msg}

                attachment_filename = os.path.basename(attachment_filepath)
                attachment = {
                    "filename": attachment_filename,
                    "content_base64": base64.b64encode(file_contents).decode("utf-8"),
                }
            except FileNotFoundError:
                error_msg = f"File not found: {attachment_filepath}"
                event = Error(error_msg)
                await self._event_broker.publish(
                    "app:comms:email_sent",
                    event.to_json(),
                )
                return {"status": "error", "error": error_msg}
            except Exception as e:
                error_msg = f"Failed to read file: {e}"
                event = Error(error_msg)
                await self._event_broker.publish(
                    "app:comms:email_sent",
                    event.to_json(),
                )
                return {"status": "error", "error": error_msg}

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
            # Get contact for the first "to" recipient for the event
            primary_email = (
                final_to[0] if final_to else (final_cc[0] if final_cc else None)
            )
            contact = (
                self._cm.contact_index.get_contact(email=primary_email)
                if primary_email
                else {}
            ) or {}
            event = EmailSent(
                contact=contact,
                body=body,
                subject=final_subject,
                email_id_replied_to=reply_email_id,
                attachments=[attachment_filename] if attachment_filename else [],
                to=final_to,
                cc=final_cc,
                bcc=final_bcc,
            )
        else:
            if not self._cm.assistant_email:
                error_msg = "You don't have an email address, please provision one."
            else:
                recipients = final_to + final_cc + final_bcc
                error_msg = response.get(
                    "error",
                    f"Failed to send email to {recipients}",
                )
            event = Error(error_msg)
        await self._event_broker.publish("app:comms:email_sent", event.to_json())
        return {"status": "ok"}

    async def make_call(
        self,
        *,
        contact_id: int | None = None,
        contact_details: ContactDetailsPhone | None = None,
    ) -> dict[str, Any]:
        """
        Start an outbound phone call to a contact.

        Args:
            contact_id: Target contact_id when known (preferred).
            contact_details: Target identity details when contact_id is unknown.
        """
        contact = await _get_or_create_contact(self._cm, contact_id, contact_details)

        outbound_error = _check_outbound_allowed(contact)
        if outbound_error:
            event = Error(outbound_error)
            await self._event_broker.publish("app:comms:make_call", event.to_json())
            return {"status": "error", "error": outbound_error}

        address_error = _check_contact_has_address(
            contact,
            "phone_number",
            "phone call",
        )
        if address_error:
            event = Error(address_error)
            await self._event_broker.publish("app:comms:make_call", event.to_json())
            return {"status": "error", "error": address_error}

        to_number = contact.get("phone_number")
        response = await comms_utils.start_call(to_number=to_number)
        if response["success"]:
            fresh_contact = (
                self._cm.contact_index.get_contact(phone_number=to_number)
                or contact
                or {}
            )
            event = PhoneCallSent(contact=fresh_contact)
        else:
            if not self._cm.assistant_number:
                error_msg = "You don't have a number, please provision one."
            else:
                error_msg = f"Failed to send call to {to_number}"
            event = Error(error_msg)
        await self._event_broker.publish("app:comms:make_call", event.to_json())
        return {"status": "ok"}

    async def act(self, *, query: str) -> dict[str, Any]:
        """
        Engage with knowledge, resources, and the world beyond immediate conversations.

        This is the all-purpose method for any work that requires searching, retrieving,
        manipulating, or acting on information. Use ``act`` liberally — if it cannot
        help, it will simply report back. There is no penalty for speculative delegation.

        **Capabilities include:**

        - **Retrieval**: Search contact records, query knowledge bases, look up past
          conversations, find calendar events, search the web, retrieve files
        - **Action**: Update records, modify spreadsheets, control the desktop/browser,
          schedule tasks, create reminders
        - **Combined**: Find information and act on it (e.g., "find David's email")

        **When uncertain, call ``act``**: If you need information you don't have (like
        a contact's email address), call ``act`` to search for it. If ``act`` can't find
        it, it will tell you, and you can then ask the user.

        Args:
            query: Natural language description of what to do or find.
        """
        global _next_handle_id

        await managers_utils.wait_for_initialization(self._cm)

        handle = await self._cm.actor.act(
            query,
            _parent_chat_context=self._cm.chat_history,
        )

        handle_id = _next_handle_id
        _next_handle_id += 1
        self._cm.in_flight_actions[handle_id] = {
            "handle": handle,
            "query": query,
            "handle_actions": [],
        }

        await self._event_broker.publish(
            f"app:actor:actor_started_handle_{handle_id}",
            ActorHandleStarted(
                handle_id=handle_id,
                action_name="act",
                query=query,
            ).to_json(),
        )

        asyncio.create_task(managers_utils.actor_watch_result(handle_id, handle))
        asyncio.create_task(managers_utils.actor_watch_notifications(handle_id, handle))
        asyncio.create_task(
            managers_utils.actor_watch_clarifications(handle_id, handle),
        )

        return {"status": "acting", "query": query}

    async def wait(self) -> dict[str, Any]:
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
        """
        return {"status": "waiting"}

    def as_tools(self) -> dict[str, "Callable[..., Any]"]:
        """Return the static tools dict for start_async_tool_loop."""
        return {
            "send_sms": self.send_sms,
            "send_unify_message": self.send_unify_message,
            "send_email": self.send_email,
            "make_call": self.make_call,
            "act": self.act,
            "wait": self.wait,
        }

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
            param_value = kwargs.get(param_name, "") if param_name else ""

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
                                },
                            )

                        # Capture values for the closure
                        _handle = handle
                        _param_value = param_value
                        _handle_id = handle_id
                        _cm_chat_history = cm.chat_history

                        # Spawn background task to perform ask and emit result
                        async def _perform_ask_and_emit():
                            try:
                                # Start the ask operation (does the LLM roundtrip)
                                ask_handle = await _handle.ask(
                                    _param_value,
                                    _parent_chat_context_cont=_cm_chat_history,
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

                        asyncio.create_task(_perform_ask_and_emit())

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
                                },
                            )
                        await handle.interject(
                            param_value,
                            _parent_chat_context_cont=cm.chat_history,
                            images=kwargs.get("images"),
                        )
                        result = "Interjected successfully"
                    case "stop":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                },
                            )
                        handle.stop(reason=param_value or None)
                        result = "Action stopped"
                        cm.in_flight_actions.pop(handle_id, None)
                    case "pause":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
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
                                },
                            )
                        result = f"Unknown operation: {operation}"
            except Exception as e:
                result = f"Error: {e}"

            return {"status": "ok", "operation": operation, "result": result}

        # Copy signature from the handle's method to get proper tool schema.
        # Parameters starting with _ (like _parent_chat_context_cont) are automatically
        # hidden by method_to_schema, and images: Optional[ImageRefs] is schema-safe.
        if handle is not None and hasattr(handle, operation):
            DynamicToolFactory._adopt_signature_and_annotations(
                getattr(handle, operation),
                steering_tool,
            )

        # Always set a custom docstring that describes this specific action
        # (overrides any docstring copied from handle, e.g. from MagicMock in tests)
        steering_tool.__doc__ = f"{docstring}\n\nFor action: {query}"
        if param_name:
            steering_tool.__doc__ += f"\n\nArgs:\n    {param_name}: {docstring}"

        return steering_tool
