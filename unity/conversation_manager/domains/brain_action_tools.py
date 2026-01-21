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
from unity.conversation_manager.task_actions import (
    STEERING_OPERATIONS,
    derive_short_name,
    build_action_name,
    safe_call_id_suffix,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from unity.conversation_manager.conversation_manager import ConversationManager


# Global handle ID counter for task tracking
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
        contact_id: int | None = None,
        contact_details: ContactDetailsEmail | None = None,
        subject: str,
        body: str,
        email_id_to_reply_to: str | None = None,
        attachment_filepath: str | None = None,
    ) -> dict[str, Any]:
        """
        Send an email to a contact, optionally with a file attachment.

        Args:
            contact_id: Target contact_id when known (preferred).
            contact_details: Target identity details when contact_id is unknown.
            subject: Email subject.
            body: Email body.
            email_id_to_reply_to: Optional email id to reply to for threading.
            attachment_filepath: Optional filepath to attach.
        """
        import base64
        import os

        contact = await _get_or_create_contact(self._cm, contact_id, contact_details)

        outbound_error = _check_outbound_allowed(contact)
        if outbound_error:
            event = Error(outbound_error)
            await self._event_broker.publish("app:comms:email_sent", event.to_json())
            return {"status": "error", "error": outbound_error}

        address_error = _check_contact_has_address(contact, "email_address", "email")
        if address_error:
            event = Error(address_error)
            await self._event_broker.publish("app:comms:email_sent", event.to_json())
            return {"status": "error", "error": address_error}

        to_email = contact.get("email_address")

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

        # Infer reply ID from email thread if available
        inferred_reply_id: str | None = None
        try:
            cid = contact.get("contact_id") if contact else contact_id
            conv_state = (
                self._cm.contact_index.get_conversation_state(cid) if cid else None
            )
            if conv_state:
                thread = conv_state.threads.get("email")
                if thread:
                    for m in reversed(thread):
                        if (
                            getattr(m, "name", None) != "You"
                            and getattr(m, "subject", None) == subject
                            and getattr(m, "email_id", None)
                        ):
                            inferred_reply_id = m.email_id
                            break
        except Exception:
            inferred_reply_id = None

        if inferred_reply_id and inferred_reply_id != email_id_to_reply_to:
            email_id_to_reply_to = inferred_reply_id

        response = await comms_utils.send_email_via_address(
            to_email=to_email,
            subject=subject,
            body=body,
            email_id=email_id_to_reply_to,
            attachment=attachment,
        )
        if response["success"]:
            fresh_contact = (
                self._cm.contact_index.get_contact(email=to_email) or contact or {}
            )
            event = EmailSent(
                contact=fresh_contact,
                body=body,
                subject=subject,
                email_id_replied_to=email_id_to_reply_to,
                attachments=[attachment_filename] if attachment_filename else [],
            )
        else:
            if not self._cm.assistant_email:
                error_msg = "You don't have an email address, please provision one."
            else:
                error_msg = response.get("error", f"Failed to send email to {to_email}")
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
        self._cm.active_tasks[handle_id] = {
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

    def build_task_steering_tools(self) -> dict[str, "Callable[..., Any]"]:
        """Build dynamic tools for steering active tasks."""
        tools: dict[str, Callable[..., Any]] = {}

        for handle_id, handle_data in (self._cm.active_tasks or {}).items():
            query = handle_data.get("query", "")
            short_name = derive_short_name(query)
            handle = handle_data.get("handle")
            handle_actions = handle_data.get("handle_actions", [])

            pending_clarifications = [
                a
                for a in handle_actions
                if a.get("action_name") == "clarification_request"
                and not a.get("response")
            ]

            for op in STEERING_OPERATIONS:
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
        """Create a closure for a task steering operation."""
        cm = self._cm
        # Use cm.event_broker to ensure the same broker is used throughout
        # (important for test patching)
        event_broker = cm.event_broker

        async def steering_tool(
            **kwargs: Any,
        ) -> dict[str, Any]:
            param_value = kwargs.get(param_name, "") if param_name else ""

            handle_data = cm.active_tasks.get(handle_id)

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
                                    parent_chat_context_cont=_cm_chat_history,
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
                            parent_chat_context_cont=cm.chat_history,
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
                        result = "Task stopped"
                        cm.active_tasks.pop(handle_id, None)
                    case "pause":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                },
                            )
                        await handle.pause()
                        result = "Task paused"
                    case "resume":
                        if handle_data:
                            handle_data["handle_actions"].append(
                                {
                                    "action_name": f"{operation}_{handle_id}",
                                    "query": param_value,
                                },
                            )
                        await handle.resume()
                        result = "Task resumed"
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

        steering_tool.__doc__ = f"{docstring}\n\nFor task: {query}"
        if param_name:
            steering_tool.__doc__ += f"\n\nArgs:\n    {param_name}: {docstring}"

        return steering_tool
