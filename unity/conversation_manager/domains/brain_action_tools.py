from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from unity.contact_manager.types import ContactDetailsEmail, ContactDetailsPhone
from unity.conversation_manager.domains import comms_utils
from unity.conversation_manager.domains import managers_utils
from unity.conversation_manager.domains.contact_index import Contact
from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.events import (
    SMSSent,
    UnifyMessageSent,
    EmailSent,
    PhoneCallSent,
    ActorHandleStarted,
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


async def _get_or_create_contact(
    cm: "ConversationManager",
    contact_id: int | None = None,
    details: ContactDetailsPhone | ContactDetailsEmail | None = None,
) -> dict:
    """Get an existing contact or create a new one.

    Args:
        cm: The ConversationManager instance.
        contact_id: Contact ID if known.
        details: Contact details for lookup/creation (Pydantic model).

    Returns:
        The contact dict.
    """
    if not contact_id and not details:
        raise ValueError("Either contact_id or details must be provided")

    # Convert Pydantic model to dict for internal use (exclude unset fields)
    # Handle both dict (from JSON tool args) and Pydantic model inputs
    details_dict: dict | None = None
    if details is not None:
        if isinstance(details, dict):
            details_dict = {k: v for k, v in details.items() if v is not None}
        else:
            details_dict = details.model_dump(exclude_none=True)

    # Update existing contact
    if contact_id and details_dict:
        contact = cm.contact_index.get_contact(contact_id=contact_id)
        updated_contacts_raw = cm.contact_manager.get_contact_info(
            contact_id=list(cm.contact_index.contacts.keys()),
        )
        # Update contacts dict with Contact objects
        for cid, uc in updated_contacts_raw.items():
            if cid in cm.contact_index.contacts:
                existing = cm.contact_index.contacts[cid]
                cm.contact_index.contacts[cid] = Contact(
                    **{**existing.model_dump(), **uc, "threads": existing.threads},
                )
            else:
                cm.contact_index.contacts[cid] = Contact(**uc)
        # Update active_conversations similarly
        for cid, c in cm.contact_index.active_conversations.items():
            if cid in updated_contacts_raw:
                uc = updated_contacts_raw[cid]
                cm.contact_index.active_conversations[cid] = Contact(
                    **{**c.model_dump(), **uc, "threads": c.threads},
                )
        phone_number = details_dict.get("phone_number")
        email_address = details_dict.get("email_address")
        contact = (
            cm.contact_index.get_contact(phone_number=phone_number)
            if phone_number
            else cm.contact_index.get_contact(email=email_address)
        )
        return contact

    # Retrieve if exists, create if not
    if details_dict:
        phone_number = details_dict.get("phone_number")
        email_address = details_dict.get("email_address")
        maybe_contact = cm.contact_index.get_contact(
            phone_number=phone_number,
        ) or cm.contact_index.get_contact(email=email_address)
        if maybe_contact:
            return maybe_contact
        tool_outcome = await asyncio.to_thread(
            cm.contact_manager._create_contact,
            **details_dict,
        )
        new_contact_id = tool_outcome["details"]["contact_id"]
        new_contact = await asyncio.to_thread(
            cm.contact_manager.get_contact_info,
            new_contact_id,
        )
        cm.contact_index.contacts[new_contact_id] = Contact(
            **new_contact[new_contact_id],
        )
        return new_contact[new_contact_id]

    # Just retrieve by contact_id
    if contact_id:
        return cm.contact_index.get_contact(contact_id=contact_id)

    raise ValueError("Could not resolve contact")


class ConversationManagerBrainActionTools:
    """
    Side-effecting tools for the Main CM Brain.

    All communication and task management actions are exposed as tool calls,
    following the async tool loop pattern.
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

        Use this when the boss or context indicates SMS is the appropriate channel.
        For active conversations, use contact_id. For new contacts, provide details.

        Args:
            contact_id: Target contact_id when known (preferred).
            contact_details: Target identity details when contact_id is unknown.
            content: SMS body to send.
        """
        contact = await _get_or_create_contact(self._cm, contact_id, contact_details)
        to_number = contact.get("phone_number")
        response = await comms_utils.send_sms_message_via_number(
            to_number=to_number,
            content=content,
        )

        if response["success"]:
            contact = self._cm.contact_index.get_contact(phone_number=to_number)
            event = SMSSent(contact=contact, content=content)
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
    ) -> dict[str, Any]:
        """
        Send a Unify message to a contact via the Unify platform (in-app messaging).

        Use this for contacts who communicate through the Unify app rather than
        SMS/email/phone. Check the contact's available communication channels
        in the active conversation to determine which medium to use.

        Args:
            content: Message content to send.
            contact_id: Target contact_id from active conversations.
        """
        response = await comms_utils.send_unify_message(
            content=content,
            contact_id=contact_id,
        )
        if response["success"]:
            contact = self._cm.contact_index.get_contact(contact_id=contact_id)
            event = UnifyMessageSent(contact=contact, content=content)
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
    ) -> dict[str, Any]:
        """
        Send an email to a contact.

        Use this when the boss or context indicates email is the appropriate channel.

        Args:
            contact_id: Target contact_id when known (preferred).
            contact_details: Target identity details when contact_id is unknown.
            subject: Email subject.
            body: Email body.
            email_id_to_reply_to: Optional email id to reply to for threading.
        """
        contact = await _get_or_create_contact(self._cm, contact_id, contact_details)
        to_email = contact.get("email_address")

        # Prefer the most recent inbound email's Message-ID for this contact+subject,
        # rather than trusting the LLM to copy it correctly.
        inferred_reply_id: str | None = None
        try:
            convo = None
            if contact_id is not None:
                convo = self._cm.contact_index.active_conversations.get(contact_id)
            if convo is None and to_email:
                convo = next(
                    (
                        c
                        for c in self._cm.contact_index.active_conversations.values()
                        if getattr(c, "email_address", None) == to_email
                    ),
                    None,
                )
            if convo is not None:
                thread = getattr(convo, "threads", {}).get("email")
                if thread:
                    for m in reversed(thread):
                        # Prefer the most recent *user* email (name != "You") with the
                        # same subject and a non-empty email_id.
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
        )
        if response["success"]:
            contact = self._cm.contact_index.get_contact(email=to_email)
            event = EmailSent(
                contact=contact,
                body=body,
                subject=subject,
                email_id_replied_to=email_id_to_reply_to,
            )
        else:
            if not self._cm.assistant_email:
                error_msg = "You don't have an email address, please provision one."
            else:
                error_msg = f"Failed to send email to {to_email}"
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

        Use this when the boss explicitly requests to communicate via phone call,
        or when voice communication is clearly the appropriate channel.

        Args:
            contact_id: Target contact_id when known (preferred).
            contact_details: Target identity details when contact_id is unknown.
        """
        contact = await _get_or_create_contact(self._cm, contact_id, contact_details)
        to_number = contact.get("phone_number")
        response = await comms_utils.start_call(to_number=to_number)
        if response["success"]:
            contact = self._cm.contact_index.get_contact(phone_number=to_number)
            event = PhoneCallSent(contact=contact)
        else:
            if not self._cm.assistant_number:
                error_msg = "You don't have a number, please provision one."
            else:
                error_msg = f"Failed to send call to {to_number}"
            event = Error(error_msg)
        await self._event_broker.publish("app:comms:make_call", event.to_json())
        return {"status": "ok"}

    async def start_task(self, *, query: str) -> dict[str, Any]:
        """
        Start a new background task for work not related to direct communication.

        Use this for tasks like searching the web, doing research, answering
        questions, managing contacts, scheduling, or any work that requires
        the Conductor to orchestrate.

        Args:
            query: The task description or question to work on.
        """
        global _next_handle_id

        await managers_utils.wait_for_initialization(self._cm)

        handle = await self._cm.actor.act(
            query,
            _parent_chat_context=self._cm.chat_history,
        )

        # Allocate handle id and register
        handle_id = _next_handle_id
        _next_handle_id += 1
        self._cm.active_tasks[handle_id] = {
            "handle": handle,
            "query": query,
            "handle_actions": [],
        }

        # Publish started event
        await self._event_broker.publish(
            f"app:actor:actor_started_handle_{handle_id}",
            ActorHandleStarted(
                handle_id=handle_id,
                action_name="start_task",
                query=query,
            ).to_json(),
        )

        # Spawn watchers
        asyncio.create_task(managers_utils.actor_watch_result(handle_id, handle))
        asyncio.create_task(managers_utils.actor_watch_notifications(handle_id, handle))
        asyncio.create_task(
            managers_utils.actor_watch_clarifications(handle_id, handle),
        )

        return {"status": "task_started", "query": query}

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
            "start_task": self.start_task,
            "wait": self.wait,
        }

    def build_task_steering_tools(self) -> dict[str, "Callable[..., Any]"]:
        """
        Build dynamic tools for steering active tasks.

        These tools are generated based on the current active_tasks and allow
        the LLM to ask, interject, stop, pause, resume, or answer clarifications
        for running tasks.
        """
        tools: dict[str, Callable[..., Any]] = {}

        for handle_id, handle_data in (self._cm.active_tasks or {}).items():
            query = handle_data.get("query", "")
            short_name = derive_short_name(query)
            handle = handle_data.get("handle")
            handle_actions = handle_data.get("handle_actions", [])

            # Get pending clarifications for this handle
            pending_clarifications = [
                a
                for a in handle_actions
                if a.get("action_name") == "clarification_request"
                and not a.get("response")
            ]

            for op in STEERING_OPERATIONS:
                if op.requires_clarification:
                    # Only generate answer_clarification if there are pending ones
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

        async def steering_tool(
            **kwargs: Any,
        ) -> dict[str, Any]:
            # Extract parameter value
            param_value = kwargs.get(param_name, "") if param_name else ""

            # Record intervention
            handle_data = cm.active_tasks.get(handle_id)
            if handle_data:
                handle_data["handle_actions"].append(
                    {"action_name": f"{operation}_{handle_id}", "query": param_value},
                )

            # Perform the steering operation
            result = ""
            try:
                match operation:
                    case "ask":
                        ask_handle = await handle.ask(
                            param_value,
                            parent_chat_context_cont=cm.chat_history,
                        )
                        result = await ask_handle.result()
                    case "interject":
                        await handle.interject(
                            param_value,
                            parent_chat_context_cont=cm.chat_history,
                        )
                        result = "Interjected successfully"
                    case "stop":
                        handle.stop(reason=param_value or None)
                        result = "Task stopped"
                        cm.active_tasks.pop(handle_id, None)
                    case "pause":
                        await handle.pause()
                        result = "Task paused"
                    case "resume":
                        await handle.resume()
                        result = "Task resumed"
                    case "answer_clarification":
                        if call_id:
                            await handle.answer_clarification(call_id, param_value)
                            result = "Clarification answered"
                        else:
                            result = "No clarification call_id available"
                    case _:
                        result = f"Unknown operation: {operation}"
            except Exception as e:
                result = f"Error: {e}"

            return {"status": "ok", "operation": operation, "result": result}

        # Set the docstring for the tool
        steering_tool.__doc__ = f"{docstring}\n\nFor task: {query}"
        if param_name:
            steering_tool.__doc__ += f"\n\nArgs:\n    {param_name}: {docstring}"

        return steering_tool
