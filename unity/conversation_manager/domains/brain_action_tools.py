from __future__ import annotations

from typing import TYPE_CHECKING, Any

from unity.conversation_manager.domains import actions as cm_actions
from unity.conversation_manager.task_actions import (
    STEERING_OPERATIONS,
    derive_short_name,
    build_action_name,
    safe_call_id_suffix,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from unity.conversation_manager.conversation_manager import ConversationManager


class ConversationManagerBrainActionTools:
    """
    Side-effecting tools for the Main CM Brain.

    All communication and task management actions are exposed as tool calls,
    following the async tool loop pattern.
    """

    def __init__(self, cm: "ConversationManager"):
        self._cm = cm

    async def send_sms(
        self,
        *,
        contact_id: int | None = None,
        contact_details: dict[str, Any] | None = None,
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
        await cm_actions.send_sms(
            self._cm,
            "send_sms",
            contact_id=contact_id,
            contact_details=contact_details,
            content=content,
        )
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
        await cm_actions.send_unify_message(
            self._cm,
            "send_unify_message",
            contact_id=contact_id,
            content=content,
        )
        return {"status": "ok"}

    async def send_email(
        self,
        *,
        contact_id: int | None = None,
        contact_details: dict[str, Any] | None = None,
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
        await cm_actions.send_email(
            self._cm,
            "send_email",
            contact_id=contact_id,
            contact_details=contact_details,
            subject=subject,
            body=body,
            email_id_to_reply_to=email_id_to_reply_to,
        )
        return {"status": "ok"}

    async def make_call(
        self,
        *,
        contact_id: int | None = None,
        contact_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Start an outbound phone call to a contact.

        Use this when the boss explicitly requests to communicate via phone call,
        or when voice communication is clearly the appropriate channel.

        Args:
            contact_id: Target contact_id when known (preferred).
            contact_details: Target identity details when contact_id is unknown.
        """
        await cm_actions.make_call(
            self._cm,
            "make_call",
            contact_id=contact_id,
            contact_details=contact_details,
        )
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
        await cm_actions.start_task_action(
            self._cm,
            "start_task",
            query=query,
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
