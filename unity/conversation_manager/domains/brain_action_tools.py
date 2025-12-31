from __future__ import annotations

from typing import TYPE_CHECKING, Any

from unity.conversation_manager.domains import actions as cm_actions

if TYPE_CHECKING:
    from collections.abc import Callable

    from unity.conversation_manager.conversation_manager import ConversationManager


class ConversationManagerBrainActionTools:
    """
    Side-effecting tools for the Main CM Brain (rolled out incrementally).

    These tools execute a subset of ConversationManager "actions" directly as
    tool calls, so we can migrate one path at a time from JSON action emission
    to tool-driven execution.
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
        Send an SMS message.

        Use this tool to send an SMS rather than emitting a `send_sms` entry in the
        final JSON `actions` list. This is part of a gradual migration of comms
        actions into direct tool calls.

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
        Send an email.

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
        Start an outbound phone call.

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

    def as_tools(self) -> dict[str, "Callable[..., Any]"]:
        """Return the tools dict for start_async_tool_loop."""
        return {
            # Keep the name aligned with existing action nomenclature for a smooth transition.
            "send_sms": self.send_sms,
            "send_unify_message": self.send_unify_message,
            "send_email": self.send_email,
            "make_call": self.make_call,
        }
