"""
Renderer: Renders conversation state for the ConversationManager LLM.

Contact information is fetched from ContactManager (source of truth).
Conversation state (threads) is fetched from ContactIndex.
"""

from datetime import datetime

from unity.common._async_tool.utils import get_handle_paused_state
from unity.conversation_manager.domains.contact_index import (
    Message,
    EmailMessage,
    UnifyMessage,
    GuidanceMessage,
    ConversationState,
    ContactIndex,
)
from unity.conversation_manager.domains.notifications import NotificationBar
from unity.conversation_manager.task_actions import (
    derive_short_name,
    iter_steering_tools_for_action,
    build_action_name,
    safe_call_id_suffix,
)


class Renderer:

    def render_state(
        self,
        contact_index: ContactIndex,
        notification_bar: NotificationBar = None,
        in_flight_actions: dict = None,
        last_snapshot: datetime = None,
    ):
        return (
            f"{self.render_notification_bar(notification_bar, last_snapshot=last_snapshot)}\n\n"
            f"{self.render_in_flight_actions(in_flight_actions)}\n\n"
            f"{self.render_active_conversations(contact_index, last_snapshot=last_snapshot)}"
        )

    def render_active_conversations(
        self,
        contact_index: ContactIndex,
        max_messages: int = 5,
        max_global_messages: int = 50,
        last_snapshot: datetime = None,
    ):
        """Render all active conversations, fetching contact info from ContactManager."""
        contacts = []
        for contact_id, conv_state in contact_index.active_conversations.items():
            # Fetch contact info from ContactManager (source of truth)
            contact_info = contact_index.get_contact(contact_id) or {}
            rendered = self.render_contact(
                contact_info=contact_info,
                conv_state=conv_state,
                max_messages=max_messages,
                max_global_messages=max_global_messages,
                last_snapshot=last_snapshot,
            )
            contacts.append(rendered)

        contacts_str = "\n\n".join(contacts)
        return f"<active_conversations>\n{contacts_str}\n</active_conversations>"

    def render_contact(
        self,
        contact_info: dict,
        conv_state: ConversationState,
        max_messages: int = 5,
        max_global_messages: int = 50,
        last_snapshot: datetime = None,
    ):
        """
        Render a single contact's conversation.

        Args:
            contact_info: Contact data from ContactManager (name, email, response_policy, etc.)
            conv_state: Conversation state from ContactIndex (threads, on_call)
        """
        contact_id = conv_state.contact_id
        first_name = contact_info.get("first_name") or ""
        surname = contact_info.get("surname") or ""
        phone_number = contact_info.get("phone_number") or ""
        email_address = contact_info.get("email_address") or ""
        bio = contact_info.get("bio") or ""
        rolling_summary = contact_info.get("rolling_summary") or ""
        response_policy = contact_info.get("response_policy") or ""
        should_respond = contact_info.get("should_respond", True)
        is_boss = contact_id == 1

        # Render threads
        global_thread = (
            self.render_thread(
                "global",
                conv_state.global_thread,
                max_messages=max_global_messages,
                last_snapshot=last_snapshot,
            )
            if conv_state.global_thread
            else ""
        )
        per_medium_threads = "\n\n".join(
            self.render_thread(
                t_name,
                t,
                max_messages=max_messages,
                last_snapshot=last_snapshot,
            )
            for t_name, t in conv_state.threads.items()
            if t
        )
        threads_content = (
            f"{global_thread}\n\n{per_medium_threads}"
            if global_thread
            else per_medium_threads
        )

        return (
            f'<contact contact_id="{contact_id}" first_name="{first_name}" surname="{surname}" '
            f'is_boss="{is_boss}" phone_number="{phone_number}" email_address="{email_address}" '
            f'on_call="{conv_state.on_call}" should_respond="{should_respond}">\n'
            f"<bio>{bio}</bio>\n"
            f"<rolling_summary>{rolling_summary}</rolling_summary>\n"
            f"<response_policy>{response_policy}</response_policy>\n"
            f"<threads>\n{threads_content}\n</threads>\n"
            f"</contact>"
        )

    def render_thread(self, thread_name, thread, max_messages=5, last_snapshot=None):
        messages = "\n".join(
            self.render_message(m, last_snapshot) for m in list(thread)[-max_messages:]
        )
        return f"<{thread_name}>\n{messages}\n</{thread_name}>"

    def render_message(
        self,
        message: Message | EmailMessage | UnifyMessage | GuidanceMessage,
        last_snapshot: datetime = None,
    ):
        # Mark all recent messages as NEW (both incoming and outbound)
        is_new = last_snapshot < message.timestamp
        new_marker = "**NEW** " if is_new else ""
        timestamp_str = message.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")

        if isinstance(message, EmailMessage):
            attachments_line = ""
            if message.attachments:
                if message.name == "You":
                    attachment_details = [
                        f"{fname} (attached)" for fname in message.attachments
                    ]
                else:
                    attachment_details = [
                        f"{fname} (auto-downloaded to Downloads/{fname})"
                        for fname in message.attachments
                    ]
                attachments_line = f"Attachments: {', '.join(attachment_details)}\n"
            # Render recipients (for reply-all context)
            recipients_lines = ""
            if message.to:
                recipients_lines += f"To: {', '.join(message.to)}\n"
            if message.cc:
                recipients_lines += f"Cc: {', '.join(message.cc)}\n"
            if message.bcc:
                recipients_lines += f"Bcc: {', '.join(message.bcc)}\n"
            # Show contact's role in this email for clarity in contact-specific threads
            # This helps the LLM understand the contact's relationship to the email
            contact_role_line = ""
            if message.contact_role:
                role_descriptions = {
                    "sender": "This contact SENT this email",
                    "to": "This contact was a DIRECT RECIPIENT (To)",
                    "cc": "This contact was CC'd on this email",
                    "bcc": "This contact was BCC'd on this email",
                }
                contact_role_line = f"[Context: {role_descriptions.get(message.contact_role, message.contact_role)}]\n"
            return (
                f"{new_marker}[{message.name} @ {timestamp_str}]:\n"
                f"{contact_role_line}"
                f"Subject: {message.subject}\n"
                f"Email ID: {message.email_id}\n"
                f"{recipients_lines}"
                f"{attachments_line}"
                f"Body:\n"
                f"{message.body}"
            )

        if isinstance(message, UnifyMessage):
            attachments_line = ""
            if message.attachments:
                if message.name == "You":
                    attachment_details = [
                        f"{fname} (attached)" for fname in message.attachments
                    ]
                else:
                    attachment_details = [
                        f"{fname} (auto-downloaded to Downloads/{fname})"
                        for fname in message.attachments
                    ]
                attachments_line = f" [Attachments: {', '.join(attachment_details)}]"
            return f"{new_marker}[{message.name} @ {timestamp_str}]: {message.content}{attachments_line}"

        return f"{new_marker}[{message.name} @ {timestamp_str}]: {message.content}"

    def render_notification_bar(
        self,
        notification_bar: NotificationBar,
        last_snapshot=None,
    ):
        pinned_notifs = [n for n in notification_bar.notifications if n.pinned]
        new_notifs = [
            n
            for n in notification_bar.notifications
            if not n.pinned and n.timestamp > last_snapshot
        ]
        all_notifs = pinned_notifs + new_notifs
        rendered_notifs = "\n".join(
            (
                "[PINNED]"
                if n.pinned
                else ""
                + f'[{n.type.title()} Notification @ {n.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}] {n.content}'
            )
            for n in all_notifs
        )
        return f"<notifications>\n{rendered_notifs}\n</notifications>"

    def render_in_flight_actions(self, in_flight_actions: dict):
        """Render currently in-flight actions with their status and history.

        These are actions that are ALREADY EXECUTING - work is in progress.
        Use steering tools to interact with them, don't duplicate with `act`.
        """
        out = "<in_flight_actions>\n"
        if not in_flight_actions:
            out += "No actions currently executing.\n"
        else:
            for handle_id, handle_data in in_flight_actions.items():
                query = handle_data.get("query", "")
                short_name = derive_short_name(query)
                handle = handle_data.get("handle")
                handle_actions = handle_data.get("handle_actions", [])

                # Determine status based on pause state
                is_paused = get_handle_paused_state(handle)
                status = "paused" if is_paused else "executing"

                pending_clarifications = [
                    a
                    for a in handle_actions
                    if a.get("action_name") == "clarification_request"
                    and not a.get("response")
                ]

                out += f"<action id='{handle_id}' short_name='{short_name}' status='{status}'>\n"
                out += f"<original_request>{query}</original_request>\n"

                out += "<steering_tools>\n"
                for action_name, description in iter_steering_tools_for_action(
                    handle_id,
                    query,
                    pending_clarifications,
                    is_paused=is_paused,
                ):
                    out += f"  - {action_name}: {description}\n"
                out += "</steering_tools>\n"

                if handle_actions:
                    out += "<history>\n"
                    for a in handle_actions:
                        action_type = a.get("action_name", "")
                        action_query = a.get("query", "")
                        action_status = a.get("status", "")

                        # Include status attribute if present (for ask operations)
                        if action_status:
                            out += f"<event type='{action_type}' status='{action_status}'>\n"
                        else:
                            out += f"<event type='{action_type}'>\n"

                        if action_query:
                            out += f"  <content>{action_query}</content>\n"
                        if a_res := a.get("response"):
                            out += f"  <response>{a_res}</response>\n"

                        # Show pending note for in-flight ask operations
                        if action_status == "pending" and action_type.startswith(
                            "ask_",
                        ):
                            out += (
                                "  <note>Result pending - you will receive another "
                                "turn when the answer is ready.</note>\n"
                            )

                        if action_type == "clarification_request" and not a.get(
                            "response",
                        ):
                            call_id = a.get("call_id", "")
                            suffix = safe_call_id_suffix(call_id)
                            action = build_action_name(
                                "answer_clarification",
                                short_name,
                                handle_id,
                                suffix,
                            )
                            out += f"  <pending>Use {action} to respond</pending>\n"
                        out += "</event>\n"
                    out += "</history>\n"

                out += "</action>\n"
        out += "</in_flight_actions>"
        return out
