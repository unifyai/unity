from datetime import datetime

from unity.conversation_manager.domains.contact_index import (
    Message,
    EmailMessage,
    Contact,
    ContactIndex,
)
from unity.conversation_manager.domains.notifications import NotificationBar
from unity.conversation_manager.task_actions import (
    derive_short_name,
    iter_available_actions_for_task,
    build_action_name,
    safe_call_id_suffix,
)


class Renderer:

    def render_state(
        self,
        contact_index: ContactIndex = None,
        notification_bar: NotificationBar = None,
        active_tasks: dict = None,
        last_snapshot: datetime = None,
    ):
        return (
            f"{self.render_notification_bar(notification_bar, last_snapshot=last_snapshot)}\n\n"
            f"{self.render_active_tasks(active_tasks)}\n\n"
            f"{self.render_active_conversations(contact_index.active_conversations, last_snapshot=last_snapshot)}"
        )

    # contact stuff
    def render_active_conversations(
        self,
        active_conversations: dict[str, Contact],
        max_messages=5,
        max_global_messages=50,
        last_snapshot=None,
    ):
        contacts = "\n\n".join(
            self.render_contact(
                c,
                max_messages=max_messages,
                max_global_messages=max_global_messages,
                last_snapshot=last_snapshot,
            )
            for c in active_conversations.values()
        )
        return "<active_conversations>\n" f"{contacts}\n" "</active_conversations>"

    def render_contact(
        self,
        contact: Contact,
        max_messages=5,
        max_global_messages=50,
        last_snapshot=None,
    ):
        bio = f"<bio>{contact.bio}</bio>"
        rolling_summary = (
            f"<rolling_summary>{contact.rolling_summary}</rolling_summary>"
        )
        response_policy = (
            f"<response_policy>{contact.response_policy}</response_policy>"
        )
        global_thread = (
            self.render_thread(
                "global",
                contact.global_thread,
                max_messages=max_global_messages,
                last_snapshot=last_snapshot,
            )
            if contact.global_thread
            else ""
        )
        per_medium_threads = "\n\n".join(
            self.render_thread(
                t_name,
                t,
                max_messages=max_messages,
                last_snapshot=last_snapshot,
            )
            for t_name, t in contact.threads.items()
            if t
        )
        threads_content = (
            f"{global_thread}\n\n{per_medium_threads}"
            if global_thread
            else per_medium_threads
        )
        return (
            f"""<contact contact_id="{contact.contact_id}" first_name="{contact.first_name}" surname="{contact.surname}" is_boss="{contact.contact_id == 1}" phone_number="{contact.phone_number or ""}" email_address="{contact.email_address or ""}" on_call="{contact.on_call}">\n"""
            f"{bio}\n"
            f"{rolling_summary}\n"
            f"{response_policy}\n"
            "<threads>\n"
            f"{threads_content}\n"
            "</threads>\n"
            "</contact>"
        )

    def render_thread(self, thread_name, thread, max_messages=5, last_snapshot=None):
        messages = "\n".join(
            self.render_message(m, last_snapshot) for m in list(thread)[-max_messages:]
        )
        return f"<{thread_name}>\n" f"{messages}\n" f"</{thread_name}>"

    def render_message(self, message: Message, last_snapshot: datetime = None):
        # Mark all recent messages as NEW (both incoming and outbound)
        is_new = last_snapshot < message.timestamp
        if isinstance(message, EmailMessage):
            return (
                f"""{'**NEW**' if is_new else ""} [{message.name} @ {message.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}]:\n"""
                f"Subject: {message.subject}\n"
                f"Email ID: {message.email_id}\n"
                f"Body:\n"
                f"{message.body}"
            )
        return f"""{'**NEW**' if is_new else ""} [{message.name} @ {message.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}]: {message.content}"""

    # notification stuff
    def render_notification_bar(
        self,
        notification_bar: NotificationBar,
        last_snapshot=None,
    ):
        pinned_notifs = [n for n in notification_bar.notifications if n.pinned]
        new_notifs = [
            n
            for n in notification_bar.notifications
            # the timestamp check is probably not needed
            if not n.pinned and n.timestamp > last_snapshot
        ]
        all_notifs = pinned_notifs + new_notifs
        rendered_notifs = "\n".join(
            (
                "[PINNED]"
                if n.pinned
                else ""
                + f"""[{n.type.title()} Notification @ {n.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}] {n.content}"""
            )
            for n in all_notifs
        )
        return "<notifications>\n" f"{rendered_notifs}\n" "</notifications>"

    def render_active_tasks(self, active_tasks: dict):
        """Render currently active tasks with their status and history."""
        out = "<active_tasks>\n"
        if not active_tasks:
            out += "No active tasks\n"
        else:
            for handle_id, handle_data in active_tasks.items():
                query = handle_data.get("query", "")
                short_name = derive_short_name(query)
                handle_actions = handle_data.get("handle_actions", [])

                # Get pending clarifications
                pending_clarifications = [
                    a
                    for a in handle_actions
                    if a.get("action_name") == "clarification_request"
                    and not a.get("response")
                ]

                out += f"<task id='{handle_id}' short_name='{short_name}'>\n"
                out += f"<description>{query}</description>\n"

                # Show available actions using centralized helper
                out += "<available_actions>\n"
                for action_name, description in iter_available_actions_for_task(
                    handle_id,
                    query,
                    pending_clarifications,
                ):
                    out += f"  - {action_name}: {description}\n"
                out += "</available_actions>\n"

                # Show task history
                if handle_actions:
                    out += "<history>\n"
                    for a in handle_actions:
                        action_type = a.get("action_name", "")
                        action_query = a.get("query", "")
                        out += f"<event type='{action_type}'>\n"
                        if action_query:
                            out += f"  <content>{action_query}</content>\n"
                        if a_res := a.get("response"):
                            out += f"  <response>{a_res}</response>\n"
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

                out += "</task>\n"
        out += "</active_tasks>"
        return out
