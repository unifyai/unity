from textwrap import dedent
from datetime import datetime

from unity.conversation_manager_2.domains.contact_index import Message, EmailMessage, Contact, ContactIndex
from unity.conversation_manager_2.domains.notifications import NotificationBar

class Renderer:

    def render_state(self, contact_index: ContactIndex=None, 
                     notification_bar: NotificationBar=None,
                     conductor_handles: dict=None, 
                     last_snapshot: datetime = None):
        return (
                f"{self.render_notification_bar(notification_bar, last_snapshot)}\n\n"
                f"{self.render_conductor_handles(conductor_handles)}\n\n"
                f"{self.render_active_conversations(contact_index.active_conversations, last_snapshot)}"
        )

    
    # contact stuff
    def render_active_conversations(self, active_conversations: dict[str, Contact], last_snapshot=None):
        contacts = "\n\n".join(self.render_contact(c, last_snapshot) for c in active_conversations.values())
        return (
            "<active_conversations>\n"
            f"{contacts}\n"
            "</active_conversations>"
        )

    def render_contact(self, contact: Contact, last_snapshot):
        bio = "<bio>...</bio>"
        rolling_summary = "<rolling_summary>...</rolling_summary>"
        response_policy = "<response_policy>...</response_policy>"
        threads = "\n\n".join(self.render_thread(t_name, t, last_snapshot) for t_name, t in contact.threads.items() if t)
        return (
            f"""<contact contact_id="{contact.contact_id}" first_name="{contact.first_name}" surname="{contact.surname}" is_boss="{contact.is_boss}" phone_number="{contact.phone_number or ""}" email_address="{contact.email_address or ""}" on_call="{contact.on_call}">\n"""
            f"{bio}\n"
            f"{rolling_summary}\n"
            f"{response_policy}\n"
            "<threads>\n"
            f"{threads}\n"
            "</threads>\n"
            "</contact>"
        )
    
    def render_thread(self, thread_name, thread, last_snapshot):
        messages = "\n".join(self.render_message(m, last_snapshot) for m in thread)
        return (
                f"<{thread_name}>\n"
                f"{messages}\n"
                f"</{thread_name}>"
        )
    
    def render_message(self, message: Message, last_snapshot: datetime = None):
        is_new = last_snapshot < message.timestamp
        if isinstance(message, EmailMessage):
            return (
                f"""{'**NEW**' if is_new else ""} [{message.name} @ {message.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}]:\n"""
                f"Subject: {message.subject}"
                f"Body:"
                f"{message.body}"
            )
        return f"""{'**NEW**' if is_new else ""} [{message.name} @ {message.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}]: {message.content}"""
    
    # notification stuff
    def render_notification_bar(self, notification_bar: NotificationBar, last_snapshot=None):
        pinned_notifs = [n for n in notification_bar.notifications if n.pinned]
        new_notifs = [
            n
            for n in notification_bar.notifications
            # the timestamp check is probably not needed
            if not n.pinned and n.timestamp > last_snapshot
        ]
        all_notifs = pinned_notifs + new_notifs
        rendered_notifs = "\n".join(
            "[PINNED]" if n.pinned else "" + f"""[{n.type.title()} Notification @ {n.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}] {n.content}"""
            for n in all_notifs
        )
        return (
            "<notifications>\n"
            f"{rendered_notifs}\n"
            "</notifications>"
        )

    
    # conductor stuff
    def render_conductor_handles(self, handles):
        if not handles:
            return ""
        out = "<active_conductor_handles>\n"
        for handle_id, handle_data in handles.items():
            out += f"<conductor_handle handle_id='{handle_id}'>\n"
            out += f"<query>{handle_data['query']}</query>\n"
            out += f"<handle_actions>\n"
            for a in handle_data['handle_actions']:
                out += f"<action action_name={a['action_name']}>\n"
                out += f"<query>{a['query']}</query>\n"
                if a_res := a.get('response'):
                    out += f"<response>{a_res}</response>\n"
                # the original code has a 'call_id'here, which i dont fully understand
                out += "</handle_actions>\n"
            out += '</conductor_handle>\n'
        out += "</active_conductor_handles>"
        return out