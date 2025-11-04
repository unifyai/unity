from textwrap import dedent
from datetime import datetime

from unity.conversation_manager_2.domains.contact_index import Message, EmailMessage, Contact, ContactIndex
from unity.conversation_manager_2.domains.notifications import NotificationBar

class Renderer:

    def render_state(self, contact_index: ContactIndex=None, notification_bar: NotificationBar=None, last_snapshot: datetime = None):
        return (
                f"{self.render_notification_bar(notification_bar, last_snapshot)}\n"
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
            """<contact contact_id="{contact.contact_id}" first_name="{contact.first_name}" surname="{contact.surname}" is_boss="{contact.is_boss}" phone_number="{contact.phone_number or ""}" email_address="{contact.email_address or ""}" on_call="{contact.on_call}">\n"""
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
            return dedent(
                f"""\
                {'**NEW**' if is_new else ""} [{message.name} @ {message.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}]:
                Subject: {message.subject}
                Body:
                {message.body}
                """
            )
        return f"""{'**NEW**' if is_new else ""} [{message.name} @ {message.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}]: {message.content}"""
    
    # notification stuff
    def render_notification_bar(self, notification_bar: NotificationBar, last_snapshot=None):
        ...
    
    # conductor stuff
    def render_conductor_handle(self, x):
        ...