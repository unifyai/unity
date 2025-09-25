from .prompt_utils import (
    ThreadMessage,
    EmailThreadMessage,
    ConversationContact,
    NotificationBar,
    add_spaces,
)
from .new_events import *


class ConversationManagerState:
    def __init__(self, phone_contacts_map: dict, email_contacts_map: dict):
        self.phone_contacts_map = phone_contacts_map
        self.email_contacts_map = email_contacts_map
        self.inverted_contacts_map = {v.id: v for v in self.phone_contacts_map.values()}

        self.active_conversations: dict[str, ConversationContact] = {}
        self.stale_conversations = {}
        self.notifications = NotificationBar()

    def push_notif(self, notif, timestamp=None):
        self.notifications.push_notif(notif, timestamp)

    def clear_notifications(self, timestamp=None):
        # print(self.notifications.notifs)
        self.notifications.clear(timestamp=timestamp)

    def push_event(self, event: Event):
        if hasattr(event, "contact"):
            contact = self.phone_contacts_map.get(
                event.contact,
            ) or self.email_contacts_map.get(event.contact)
            if not contact:
                # will deal with this later
                # but in general probably either create a new contact here or
                # add a new anon contact
                ...
            contact_added = False
            if not self.is_contact_in_active_conversations(contact=contact):
                on_phone = False
                if isinstance(event, PhoneCallStarted):
                    on_phone = True
                self.add_contact_to_active_conversations(
                    contact,
                    on_phone,
                    event.timestamp,
                )
                contact_added = True

            active_c = self.active_conversations[contact.id]

        if isinstance(event, PhoneCallRecieved):
            active_c.push_message(
                "phone",
                message=ThreadMessage(
                    contact.name,
                    "<Phone call Sent...>",
                    event.timestamp,
                ),
            )
            self.notifications.push_notif(
                "comms",
                f"Phone Call Recieved by '{contact.name}'",
                event.timestamp,
            )
        elif isinstance(event, PhoneCallStarted):
            active_c.push_message(
                "phone",
                message=ThreadMessage(
                    contact.name,
                    "<Phone call Started...>",
                    event.timestamp,
                ),
            )
            self.notifications.push_notif(
                "comms",
                f"Phone Call Started with '{contact.name}'",
                event.timestamp,
            )
        elif isinstance(event, PhoneUtterance):
            active_c.push_message(
                "phone",
                message=ThreadMessage(contact.name, event.content, event.timestamp),
            )
            self.notifications.push_notif(
                "comms",
                f"Phone utterance recieved from '{contact.name}'",
                event.timestamp,
            )
        elif isinstance(event, PhoneCallEnded):
            active_c.push_message(
                "phone",
                message=ThreadMessage(
                    contact.name,
                    "<Phone call Ended...>",
                    event.timestamp,
                ),
            )
            self.notifications.push_notif(
                "comms",
                f"Phone Call Ended with '{contact.name}'",
                event.timestamp,
            )

        elif isinstance(event, SMSRecieved):
            active_c.push_message(
                "sms",
                message=ThreadMessage(contact.name, event.content, event.timestamp),
            )
            self.notifications.push_notif(
                "comms",
                f"SMS recieved recieved from '{contact.name}'",
                event.timestamp,
            )
        elif isinstance(event, EmailRecieved):
            active_c.push_message(
                "email",
                message=EmailThreadMessage(contact.name, event.subject, event.body, event.timestamp),
            )
            self.notifications.push_notif(
                "comms", f"Email recieved recieved from '{contact.name}'", event.timestamp
            )
        # assistant events
        elif isinstance(event, PhoneCallSent):
            active_c.push_message(
                "phone",
                message=ThreadMessage("You", "<Phone Call Sent...>", event.timestamp)
            )
            self.notifications.push_notif(
                "comms",
                f"Phone Call Sent to '{contact.name}'",
                event.timestamp,
            )
        elif isinstance(event, SMSSent):
            active_c.push_message(
                "sms",
                message=ThreadMessage("You", event.content, event.timestamp),
            )
            self.notifications.push_notif(
                "comms",
                f"SMS sent to '{contact.name}'",
                event.timestamp,
            )
        
        elif isinstance(event, EmailSent):
            active_c.push_message(
                "email", message=EmailThreadMessage("You", event.subject, event.body, event.timestamp)
            )
            self.notifications.push_notif(
                "comms", f"Email sent to '{contact.name}'", event.timestamp
            )

        # elif isinstance(event, ConductorQuerySent):
        #     self.notifications.push_notif(
        #         "conductor",
        #         f"Query '{event.query}' with id={event.id} sent and recieved by conductor and is being processed...",
        #         event.timestamp,
        #     )

        # elif isinstance(event, ConductorResult):
        #     self.notifications.push_notif(
        #         "conductor",
        #         f"Query with id={event.id} result: {event.result}",
        #         event.timestamp,
        #     )
        if hasattr(event, "contact"):
            if contact_added:
                self.notifications.push_notif(
                    "comms",
                    f"Added contact '{contact.name}' to active conversations",
                    event.timestamp,
                )

    def add_contact_to_active_conversations(
        self,
        contact,
        on_phone=False,
        timestamp=None,
    ):
        if contact.id in self.active_conversations:
            return
        self.active_conversations[contact.id] = ConversationContact(
            contact.id,
            contact.name,
            contact.is_boss,
            on_phone=on_phone,
        )

    def is_contact_in_active_conversations(self, contact_id=None, contact=None):
        if contact is None and contact_id is None:
            raise Exception("contact or contact_id must be provided")
        if contact and contact_id:
            raise Exception("Either contact_id or contact can be provided, not both")
        if contact:
            return contact.id in self.active_conversations
        elif contact_id:
            return contact_id in self.active_conversations

    def is_contact_in_stale_conversations(self, contact_id=None, contact=None):
        return contact.id in self.stale_conversations

    def __str__(self):
        active_convs = "\n\n".join([str(c) for c in self.active_conversations.values()])
        notif = str(self.notifications)
        state = f"<notifications>\n{add_spaces(notif)}\n</notifications>\n<active_conversations>\n{add_spaces(active_convs)}\n</active_conversations>"

        return state
