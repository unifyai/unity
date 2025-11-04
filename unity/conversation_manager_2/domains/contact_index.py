from collections import deque
from datetime import datetime
from dataclasses import dataclass

from pydantic import Field

from unity.contact_manager.types.contact import Contact as ContactType

class Contact(ContactType):
    is_boss: bool = False
    on_call: bool = False
    threads: dict[str, deque] = Field(
        default_factory=lambda: {
            "sms": deque(maxlen=5),
            "email": deque(maxlen=5),
            "phone": deque(maxlen=5),
            "unify_call": deque(maxlen=5),
            "unify_message": deque(maxlen=5),
        },
    )

    @property
    def full_name(self):
        name = self.first_name + " " + self.surname if self.surname else ""
        return name.strip()

@dataclass
class Message:
    name: str
    content: str
    timestamp: datetime


@dataclass
class EmailMessage:
    name: str
    subject: str
    body: str
    timestamp: datetime


class ContactIndex:
    def __init__(self):
        self.active_conversations: dict[str, Contact] = {}
        self.contacts = None
    
    @property
    def boss_contact(self):
        # this will have empty threads
        return Contact(**self.contacts.get(1))
    
    # is this supposed to fail for any reason?
    def push_message(self, contact: dict, thread_name, message_content=None, subject=None, body=None, timestamp=None):
        if not timestamp: timestamp = datetime.now()
        contact_id = contact["contact_id"]
        if contact_id not in self.active_conversations:
            self.active_conversations[contact_id] = Contact(**contact)
        contact = self.active_conversations[contact_id]
        if thread_name == "email": 
            message = EmailMessage(contact.full_name, body, subject, timestamp)
        else: 
            message = Message(contact.full_name, message_content, timestamp)
        contact.threads[thread_name].append(message)
    
    # should check if the contact exists
    def get_contact(self, contact_id: str) -> Contact:
        return self.contacts.get(contact_id)
