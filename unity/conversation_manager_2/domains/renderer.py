from textwrap import dedent

from contact_index import Message, Contact, ContactIndex
from notifications import NotificationBar

class Renderer:

    def render_state(self, contact_index: ContactIndex=None, notification_bar: NotificationBar=None):
        ...
    
    # contact stuff
    def render_active_conversations(self, active_conversations: dict[str, Contact]):
        ...

    def render_contact(self, contact: Contact):
        ...
    
    def render_thread(self, thread):
        ...
    
    def render_message(self, message: Message):
        ...
    
    # notification stuff
    def render_notification_bar(self):
        ...
    
    # conductor stuff
    def render_conductor_handle(self):
        ...