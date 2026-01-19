from datetime import datetime
from typing import Optional
from dataclasses import dataclass


@dataclass
class Notification:
    type: str
    content: str
    timestamp: datetime
    pinned: bool = False
    interjection_id: Optional[str] = None


class NotificationBar:
    def __init__(self):
        self.notifications = []

    def push_notif(
        self,
        type,
        notif_content,
        timestamp: datetime,
        pinned=False,
        id=None,
    ):
        self.notifications.append(
            Notification(type, notif_content, timestamp, pinned, id),
        )

    def remove_notif(self, interjection_id: str):
        """Remove a notification by its interjection_id."""
        self.notifications = [
            n for n in self.notifications if n.interjection_id != interjection_id
        ]
