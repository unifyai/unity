"""
Event Publisher for the ConversationManager sandbox.

This module converts sandbox commands into inbound CM events and publishes them
to the in-memory event broker using the same `app:comms:*` topics as production.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from unity.conversation_manager.events import (
    EmailReceived,
    InboundPhoneUtterance,
    PhoneCallEnded,
    PhoneCallStarted,
    SMSReceived,
)


def get_simulated_user_contact() -> dict:
    """Build a simulated boss/user contact dict for sandbox events."""
    return {
        "contact_id": 1,
        "first_name": os.getenv("USER_NAME", "User"),
        "surname": os.getenv("USER_SURNAME", ""),
        "phone_number": os.getenv("USER_PHONE_NUMBER", "+15550001234"),
        "email_address": os.getenv("USER_EMAIL", "user@example.com"),
    }


@dataclass
class EventPublisher:
    cm: object
    state: object

    async def publish_sms(self, message: str) -> None:
        contact = get_simulated_user_contact()
        # Make boss contact available for prompt building even before ContactManager sync.
        try:
            self.cm.contact_index.set_fallback_contacts([contact])
        except Exception:
            pass

        event = SMSReceived(contact=contact, content=message)
        self.state.brain_run_in_flight = True
        self.state.last_event_published_at = time.monotonic()
        await self.cm.event_broker.publish("app:comms:msg_message", event.to_json())

    async def publish_email(self, subject: str, body: str) -> None:
        contact = get_simulated_user_contact()
        try:
            self.cm.contact_index.set_fallback_contacts([contact])
        except Exception:
            pass

        event = EmailReceived(contact=contact, subject=subject, body=body)
        self.state.brain_run_in_flight = True
        self.state.last_event_published_at = time.monotonic()
        await self.cm.event_broker.publish("app:comms:email_message", event.to_json())

    async def publish_call_start(self) -> None:
        contact = get_simulated_user_contact()
        try:
            self.cm.contact_index.set_fallback_contacts([contact])
        except Exception:
            pass

        self.state.in_call = True
        event = PhoneCallStarted(contact=contact)
        self.state.brain_run_in_flight = True
        self.state.last_event_published_at = time.monotonic()
        await self.cm.event_broker.publish(
            "app:comms:phone_call_started",
            event.to_json(),
        )

    async def publish_phone_utterance(self, text: str) -> None:
        contact = get_simulated_user_contact()
        try:
            self.cm.contact_index.set_fallback_contacts([contact])
        except Exception:
            pass

        event = InboundPhoneUtterance(contact=contact, content=text)
        self.state.brain_run_in_flight = True
        self.state.last_event_published_at = time.monotonic()
        await self.cm.event_broker.publish("app:comms:phone_utterance", event.to_json())

    async def publish_call_end(self) -> None:
        contact = get_simulated_user_contact()
        try:
            self.cm.contact_index.set_fallback_contacts([contact])
        except Exception:
            pass

        self.state.in_call = False
        event = PhoneCallEnded(contact=contact)
        self.state.brain_run_in_flight = True
        self.state.last_event_published_at = time.monotonic()
        await self.cm.event_broker.publish(
            "app:comms:phone_call_ended",
            event.to_json(),
        )
