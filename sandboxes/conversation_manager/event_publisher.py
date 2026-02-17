"""
Event Publisher for the ConversationManager sandbox.

Converts sandbox commands into inbound CM events and publishes them to the
in-memory event broker using the same ``app:comms:*`` topics as production.

When ``--live-voice`` is active, the ``call`` / ``end_call`` commands spawn
the production voice agent subprocess over LiveKit instead of simulating
text-based events.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from unity.conversation_manager.events import (
    EmailReceived,
    Event,
    InboundPhoneUtterance,
    PhoneCallEnded,
    PhoneCallStarted,
    SMSReceived,
)

if TYPE_CHECKING:
    from sandboxes.conversation_manager.live_voice import LiveVoiceSession


def get_simulated_user_contact() -> dict:
    """Build a simulated boss/user contact dict for sandbox events."""
    return {
        "contact_id": 1,
        "first_name": os.getenv("USER_NAME", "User"),
        "surname": os.getenv("USER_SURNAME", ""),
        "phone_number": os.getenv("USER_NUMBER", "+15550001234"),
        "email_address": os.getenv("USER_EMAIL", "user@example.com"),
    }


@dataclass
class EventPublisher:
    cm: object
    state: object
    args: object | None = None

    async def publish_event(self, event: Event) -> None:
        """Publish any Event instance to the broker using its ``topic`` ClassVar."""
        topic = getattr(type(event), "topic", None)
        if not topic:
            raise ValueError(
                f"Event {type(event).__name__} has no topic ClassVar set",
            )
        contact = get_simulated_user_contact()
        try:
            self.cm.contact_index.set_fallback_contacts([contact])
        except Exception:
            pass
        self.state.brain_run_in_flight = True
        self.state.last_event_published_at = time.monotonic()
        await self.cm.event_broker.publish(topic, event.to_json())

    async def publish_sms(self, message: str) -> None:
        contact = get_simulated_user_contact()
        await self.publish_event(SMSReceived(contact=contact, content=message))

    async def publish_email(self, subject: str, body: str) -> None:
        contact = get_simulated_user_contact()
        await self.publish_event(
            EmailReceived(contact=contact, subject=subject, body=body),
        )

    async def publish_call_start(self) -> None:
        self.state.in_call = True
        contact = get_simulated_user_contact()
        await self.publish_event(PhoneCallStarted(contact=contact))

    async def publish_phone_utterance(self, text: str) -> None:
        contact = get_simulated_user_contact()
        await self.publish_event(
            InboundPhoneUtterance(contact=contact, content=text),
        )

    async def publish_call_end(self) -> None:
        self.state.in_call = False
        contact = get_simulated_user_contact()
        await self.publish_event(PhoneCallEnded(contact=contact))

    # ── Live voice ────────────────────────────────────────────────────────

    async def start_live_call(self) -> "list[str]":
        """
        Start a live voice call using the production voice agent over LiveKit.

        Returns a list of status lines to display to the user.
        Connection details are written to .live_voice_connect.json and the
        token is copied to the system clipboard for easy pasting.
        """
        from sandboxes.conversation_manager.live_voice import start_session

        contact = get_simulated_user_contact()
        boss = get_simulated_user_contact()
        try:
            self.cm.contact_index.set_fallback_contacts([contact])
        except Exception:
            pass

        session = await start_session(
            cm=self.cm,
            contact=contact,
            boss=boss,
        )
        self.state.live_voice_session = session
        self.state.in_call = True
        self.state.brain_run_in_flight = True
        self.state.last_event_published_at = time.monotonic()

        clipboard_ok = getattr(session, "clipboard_ok", False)
        clipboard_line = (
            "  Token copied to clipboard!"
            if clipboard_ok
            else "  (Could not copy token to clipboard.)"
        )
        browser_line = (
            "  Opened LiveKit playground in your browser."
            if getattr(session, "browser_opened", False)
            else "  (Could not open browser automatically.)"
        )

        waited = float(getattr(session, "ready_wait_seconds", 0.0) or 0.0)
        timeout = float(getattr(session, "ready_timeout_seconds", 0.0) or 0.0)
        source = str(getattr(session, "ready_source", "") or "").strip()
        if bool(getattr(session, "ready", False)):
            readiness_line = (
                f"✅ Voice agent ready ({waited:.1f}s; signal: {source}). "
                "You can speak immediately after connecting."
            )
        elif bool(getattr(session, "agent_joined_room", False)):
            readiness_line = (
                f"⏳ Agent joined room but CM is still waiting for "
                f"`UnifyMeetStarted` ({waited:.1f}s / {timeout:.1f}s). "
                "Connect now and wait for the greeting before speaking."
            )
        else:
            readiness_line = (
                f"⏳ Voice agent still booting ({waited:.1f}s / {timeout:.1f}s). "
                "Connect now; audio starts once initialization completes."
            )

        return [
            "",
            "🎙️  Live voice call started!",
            "",
            readiness_line,
            "",
            "Connect via the LiveKit Agents Playground:",
            "  1. Open  https://agents-playground.livekit.io",
            '  2. Click the "Manual" tab',
            f"  3. Paste URL:  {session.livekit_url}",
            f"  4. Paste Token (from clipboard or {session.connection_file})",
            '  5. Click "Connect"',
            "",
            browser_line,
            clipboard_line,
            f"  Connection details saved to: {session.connection_file}",
            "",
            f"  Room:      {session.room_name}",
            f"  Agent log: {session.log_file}",
            "",
            "Speak through your browser mic; type `end_call` here when done.",
            "",
        ]

    async def end_live_call(self) -> "list[str]":
        """Stop the live voice call and clean up LiveKit + subprocess resources."""
        from sandboxes.conversation_manager.live_voice import stop_session

        session: LiveVoiceSession | None = getattr(
            self.state,
            "live_voice_session",
            None,
        )
        if session is None:
            self.state.in_call = False
            return ["⚠️ No live voice session to end."]

        await stop_session(cm=self.cm, session=session)
        self.state.live_voice_session = None
        self.state.in_call = False
        self.state.brain_run_in_flight = True
        self.state.last_event_published_at = time.monotonic()

        return ["📞 Live voice call ended. Room and subprocess cleaned up."]
