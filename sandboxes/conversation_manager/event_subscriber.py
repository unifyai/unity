"""
Event Subscriber for the ConversationManager sandbox.

Subscribes to `app:comms:*` and renders user-facing outbound events (SMS/email/phone)
for sandbox UIs. This is a best-effort display layer and should never crash the
sandbox on malformed events.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Optional

from unity.conversation_manager.events import (
    ActorClarificationRequest,
    ActorClarificationResponse,
    ActorHandleStarted,
    ActorNotification,
    ActorResult,
    ActorSessionResponse,
    AssistantScreenShareStarted,
    AssistantScreenShareStopped,
    FastBrainNotification,
    DirectMessageEvent,
    EmailReceived,
    EmailSent,
    Error,
    Event,
    InboundPhoneUtterance,
    InboundUnifyMeetUtterance,
    OutboundPhoneUtterance,
    OutboundUnifyMeetUtterance,
    PhoneCallAnswered,
    PhoneCallEnded,
    PhoneCallNotAnswered,
    PhoneCallReceived,
    PhoneCallStarted,
    SMSReceived,
    SMSSent,
    UnifyMeetEnded,
    UnifyMeetReceived,
    UnifyMeetStarted,
    UnifyMessageReceived,
    UnifyMessageSent,
    UnknownContactCreated,
    UserRemoteControlStarted,
    UserRemoteControlStopped,
    UserScreenShareStarted,
    UserScreenShareStopped,
    UserWebcamStarted,
    UserWebcamStopped,
    VoiceInterrupt,
)

from sandboxes.conversation_manager.event_tree_display import EventTreeDisplay
from sandboxes.conversation_manager.log_aggregator import LogAggregator
from sandboxes.conversation_manager.trace_display import TraceDisplay
from unity.events.event_bus import EVENT_BUS, Event as BusEvent
from unity.events.types.manager_method import ManagerMethodPayload

LG = logging.getLogger("conversation_manager_sandbox")

DisplayCallback = Callable[[str], Awaitable[None]] | Callable[[str], None]
EventCallback = (
    Callable[[str, dict[str, Any]], Awaitable[None]]
    | Callable[[str, dict[str, Any]], None]
)
ManagerMethodCallback = (
    Callable[[str, ManagerMethodPayload], Awaitable[None]]
    | Callable[[str, ManagerMethodPayload], None]
)


def _contact_label(contact: object, *, fallback: str) -> str:
    """Best-effort display label for a contact dict."""
    try:
        if isinstance(contact, dict):
            first = str(contact.get("first_name") or "").strip()
            last = str(contact.get("surname") or "").strip()
            full = " ".join([p for p in (first, last) if p]).strip()
            if full:
                return full
            for key in ("name", "phone_number", "email_address", "email"):
                val = str(contact.get(key) or "").strip()
                if val:
                    return val
    except Exception:
        pass
    return fallback


_MEET_INTERACTION_DISPLAY: dict[type, str] = {
    AssistantScreenShareStarted: "The user enabled assistant screen sharing \u2014 they can now see your desktop.",
    AssistantScreenShareStopped: "The user disabled assistant screen sharing \u2014 they can no longer see your desktop.",
    UserScreenShareStarted: "The user started sharing their screen with you.",
    UserScreenShareStopped: "The user stopped sharing their screen.",
    UserWebcamStarted: "The user enabled their webcam \u2014 you can now see them.",
    UserWebcamStopped: "The user disabled their webcam.",
    UserRemoteControlStarted: "The user took remote control of your desktop \u2014 they now have mouse and keyboard control.",
    UserRemoteControlStopped: "The user released remote control of your desktop \u2014 you may resume computer actions.",
}


def _format_outbound_event(event: Event, *, sandbox_state: object) -> Optional[str]:
    if isinstance(event, InboundPhoneUtterance):
        sender = _contact_label(getattr(event, "contact", {}), fallback="User")
        return f"[{sender} → Assistant • Phone] {event.content}"
    if isinstance(event, InboundUnifyMeetUtterance):
        sender = _contact_label(getattr(event, "contact", {}), fallback="User")
        return f"[{sender} → Assistant • Unify Meet] {event.content}"
    if isinstance(event, SMSReceived):
        sender = _contact_label(getattr(event, "contact", {}), fallback="User")
        return f"[{sender} → Assistant • SMS] {event.content}"
    if isinstance(event, UnifyMessageReceived):
        sender = _contact_label(getattr(event, "contact", {}), fallback="User")
        content = str(getattr(event, "content", "") or "").strip()
        attachments = getattr(event, "attachments", []) or []
        if attachments:
            return (
                f"[{sender} → Assistant • Unify] {content} [+{len(attachments)} files]"
            )
        return f"[{sender} → Assistant • Unify] {content}"
    if isinstance(event, EmailReceived):
        sender = _contact_label(getattr(event, "contact", {}), fallback="User")
        subject = str(getattr(event, "subject", "") or "").strip()
        body = str(getattr(event, "body", "") or "").strip()
        attachments = getattr(event, "attachments", []) or []
        header = f"[{sender} → Assistant • Email] Subject: {subject}"
        if attachments:
            header += f" [+{len(attachments)} files]"
        if body:
            return f"{header}\n{body}"
        return header
    if isinstance(event, PhoneCallReceived):
        sender = _contact_label(getattr(event, "contact", {}), fallback="User")
        return f"[{sender} → Assistant • Phone] Incoming call"
    if isinstance(event, UnifyMeetReceived):
        sender = _contact_label(getattr(event, "contact", {}), fallback="User")
        return f"[{sender} → Assistant • Unify Meet] Session connected"
    if isinstance(event, VoiceInterrupt):
        sender = _contact_label(getattr(event, "contact", {}), fallback="User")
        return f"[{sender} → Assistant • Phone] (interrupt)"

    if isinstance(event, SMSSent):
        to_name = _contact_label(getattr(event, "contact", {}), fallback="User")
        return f"[Assistant → {to_name} • SMS] {event.content}"
    if isinstance(event, UnifyMessageSent):
        to_name = _contact_label(getattr(event, "contact", {}), fallback="User")
        content = str(getattr(event, "content", "") or "").strip()
        attachments = getattr(event, "attachments", []) or []
        if attachments:
            return (
                f"[Assistant → {to_name} • Unify] {content} [+{len(attachments)} files]"
            )
        return f"[Assistant → {to_name} • Unify] {content}"
    if isinstance(event, EmailSent):
        to_addrs = getattr(event, "to", []) or []
        to_label = (
            ", ".join(to_addrs[:2]) + ("..." if len(to_addrs) > 2 else "")
            if to_addrs
            else "User"
        )
        subject = str(getattr(event, "subject", "") or "").strip()
        body = str(getattr(event, "body", "") or "").strip()
        header = f"[Assistant → {to_label} • Email] Subject: {subject}"
        if body:
            return f"{header}\n{body}"
        return header
    if isinstance(event, OutboundPhoneUtterance):
        return f"[Assistant → User • Phone] {event.content}"
    if isinstance(event, OutboundUnifyMeetUtterance):
        return f"[Assistant → User • Unify Meet] {event.content}"
    if isinstance(event, UnifyMeetStarted):
        return "🎙️ Live voice ready — you can start speaking."
    if isinstance(event, UnifyMeetEnded):
        return "📞 Live voice call ended."
    if isinstance(event, FastBrainNotification):
        # In a simulated call (no real voice agent), treat the notification as
        # the assistant's spoken reply. In live voice mode the real voice agent
        # speaks the guidance, so show it as observability info instead.
        try:
            in_call = bool(getattr(sandbox_state, "in_call", False))
            live_voice = bool(getattr(sandbox_state, "live_voice_active", False))
            if in_call and not live_voice:
                return f"[Assistant → User • Phone] {event.content}"
        except Exception:
            pass
        return f"[Call Guidance] {event.content}"
    if isinstance(event, ActorHandleStarted):
        return f"[Actor] started: {event.query}"
    if isinstance(event, ActorNotification):
        # Some ActorNotification events are "empty" (response=None). Suppress those
        # to avoid noisy "[Actor] None" lines in the conversation pane.
        try:
            r = getattr(event, "response", None)
            if r is None:
                return None
            r_txt = str(r).strip()
            if not r_txt or r_txt == "None":
                return None
            return f"[Actor] {r_txt}"
        except Exception:
            return None
    if isinstance(event, ActorResult):
        # Compact result; detailed result is already in notifications bar / logs.
        return f"[Actor] completed: {event.result}"
    if isinstance(event, ActorClarificationRequest):
        return f"[Actor] clarification requested: {event.query}"
    if isinstance(event, Error):
        return f"[Error] {event.message}"

    meet_desc = _MEET_INTERACTION_DISPLAY.get(type(event))
    if meet_desc is not None:
        reason = str(getattr(event, "reason", "") or "").strip()
        if reason:
            return f"[Meet] {meet_desc} (reason: {reason})"
        return f"[Meet] {meet_desc}"

    return None


async def _maybe_call(cb: DisplayCallback, text: str) -> None:
    try:
        ret = cb(text)
        if asyncio.iscoroutine(ret):
            await ret  # type: ignore[misc]
    except Exception:
        pass


async def subscribe_to_responses(
    *,
    cm: object,
    sandbox_state: object,
    display_callback: DisplayCallback,
    event_callback: EventCallback | None = None,
    manager_method_callback: ManagerMethodCallback | None = None,
    include_call_guidance: bool = False,
    voice_enabled: bool = False,
    stop_event: asyncio.Event | None = None,
    trace_display: TraceDisplay | None = None,
    event_tree_display: EventTreeDisplay | None = None,
    log_aggregator: LogAggregator | None = None,
    ui_refresh_callback: Callable[[], None] | None = None,
) -> None:
    """
    Subscribe to outbound CM events and print/display them.

    stop_event:
      If provided, exit when stop_event is set.
    """
    backoff = 0.5
    max_backoff = 8.0
    # Best-effort idle detection: if we haven't seen any outbound and the UI has
    # been "active" for too long since the last inbound publish, clear the flag
    # to avoid blocking scenario seeding forever in "no outbound" cases.
    idle_grace_s = 8.0
    # UX: If the Actor is running but doesn't emit frequent notifications,
    # print a lightweight "still working" line so the REPL doesn't feel stuck.
    # Track in-flight Actor handles deterministically (by handle_id). This avoids
    # false "still working" hints caused by late notifications after completion.
    actor_in_flight_ids: set[int] = set()
    actor_completed_ids: set[int] = set()
    actor_idle_ids: set[int] = set()
    actor_waiting_clarification_ids: set[int] = set()
    last_actor_event_at = 0.0
    last_progress_hint_at = 0.0
    progress_hint_every_s = 6.0

    while True:
        if stop_event is not None and stop_event.is_set():
            return

        try:
            async with cm.event_broker.pubsub() as pubsub:
                await pubsub.psubscribe("app:comms:*", "app:actor:*")
                await pubsub.subscribe("app:call:notification")
                backoff = 0.5  # reset after successful subscription

                # Register (once) for ManagerMethod events on the in-process EventBus.
                # This is the source of truth for manager call hierarchy emitted by the Actor
                # and state managers.
                if (
                    event_tree_display is not None
                    or log_aggregator is not None
                    or manager_method_callback is not None
                ):
                    await _ensure_manager_method_subscription(
                        event_tree_display=event_tree_display,
                        log_aggregator=log_aggregator,
                        manager_method_callback=manager_method_callback,
                    )
                # Best-effort UI refresh hook for the GUI.
                _MM_SUB_STATE["refresh"] = ui_refresh_callback

                while True:
                    if stop_event is not None and stop_event.is_set():
                        return

                    msg = await pubsub.get_message(
                        timeout=1.0,
                        ignore_subscribe_messages=True,
                    )
                    if not msg:
                        # Best-effort progress hint while Actor is running.
                        try:
                            now = time.monotonic()
                            actor_in_flight = bool(
                                actor_in_flight_ids
                                - actor_completed_ids
                                - actor_idle_ids,
                            )
                            actor_waiting_clarification = bool(
                                actor_waiting_clarification_ids,
                            )
                            if actor_in_flight and (not actor_waiting_clarification):
                                if (
                                    last_actor_event_at
                                    and (now - last_actor_event_at)
                                    >= progress_hint_every_s
                                ):
                                    if (
                                        now - last_progress_hint_at
                                    ) >= progress_hint_every_s:
                                        await _maybe_call(
                                            display_callback,
                                            "[Actor] still working... (tip: `/ask <q>` for status, `/stop` to abort)",
                                        )
                                        last_progress_hint_at = now
                        except Exception:
                            pass
                        continue

                    try:
                        event = Event.from_json(msg["data"])
                    except Exception:
                        continue

                    # Optional raw event callback (used by IPC UIs to build their own panes).
                    try:
                        if event_callback is not None:
                            ch_raw = msg.get("channel") or ""
                            if isinstance(ch_raw, (bytes, bytearray)):
                                ch = ch_raw.decode("utf-8", "ignore")
                            else:
                                ch = str(ch_raw)
                            d = event.to_dict()
                            ret = event_callback(ch, d)
                            if asyncio.iscoroutine(ret):
                                await ret  # type: ignore[misc]
                    except Exception:
                        pass

                    # Categorized logs (structured, high-signal) from broker channels.
                    try:
                        if log_aggregator is not None:
                            ch_raw = msg.get("channel") or ""
                            if isinstance(ch_raw, (bytes, bytearray)):
                                ch = ch_raw.decode("utf-8", "ignore")
                            else:
                                ch = str(ch_raw)
                            if ch.startswith("app:comms:"):
                                # Prefer a contentful message so repeated events don't look
                                # like duplicates in the GUI.
                                # NOTE: Store full message; truncation happens at render time.
                                m = event.__class__.__name__
                                try:
                                    # SMS events
                                    if isinstance(event, SMSReceived):
                                        content = str(
                                            getattr(event, "content", "") or "",
                                        ).strip()
                                        if content:
                                            m = f"SMSReceived: {content}"
                                    elif isinstance(event, SMSSent):
                                        content = str(
                                            getattr(event, "content", "") or "",
                                        ).strip()
                                        if content:
                                            m = f"SMSSent: {content}"
                                    # Unify console message events
                                    elif isinstance(event, UnifyMessageReceived):
                                        content = str(
                                            getattr(event, "content", "") or "",
                                        ).strip()
                                        attachments = (
                                            getattr(event, "attachments", []) or []
                                        )
                                        if attachments:
                                            m = f"UnifyMessageReceived: {content} [+{len(attachments)} files]"
                                        elif content:
                                            m = f"UnifyMessageReceived: {content}"
                                    elif isinstance(event, UnifyMessageSent):
                                        content = str(
                                            getattr(event, "content", "") or "",
                                        ).strip()
                                        attachments = (
                                            getattr(event, "attachments", []) or []
                                        )
                                        if attachments:
                                            m = f"UnifyMessageSent: {content} [+{len(attachments)} files]"
                                        elif content:
                                            m = f"UnifyMessageSent: {content}"
                                    # Email events
                                    elif isinstance(event, EmailSent):
                                        subj = str(
                                            getattr(event, "subject", "") or "",
                                        ).strip()
                                        if subj:
                                            m = f"EmailSent: {subj}"
                                    elif isinstance(event, EmailReceived):
                                        subj = str(
                                            getattr(event, "subject", "") or "",
                                        ).strip()
                                        attachments = (
                                            getattr(event, "attachments", []) or []
                                        )
                                        if attachments:
                                            m = f"EmailReceived: {subj} [+{len(attachments)} files]"
                                        elif subj:
                                            m = f"EmailReceived: {subj}"
                                    # Phone call state events
                                    elif isinstance(event, PhoneCallReceived):
                                        m = "PhoneCallReceived"
                                    elif isinstance(event, PhoneCallStarted):
                                        m = "PhoneCallStarted"
                                    elif isinstance(event, PhoneCallAnswered):
                                        m = "PhoneCallAnswered"
                                    elif isinstance(event, PhoneCallNotAnswered):
                                        reason = str(
                                            getattr(event, "reason", "") or "",
                                        ).strip()
                                        m = (
                                            f"PhoneCallNotAnswered: {reason}"
                                            if reason
                                            else "PhoneCallNotAnswered"
                                        )
                                    elif isinstance(event, PhoneCallEnded):
                                        m = "PhoneCallEnded"
                                    # Unify Meet state events
                                    elif isinstance(event, UnifyMeetReceived):
                                        m = "UnifyMeetReceived"
                                    elif isinstance(event, UnifyMeetStarted):
                                        m = "UnifyMeetStarted"
                                    elif isinstance(event, UnifyMeetEnded):
                                        m = "UnifyMeetEnded"
                                    # Phone utterance events
                                    elif isinstance(event, InboundPhoneUtterance):
                                        content = str(
                                            getattr(event, "content", "") or "",
                                        ).strip()
                                        if content:
                                            m = f"InboundPhoneUtterance: {content}"
                                    elif isinstance(event, OutboundPhoneUtterance):
                                        content = str(
                                            getattr(event, "content", "") or "",
                                        ).strip()
                                        if content:
                                            m = f"OutboundPhoneUtterance: {content}"
                                    # Unify Meet utterance events
                                    elif isinstance(event, InboundUnifyMeetUtterance):
                                        content = str(
                                            getattr(event, "content", "") or "",
                                        ).strip()
                                        if content:
                                            m = f"InboundUnifyMeetUtterance: {content}"
                                    elif isinstance(event, OutboundUnifyMeetUtterance):
                                        content = str(
                                            getattr(event, "content", "") or "",
                                        ).strip()
                                        if content:
                                            m = f"OutboundUnifyMeetUtterance: {content}"
                                    # Voice interrupt
                                    elif isinstance(event, VoiceInterrupt):
                                        m = "VoiceInterrupt"
                                    # Call guidance
                                    elif isinstance(event, FastBrainNotification):
                                        content = str(
                                            getattr(event, "content", "") or "",
                                        ).strip()
                                        if content:
                                            m = f"FastBrainNotification: {content}"
                                    # Other useful events
                                    elif isinstance(event, UnknownContactCreated):
                                        medium = str(
                                            getattr(event, "medium", "") or "",
                                        ).strip()
                                        preview = str(
                                            getattr(event, "message_preview", "") or "",
                                        ).strip()
                                        if preview:
                                            m = f"UnknownContactCreated ({medium}): {preview}"
                                        else:
                                            m = f"UnknownContactCreated ({medium})"
                                    elif isinstance(event, DirectMessageEvent):
                                        content = str(
                                            getattr(event, "content", "") or "",
                                        ).strip()
                                        source = str(
                                            getattr(event, "source", "") or "",
                                        ).strip()
                                        if content:
                                            m = f"DirectMessage [{source}]: {content}"
                                    elif isinstance(event, Error):
                                        message = str(
                                            getattr(event, "message", "") or "",
                                        ).strip()
                                        if message:
                                            m = f"Error: {message}"
                                    # Meet interaction events
                                    else:
                                        meet_name = _MEET_INTERACTION_DISPLAY.get(
                                            type(event),
                                        )
                                        if meet_name is not None:
                                            reason = str(
                                                getattr(event, "reason", "") or "",
                                            ).strip()
                                            m = event.__class__.__name__
                                            if reason:
                                                m = f"{m}: {reason}"
                                except Exception:
                                    pass
                                log_aggregator.handle_structured_event(
                                    category="cm",
                                    message=m,
                                )
                            elif ch.startswith("app:actor:"):
                                # Include content for actor events (otherwise the pane
                                # is not very informative).
                                # NOTE: Store full message; truncation happens at render time.
                                msg = event.__class__.__name__
                                actor_hid: int | None = None
                                try:
                                    actor_hid = int(getattr(event, "handle_id", -1))
                                    if actor_hid < 0:
                                        actor_hid = None
                                except Exception:
                                    actor_hid = None
                                try:
                                    if isinstance(event, ActorHandleStarted):
                                        q = str(
                                            getattr(event, "query", "") or "",
                                        ).strip()
                                        if q:
                                            msg = f"ActorHandleStarted: {q}"
                                    elif isinstance(event, ActorClarificationRequest):
                                        q = str(
                                            getattr(event, "query", "") or "",
                                        ).strip()
                                        if q:
                                            msg = f"ActorClarificationRequest: {q}"
                                    elif isinstance(event, ActorClarificationResponse):
                                        r = str(
                                            getattr(event, "response", "") or "",
                                        ).strip()
                                        if r:
                                            msg = f"ActorClarificationResponse: {r}"
                                    elif isinstance(event, ActorSessionResponse):
                                        r = str(
                                            getattr(event, "content", "") or "",
                                        ).strip()
                                        if r:
                                            msg = f"ActorSessionResponse: {r}"
                                    elif isinstance(event, ActorNotification):
                                        r = str(
                                            getattr(event, "response", "") or "",
                                        ).strip()
                                        if r:
                                            msg = f"ActorNotification: {r}"
                                    elif isinstance(event, ActorResult):
                                        r = str(
                                            getattr(event, "result", "") or "",
                                        ).strip()
                                        if r:
                                            msg = f"ActorResult: {r}"
                                except Exception:
                                    pass
                                log_aggregator.handle_structured_event(
                                    category="actor",
                                    message=msg,
                                    handle_id=actor_hid,
                                )
                    except Exception:
                        pass

                    # Best-effort: refresh logs panel in GUI.
                    try:
                        cb = _MM_SUB_STATE.get("refresh")
                        if callable(cb):
                            cb()
                    except Exception:
                        pass

                    if (not include_call_guidance) and isinstance(
                        event,
                        FastBrainNotification,
                    ):
                        continue

                    # Track Actor in-flight status for UX hints.
                    try:
                        now = time.monotonic()
                        if isinstance(event, ActorHandleStarted):
                            hid = int(getattr(event, "handle_id", -1))
                            if hid >= 0:
                                actor_in_flight_ids.add(hid)
                                actor_completed_ids.discard(hid)
                                actor_idle_ids.discard(hid)
                                actor_waiting_clarification_ids.discard(hid)
                            last_actor_event_at = now
                            try:
                                if trace_display is not None:
                                    trace_display.set_event_context(
                                        event_id=f"handle-{hid}",
                                        handle_id=hid if hid >= 0 else None,
                                    )
                            except Exception:
                                pass
                            try:
                                if event_tree_display is not None:
                                    event_tree_display.set_handle_context(
                                        handle_id=hid if hid >= 0 else None,
                                    )
                            except Exception:
                                pass
                            try:
                                if log_aggregator is not None:
                                    log_aggregator.set_handle_context(
                                        handle_id=hid if hid >= 0 else None,
                                    )
                            except Exception:
                                pass
                        elif isinstance(event, ActorNotification):
                            hid = int(getattr(event, "handle_id", -1))
                            # Ignore late notifications after completion.
                            if hid in actor_completed_ids:
                                continue
                            if hid >= 0:
                                actor_in_flight_ids.add(hid)
                                actor_idle_ids.discard(hid)
                                actor_waiting_clarification_ids.discard(hid)
                            last_actor_event_at = now
                        elif isinstance(event, ActorClarificationRequest):
                            # Ignore empty clarification artifacts.
                            q = getattr(event, "query", None)
                            if not q:
                                continue
                            hid = int(getattr(event, "handle_id", -1))
                            if hid >= 0:
                                actor_in_flight_ids.add(hid)
                                actor_waiting_clarification_ids.add(hid)
                            last_actor_event_at = now
                        elif isinstance(event, ActorSessionResponse):
                            hid = int(getattr(event, "handle_id", -1))
                            if hid >= 0:
                                actor_idle_ids.add(hid)
                                actor_waiting_clarification_ids.discard(hid)
                            last_actor_event_at = now
                        elif isinstance(event, ActorResult):
                            hid = int(getattr(event, "handle_id", -1))
                            if hid >= 0:
                                actor_in_flight_ids.discard(hid)
                                actor_idle_ids.discard(hid)
                                actor_waiting_clarification_ids.discard(hid)
                                actor_completed_ids.add(hid)
                            last_actor_event_at = now
                    except Exception:
                        pass

                    # Surface clarification-waiting state to the sandbox UI when supported.
                    try:
                        setattr(
                            sandbox_state,
                            "pending_clarification",
                            bool(actor_waiting_clarification_ids),
                        )
                    except Exception:
                        pass

                    rendered = _format_outbound_event(
                        event,
                        sandbox_state=sandbox_state,
                    )
                    if rendered is None:
                        continue

                    # Optional TTS for phone-call assistant utterances.
                    if voice_enabled and (
                        isinstance(event, OutboundPhoneUtterance)
                        or (
                            bool(getattr(sandbox_state, "in_call", False))
                            and isinstance(event, FastBrainNotification)
                        )
                    ):
                        try:
                            from sandboxes.utils import speak

                            content = str(getattr(event, "content", "") or "").strip()
                            if content:
                                speak(content)
                        except Exception:
                            pass

                    # UX: when a real outbound email is emitted while we're in a call,
                    # also acknowledge it in the call channel so the conversation feels natural.
                    if isinstance(event, EmailSent) and bool(
                        getattr(sandbox_state, "in_call", False),
                    ):
                        try:
                            to_email = (
                                (event.contact or {}).get("email_address")
                                or (event.contact or {}).get("email")
                                or ""
                            )
                        except Exception:
                            to_email = ""
                        ack = "✅ Email sent."
                        if to_email:
                            ack = f"✅ Email sent to {to_email}."
                        await _maybe_call(
                            display_callback,
                            f"[Assistant → User • Phone] {ack}",
                        )

                    # UX: when an outbound SMS is emitted while we're in a call,
                    # acknowledge it in the call channel. Do not "speak" the SMS body
                    # as if it were phone speech; keep that as a separate SMS line.
                    if isinstance(event, SMSSent) and bool(
                        getattr(sandbox_state, "in_call", False),
                    ):
                        try:
                            c = getattr(event, "contact", None) or {}
                            first = (c.get("first_name") or "").strip()
                            last = (c.get("surname") or "").strip()
                            to_name = " ".join([p for p in (first, last) if p]).strip()
                            if not to_name:
                                to_name = c.get("phone_number") or "recipient"
                        except Exception:
                            to_name = "recipient"
                        ack = "✅ SMS sent."
                        if to_name:
                            ack = f"✅ SMS sent to {to_name}."
                        await _maybe_call(
                            display_callback,
                            f"[Assistant → User • Phone] {ack}",
                        )
                        if voice_enabled:
                            try:
                                from sandboxes.utils import speak

                                speak(ack)
                            except Exception:
                                pass

                    await _maybe_call(display_callback, rendered)
        except Exception as exc:
            LG.warning("event subscriber failed; retrying: %s", exc)
            await asyncio.sleep(backoff)
            backoff = min(max_backoff, backoff * 2)


# ──────────────────────────────────────────────────────────────────────────────
# EventBus subscription (ManagerMethod)
# ──────────────────────────────────────────────────────────────────────────────

_MM_SUB_STATE: dict[str, object | None] = {
    "registered": False,
    "task": None,
    "tree": None,
    "logs": None,
    "manager_method_callback": None,
    "refresh": None,
}


async def _ensure_manager_method_subscription(
    *,
    event_tree_display: EventTreeDisplay | None,
    log_aggregator: LogAggregator | None,
    manager_method_callback: ManagerMethodCallback | None,
) -> None:
    """
    Register a single EventBus callback for ManagerMethod events.

    We keep a module-level sink so repeated sandbox restarts can update the
    target display instances without registering additional callbacks.
    """
    _MM_SUB_STATE["tree"] = event_tree_display
    _MM_SUB_STATE["logs"] = log_aggregator
    _MM_SUB_STATE["manager_method_callback"] = manager_method_callback

    # Already registered.
    if _MM_SUB_STATE.get("registered") is True:
        return

    async def _on_events(evts: list[BusEvent]) -> None:
        tree = _MM_SUB_STATE.get("tree")
        logs = _MM_SUB_STATE.get("logs")
        for e in evts:
            try:
                payload = ManagerMethodPayload.model_validate(e.payload)
            except Exception:
                continue

            try:
                if isinstance(tree, EventTreeDisplay):
                    tree.handle_manager_method(call_id=e.calling_id, payload=payload)
            except Exception:
                pass
            # Append manager-method logs (these back the GUI "Manager Logs" pane).
            try:
                if isinstance(logs, LogAggregator):
                    direction = (payload.phase or "").strip().lower()
                    # Include hierarchy_label when present (it’s the most informative),
                    # but keep it short for the log pane.
                    label = (payload.hierarchy_label or "").strip()
                    msg = f"{payload.manager}.{payload.method}"
                    if direction:
                        msg += f" [{direction}]"
                    if label:
                        msg += f" — {label}"
                    logs.handle_structured_event(category="manager", message=msg)
            except Exception:
                pass
            try:
                cb = _MM_SUB_STATE.get("manager_method_callback")
                if callable(cb):
                    ret = cb(e.calling_id, payload)
                    if asyncio.iscoroutine(ret):
                        await ret  # type: ignore[misc]
            except Exception:
                pass

        # Best-effort: request a GUI refresh after applying a batch.
        try:
            cb = _MM_SUB_STATE.get("refresh")
            if callable(cb):
                cb()
        except Exception:
            pass

    async def _register() -> None:
        await EVENT_BUS.register_callback(
            event_type="ManagerMethod",
            callback=_on_events,
            every_n=1,
        )

    # Ensure only one in-flight registration task at a time, but allow retry on failure.
    t = _MM_SUB_STATE.get("task")
    task = t if isinstance(t, asyncio.Task) else None
    if task is None or task.done():
        task = asyncio.create_task(_register())
        _MM_SUB_STATE["task"] = task

    try:
        await task
        _MM_SUB_STATE["registered"] = True
    except Exception:
        _MM_SUB_STATE["registered"] = False
        return
