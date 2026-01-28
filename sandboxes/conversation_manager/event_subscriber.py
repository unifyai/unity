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
from typing import Awaitable, Callable, Optional

from unity.conversation_manager.events import (
    CallGuidance,
    ActorHandleStarted,
    ActorNotification,
    ActorResult,
    ActorClarificationRequest,
    EmailSent,
    Event,
    OutboundPhoneUtterance,
    SMSSent,
)

LG = logging.getLogger("conversation_manager_sandbox")

DisplayCallback = Callable[[str], Awaitable[None]] | Callable[[str], None]


def _format_outbound_event(event: Event, *, sandbox_state: object) -> Optional[str]:
    if isinstance(event, SMSSent):
        try:
            if bool(getattr(sandbox_state, "in_call", False)):
                return f"[Phone → User] {event.content}"
        except Exception:
            pass
        return f"[SMS → User] {event.content}"
    if isinstance(event, EmailSent):
        return f"[Email → User] Subject: {event.subject}\n{event.body}"
    if isinstance(event, OutboundPhoneUtterance):
        return f"[Phone → User] {event.content}"
    if isinstance(event, CallGuidance):
        # Debug-only; keep lightweight.
        return f"[Call Guidance] {event.content}"
    if isinstance(event, ActorHandleStarted):
        return f"[Actor] started: {event.query}"
    if isinstance(event, ActorNotification):
        return f"[Actor] {event.response}"
    if isinstance(event, ActorResult):
        # Compact result; detailed result is already in notifications bar / logs.
        return f"[Actor] completed: {event.result}"
    if isinstance(event, ActorClarificationRequest):
        return f"[Actor] clarification requested: {event.query}"
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
    include_call_guidance: bool = False,
    voice_enabled: bool = False,
    stop_event: asyncio.Event | None = None,
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
                backoff = 0.5  # reset after successful subscription

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
                            actor_in_flight = bool(actor_in_flight_ids)
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
                                            "[Actor] still working… (tip: `/ask <q>` for status, `/stop` to abort)",
                                        )
                                        last_progress_hint_at = now
                        except Exception:
                            pass
                        # Best-effort timeout-based completion
                        try:
                            if getattr(sandbox_state, "brain_run_in_flight", False):
                                last = float(
                                    getattr(
                                        sandbox_state,
                                        "last_event_published_at",
                                        0.0,
                                    )
                                    or 0.0,
                                )
                                if last and (time.monotonic() - last) > idle_grace_s:
                                    sandbox_state.brain_run_in_flight = False
                        except Exception:
                            pass
                        continue

                    try:
                        event = Event.from_json(msg["data"])
                    except Exception:
                        continue

                    if (not include_call_guidance) and isinstance(event, CallGuidance):
                        continue

                    # Track Actor in-flight status for UX hints.
                    try:
                        now = time.monotonic()
                        if isinstance(event, ActorHandleStarted):
                            hid = int(getattr(event, "handle_id", -1))
                            if hid >= 0:
                                actor_in_flight_ids.add(hid)
                                actor_completed_ids.discard(hid)
                                actor_waiting_clarification_ids.discard(hid)
                            last_actor_event_at = now
                        elif isinstance(event, ActorNotification):
                            hid = int(getattr(event, "handle_id", -1))
                            # Ignore late notifications after completion.
                            if hid in actor_completed_ids:
                                continue
                            if hid >= 0:
                                actor_in_flight_ids.add(hid)
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
                        elif isinstance(event, ActorResult):
                            hid = int(getattr(event, "handle_id", -1))
                            if hid >= 0:
                                actor_in_flight_ids.discard(hid)
                                actor_waiting_clarification_ids.discard(hid)
                                actor_completed_ids.add(hid)
                            last_actor_event_at = now
                    except Exception:
                        pass

                    rendered = _format_outbound_event(
                        event,
                        sandbox_state=sandbox_state,
                    )
                    if rendered is None:
                        continue

                    # Mark brain run complete on user-facing outbound.
                    try:
                        sandbox_state.brain_run_in_flight = False
                    except Exception:
                        pass

                    # Optional TTS for phone-call assistant utterances.
                    if voice_enabled and (
                        isinstance(event, OutboundPhoneUtterance)
                        or (
                            bool(getattr(sandbox_state, "in_call", False))
                            and isinstance(event, SMSSent)
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
                        ack = "✅ Sent that email."
                        if to_email:
                            ack = f"✅ Sent that email to {to_email}."
                        await _maybe_call(display_callback, f"[Phone → User] {ack}")

                    await _maybe_call(display_callback, rendered)
        except Exception as exc:
            LG.warning("event subscriber failed; retrying: %s", exc)
            await asyncio.sleep(backoff)
            backoff = min(max_backoff, backoff * 2)
