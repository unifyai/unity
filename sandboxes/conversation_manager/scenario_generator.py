"""
Scenario seeding helpers for the ConversationManager sandbox.

Generates typed Event sequences via LLM introspection of ``Event._registry``
and publishes them into CM through ``EventPublisher.publish_event``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from unity.conversation_manager.events import Event

# ── Exclusion list ────────────────────────────────────────────────────────
# Events that should never appear in a generated scenario.  Maintained as a
# hard-coded set; the sandbox is not on the production path so minor drift
# is acceptable.

_NON_SIMULATABLE: set[str] = {
    # Config / lifecycle
    "StartupEvent",
    "AssistantUpdateEvent",
    "InitializationComplete",
    "_SessionConfigBase",
    "Ping",
    "SyncContacts",
    "BackupContactsEvent",
    # Internal state
    "StoreChatHistory",
    "GetChatHistory",
    "GetBusEventsResponse",
    "SummarizeContext",
    "LLMInput",
    "LLMUserMessage",
    "LLMAssistantMessage",
    "ContactInfoResponse",
    "LogMessageResponse",
    "UpdateContactRollingSummaryResponse",
    # Actor internals (dynamic handle_id topics, not simulatable)
    "ActorRequest",
    "ActorResponse",
    "ActorHandleRequest",
    "ActorHandleResponse",
    "ActorResult",
    "ActorClarificationRequest",
    "ActorClarificationResponse",
    "ActorNotification",
    "ActorSessionResponse",
    "ActorHandleStarted",
    # Steering (injected by handle, not scenario)
    "NotificationInjectedEvent",
    "NotificationUnpinnedEvent",
    # Error (published reactively, not scenario-driven)
    "Error",
    # Pre-hire (separate flow)
    "PreHireMessage",
    # Direct message (bypass event, not scenario)
    "DirectMessageEvent",
}


# ── Dynamic event catalogue ──────────────────────────────────────────────


def build_event_catalogue() -> str:
    """Build an LLM-friendly description of all simulatable Event types.

    Introspects ``Event._registry`` at runtime so the catalogue stays in sync
    with production event definitions without any manual list maintenance.
    """
    lines: list[str] = []
    for name, cls in sorted(Event._registry.items()):
        if name.startswith("_") or name in _NON_SIMULATABLE:
            continue
        topic = getattr(cls, "topic", None)
        if not topic:
            continue

        doc = (cls.__doc__ or "").strip().split("\n")[0]  # first line only
        fields = dataclasses.fields(cls)
        field_parts = []
        for f in fields:
            if f.name == "timestamp":
                continue
            type_str = f.type.__name__ if isinstance(f.type, type) else str(f.type)
            field_parts.append(f"{f.name}: {type_str}")
        field_desc = ", ".join(field_parts)

        lines.append(f"- {name}: {doc or name}")
        if field_desc:
            lines.append(f"  Fields: {field_desc}")
    return "\n".join(lines)


def _instantiate_event(event_type: str, fields: dict[str, Any]) -> Event:
    """Instantiate an Event from its registry name and a fields dict."""
    cls = Event._registry.get(event_type)
    if cls is None:
        raise ValueError(f"Unknown event type: {event_type}")
    # Filter to valid dataclass fields to tolerate extra keys from the LLM.
    valid = {f.name for f in dataclasses.fields(cls)} - {"timestamp"}
    filtered = {k: v for k, v in fields.items() if k in valid}
    return cls(**filtered)


# ── Scenario generator ───────────────────────────────────────────────────


def _summarize_events(events: list[dict]) -> str:
    counts: dict[str, int] = {}
    for e in events:
        name = str(e.get("event_type") or "unknown")
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "0 events"
    parts = [f"{v} {k}" for k, v in sorted(counts.items())]
    return ", ".join(parts)


@dataclass
class ScenarioGenerator:
    publisher: Any
    state: Any

    async def generate_and_publish(self, description: str) -> None:
        """Generate a typed event sequence and publish each into CM."""
        desc = (description or "").strip()
        if not desc:
            raise ValueError("Scenario description was empty.")

        print("[generate] Building synthetic scenario – this can take a moment…")

        from sandboxes.conversation_manager.scenario_llm import generate_scenario

        events = await generate_scenario(desc)

        print(f"✅ Scenario generated: {_summarize_events(events)}")

        published = 0
        for entry in events:
            event_type = str(entry.get("event_type") or "").strip()
            fields = entry.get("fields") or {}
            if not event_type:
                continue
            try:
                event = _instantiate_event(event_type, fields)
            except Exception as exc:
                print(f"⚠️ Skipping {event_type}: {exc}")
                continue

            # Track call state for the REPL.
            if event_type in {"PhoneCallStarted", "UnifyMeetStarted"}:
                self.state.in_call = True
            elif event_type in {"PhoneCallEnded", "UnifyMeetEnded"}:
                self.state.in_call = False

            await self.publisher.publish_event(event)
            published += 1

        print(f"✅ Published {published} event(s) into ConversationManager.")
