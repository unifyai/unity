"""
LLM-driven scenario generation for the ConversationManager sandbox.

Produces a list of typed Event dicts by introspecting ``Event._registry``
and passing the full event catalogue to a ScenarioBuilder tool loop.
"""

from __future__ import annotations

import json
from typing import Any, List

from pydantic import BaseModel, Field

from sandboxes.conversation_manager.scenario_generator import build_event_catalogue
from sandboxes.scenario_builder import ScenarioBuilder
from unity.common.llm_client import DEFAULT_MODEL


class ScenarioEvent(BaseModel):
    """A single event in a generated scenario sequence."""

    event_type: str = Field(..., description="Event class name from the catalogue")
    fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Dataclass field values (excluding timestamp)",
    )


class ScenarioPayload(BaseModel):
    """Structured payload for the submit_scenario tool."""

    events: List[ScenarioEvent] = Field(
        ...,
        description="Ordered sequence of events forming the scenario",
    )


async def generate_scenario(
    description: str,
    *,
    endpoint: str = DEFAULT_MODEL,
    min_events: int = 10,
    max_events: int = 20,
) -> list[dict]:
    """Generate a typed event sequence from a free-form description.

    Returns a list of ``{"event_type": str, "fields": dict}`` dicts ready
    for ``ScenarioGenerator`` to instantiate and publish.
    """

    catalogue = build_event_catalogue()
    result_events: list[dict] = []

    def submit_scenario(
        payload: ScenarioPayload | dict | str | None = None,
        **kwargs: Any,
    ) -> str:
        """Submit the generated scenario as a sequence of typed events.

        Call this exactly once with a ``ScenarioPayload`` containing the
        ordered event list.
        """
        # Normalise input
        model_payload: ScenarioPayload
        if payload is None:
            if "payload" in kwargs:
                candidate = kwargs["payload"]
            elif "events" in kwargs:
                candidate = {"events": kwargs["events"]}
            else:
                raise ValueError("submit_scenario requires a payload")

            if isinstance(candidate, str):
                model_payload = ScenarioPayload.model_validate(
                    json.loads(candidate),
                )
            elif isinstance(candidate, dict):
                model_payload = ScenarioPayload.model_validate(candidate)
            elif isinstance(candidate, ScenarioPayload):
                model_payload = candidate
            else:
                raise ValueError("Unsupported payload type")
        else:
            if isinstance(payload, str):
                model_payload = ScenarioPayload.model_validate(
                    json.loads(payload),
                )
            elif isinstance(payload, dict):
                model_payload = ScenarioPayload.model_validate(payload)
            elif isinstance(payload, ScenarioPayload):
                model_payload = payload
            else:
                raise ValueError("Unsupported payload type")

        for evt in model_payload.events:
            result_events.append(
                {"event_type": evt.event_type, "fields": evt.fields},
            )

        return f"{len(result_events)} events submitted"

    prompt = (
        "You are a **Scenario Synthesis Assistant** for a ConversationManager.\n\n"
        "Your task is to generate a realistic sequence of events that would occur\n"
        "in the described scenario. Use the event types listed below.\n\n"
        "## Available Event Types\n\n"
        f"{catalogue}\n\n"
        "## Rules\n\n"
        "- Produce a realistic back-and-forth: inbound events (user messages,\n"
        "  calls received) should be followed by corresponding outbound events\n"
        "  (assistant replies, calls made).\n"
        "- Phone calls must start with PhoneCallStarted and end with PhoneCallEnded.\n"
        "  Utterances go between them.\n"
        "- The `contact` field should be a dict with at minimum `contact_id` and\n"
        "  `first_name`. Use contact_id=1 for the boss/user. Use incrementing ids\n"
        "  for other contacts.\n"
        "- Include a mix of event types where the scenario description allows.\n"
        f"- Aim for roughly {min_events}-{max_events} events total.\n"
        "- Call `submit_scenario` exactly once with the full event sequence.\n"
        "- After calling the tool, do not output anything else.\n\n"
        f"## Scenario Description\n\n{description}"
    )

    builder = ScenarioBuilder(
        description=prompt,
        tools={"submit_scenario": submit_scenario},
        endpoint=endpoint,
        stateful=False,
    )

    await builder.create()

    return result_events
