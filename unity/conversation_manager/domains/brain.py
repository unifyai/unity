from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, create_model

from unity.conversation_manager.prompt_builders import build_system_prompt

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager


def _build_response_models() -> dict:
    """
    Create response models for ConversationManager's main brain.

    All actions (comms, task steering, etc.) are now tool calls.
    The response model only captures the LLM's reasoning.

    Returns:
        dict: Response models for different modes (call, unify_meet, text)
    """
    # Text mode: just thoughts
    TextResponse = create_model(
        "TextResponse",
        thoughts=(
            str,
            Field(..., description="Your concise reasoning before taking actions"),
        ),
        __base__=BaseModel,
    )

    # Voice mode: thoughts + guidance for the Voice Agent
    # Both TTS and Realtime modes use call_guidance - the Main CM Brain
    # provides guidance/data to the voice agent (fast brain) which handles
    # the actual conversation.
    VoiceResponse = create_model(
        "VoiceResponse",
        thoughts=(
            str,
            Field(..., description="Your concise reasoning before taking actions"),
        ),
        call_guidance=(
            str,
            Field(..., description="Guidance for the Voice Agent handling the call"),
        ),
        __base__=BaseModel,
    )

    return {
        "call": VoiceResponse,
        "unify_meet": VoiceResponse,
        "text": TextResponse,
    }


# Cache the response models since they don't change
_RESPONSE_MODELS = _build_response_models()


def build_response_models() -> dict:
    """
    Public accessor for response models used by ConversationManager's brain.

    Returns cached models for different modes (call, unify_meet, text).
    """
    return _RESPONSE_MODELS


@dataclass(frozen=True)
class BrainSpec:
    """
    Fully materialized inputs needed for a single Main CM Brain run.

    This is intentionally a small, explicit data structure so we can refactor
    how the brain is executed (plain generate vs async tool loop) without
    tangling prompt/model construction with execution and side effects.
    """

    system_prompt: str
    state_prompt: str
    response_model: type["BaseModel"]

    def state_message(self) -> dict:
        # Mark this as a state snapshot so the async tool loop can treat it as
        # transient state (e.g., keep only the latest snapshot when generating).
        return {
            "role": "user",
            "content": self.state_prompt,
            "_cm_state_snapshot": True,
        }


def build_brain_spec(cm: "ConversationManager") -> BrainSpec:
    """
    Build the prompt + response model inputs for a single Main CM Brain run.

    The returned spec is *pure* (no side effects) and can be used by either the
    legacy single-shot generate path or the async tool loop path.
    """
    prompt = cm.prompt_renderer.render_state(
        cm.contact_index,
        cm.notifications_bar,
        cm.active_tasks,
        cm.last_snapshot,
    )

    boss_contact = cm.contact_index.boss_contact
    system_prompt = build_system_prompt(
        bio=cm.assistant_about,
        contact_id=boss_contact.contact_id,
        first_name=boss_contact.first_name,
        surname=boss_contact.surname,
        phone_number=boss_contact.phone_number,
        email_address=boss_contact.email_address,
        is_voice_call=cm.call_manager.uses_realtime_api,
    )

    response_model = _RESPONSE_MODELS[cm.mode]

    # Validate we can JSON-encode state prompt early (helps catch accidental objects)
    json.dumps({"state_prompt": prompt})

    return BrainSpec(
        system_prompt=system_prompt,
        state_prompt=prompt,
        response_model=response_model,
    )
