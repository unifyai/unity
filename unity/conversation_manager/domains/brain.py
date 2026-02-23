from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, create_model

from unity.common.prompt_helpers import PromptParts
from unity.conversation_manager.prompt_builders import build_system_prompt
from unity.conversation_manager.types import Mode, ScreenshotEntry

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager


def _build_response_models() -> dict[Mode, type[BaseModel]]:
    """
    Create response models for ConversationManager's main brain.

    All actions (comms, task steering, etc.) are now tool calls.
    The response model only captures the LLM's reasoning.

    Returns:
        dict: Response models for different modes (Mode.CALL, Mode.MEET, Mode.TEXT)
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

    # Voice mode uses the same response model as text mode.
    # Guidance is delivered via the standalone guide_voice_agent tool (called in
    # parallel with the action tool) rather than the response content, because
    # the model reliably populates tool arguments but intermittently skips the
    # content field when tool_choice is required.
    return {
        Mode.CALL: TextResponse,
        Mode.MEET: TextResponse,
        Mode.TEXT: TextResponse,
    }


# Cache the response models since they don't change
_RESPONSE_MODELS = _build_response_models()


def build_response_models() -> dict[Mode, type[BaseModel]]:
    """
    Public accessor for response models used by ConversationManager's brain.

    Returns cached models for different modes (Mode.CALL, Mode.MEET, Mode.TEXT).
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

    system_prompt: PromptParts
    state_prompt: str
    response_model: type["BaseModel"]
    # Buffered screenshots captured during screen sharing, aligned with user turns.
    screenshots: list[ScreenshotEntry] = field(default_factory=list)
    # Relative file paths for each screenshot (parallel to screenshots list).
    screenshot_paths: list[str] = field(default_factory=list)

    def state_message(self) -> dict:
        # Mark this as a state snapshot so the async tool loop can treat it as
        # transient state (e.g., keep only the latest snapshot when generating).
        if not self.screenshots:
            return {
                "role": "user",
                "content": self.state_prompt,
                "_cm_state_snapshot": True,
            }

        # Build multimodal content: text state + screenshot blocks aligned with
        # the user utterances that triggered them.
        sources = {s.source for s in self.screenshots}
        if len(sources) > 1:
            header = (
                "The following screenshots were captured from multiple visual "
                "sources (desktop, user screen, and/or webcam), each paired with "
                "what the user said at that moment. They are in chronological order."
            )
        elif "user" in sources:
            header = (
                "The following screenshots were captured from the user's screen "
                "during screen sharing, each paired with what the user said "
                "at that moment. They are in chronological order."
            )
        elif "webcam" in sources:
            header = (
                "The following frames were captured from the user's webcam, "
                "each paired with what the user said at that moment. "
                "They are in chronological order."
            )
        else:
            header = (
                "The following screenshots were captured from your desktop "
                "during screen sharing, each paired with what the user said "
                "at that moment. They are in chronological order."
            )

        content_parts: list[dict] = [
            {"type": "text", "text": self.state_prompt},
            {
                "type": "text",
                "text": (
                    f"\n\n<screen_share_snapshots>\n{header}\n"
                    "</screen_share_snapshots>"
                ),
            },
        ]
        source_labels = {
            "assistant": "Assistant's Screen",
            "user": "User's Screen",
            "webcam": "User's Webcam",
        }
        for i, entry in enumerate(self.screenshots, 1):
            label = source_labels.get(entry.source, "Screenshot")
            path_suffix = ""
            if i <= len(self.screenshot_paths):
                path_suffix = f" -- {self.screenshot_paths[i - 1]}"
            content_parts.append(
                {
                    "type": "text",
                    "text": (
                        f"\n[{label} - Screenshot {i}/{len(self.screenshots)}"
                        f"{path_suffix}] "
                        f'User said: "{entry.utterance}"'
                    ),
                },
            )
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{entry.b64}"},
                },
            )

        return {
            "role": "user",
            "content": content_parts,
            "_cm_state_snapshot": True,
        }


def build_brain_spec(
    cm: "ConversationManager",
    screenshots: list[ScreenshotEntry] | None = None,
    screenshot_paths: list[str] | None = None,
) -> BrainSpec:
    """
    Build the prompt + response model inputs for a single Main CM Brain run.

    The returned spec is *pure* (no side effects) and can be used by either the
    legacy single-shot generate path or the async tool loop path.

    Parameters
    ----------
    cm : ConversationManager
        The conversation manager instance.
    screenshots : list[ScreenshotEntry] | None
        Buffered screenshots from screen sharing (assistant and/or user), each
        paired with the user utterance that triggered capture and a timestamp.
    screenshot_paths : list[str] | None
        Relative file paths corresponding to each screenshot (parallel list).
    """
    from unity.settings import SETTINGS

    prompt = cm.prompt_renderer.render_state(
        cm.contact_index,
        cm.notifications_bar,
        cm.in_flight_actions,
        cm.completed_actions,
        cm.last_snapshot,
        assistant_screen_share_active=cm.assistant_screen_share_active,
        user_screen_share_active=cm.user_screen_share_active,
        user_remote_control_active=cm.user_remote_control_active,
    ).full_render

    # Get boss contact (contact_id=1) from ContactManager - the source of truth
    boss_contact = cm.contact_index.get_contact(1) or {}
    is_boss_on_call = cm.mode.is_voice and (
        (cm.get_active_contact() or {}).get("contact_id") == 1
    )
    system_prompt = build_system_prompt(
        bio=cm.assistant_about,
        contact_id=1,
        first_name=boss_contact.get("first_name") or "",
        surname=boss_contact.get("surname") or "",
        phone_number=boss_contact.get("phone_number"),
        email_address=boss_contact.get("email_address"),
        is_voice_call=cm.mode.is_voice,
        is_boss_on_call=is_boss_on_call,
        demo_mode=SETTINGS.DEMO_MODE,
    )

    response_model = _RESPONSE_MODELS[cm.mode]

    # Validate we can JSON-encode state prompt early (helps catch accidental objects)
    json.dumps({"state_prompt": prompt})

    return BrainSpec(
        system_prompt=system_prompt,
        state_prompt=prompt,
        response_model=response_model,
        screenshots=screenshots or [],
        screenshot_paths=screenshot_paths or [],
    )
