"""
Extra tests ensuring the optional **guidance** parameter is honoured
by every public method of *SimulatedMemoryManager*.

For each method we:

• monkey-patch the corresponding `prompt_builders.build_*` helper with a
  wrapper that records the *guidance* argument and the generated prompt;
• invoke the method with a distinctive guidance string; and
• assert that the helper was called with that exact string **and** that
  the prompt body contains the guidance text (meaning `_with_guidance`
  appended it as expected).
"""

from __future__ import annotations

import json
import functools
import pytest

from unity.memory_manager.simulated import SimulatedMemoryManager
from unity.memory_manager import prompt_builders as pb

# shared fixture from the existing test-suite
from tests.helpers import _handle_project

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def _build_transcript(line: str) -> str:
    """Return a 50-message JSON blob with *line* in slot 37."""
    msgs = [{"sender": "User", "content": f"Chatter {i}."} for i in range(50)]
    msgs[37]["content"] = line
    return json.dumps(msgs, indent=4)


def _wrap_builder(monkeypatch, name: str, bucket: dict):
    """
    Monkey-patch `prompt_builders.<name>` so we can inspect its inputs.
    The original builder is still called so behaviour remains unchanged.
    """

    original = getattr(pb, name)

    @functools.wraps(original)
    def spy(*args, **kwargs):  # tools, guidance, ...
        bucket["called"] = True
        bucket["guidance"] = kwargs.get("guidance")
        prompt = original(*args, **kwargs)
        bucket["prompt"] = prompt
        return prompt

    monkeypatch.setattr(pb, name, spy, raising=True)


# --------------------------------------------------------------------------- #
# parametrised test – covers all four methods                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize(
    "method_name,builder_name,useful_line",
    [
        (
            "update_contacts",
            "build_contact_update_prompt",
            "New contact: Alex Rivera, alex@example.com.",
        ),
        (
            "update_contact_bio",
            "build_bio_prompt",
            "FYI – Alex just earned a PMP certification.",
        ),
        (
            "update_contact_rolling_summary",
            "build_rolling_prompt",
            "Action items: organise kickoff call next Tuesday.",
        ),
        (
            "update_knowledge",
            "build_knowledge_prompt",
            "Company switched to PostgreSQL 15 in March 2025.",
        ),
        (
            "update_tasks",
            "build_task_prompt",
            "Please schedule a follow-up meeting next Monday.",
        ),
        (
            "update_contact_response_policy",
            "build_response_policy_prompt",
            "Please respond formally in future.",
        ),
    ],
)
async def test_guidance_is_propagated(
    monkeypatch,
    method_name: str,
    builder_name: str,
    useful_line: str,
):
    bucket: dict[str, object] = {}
    _wrap_builder(monkeypatch, builder_name, bucket)

    guidance = f"TEST-GUIDANCE for {method_name}"
    mm = SimulatedMemoryManager("Guidance propagation demo")

    # Build 50-line transcript & dispatch to the target method
    transcript = _build_transcript(useful_line)
    method = getattr(mm, method_name)

    # call signatures differ slightly; pack kwargs accordingly
    if method_name in {
        "update_contact_bio",
        "update_contact_rolling_summary",
        "update_contact_response_policy",
    }:
        await method(transcript, contact_id=1, guidance=guidance)
    else:
        await method(transcript, guidance=guidance)

    # ------------------------------------------------------------------ #
    # expectations                                                       #
    # ------------------------------------------------------------------ #
    assert bucket.get("called"), f"{builder_name} should be invoked"
    assert bucket.get("guidance") == guidance, "Guidance arg must be forwarded"
    assert guidance in bucket.get("prompt", ""), "Prompt must embed the guidance text"
