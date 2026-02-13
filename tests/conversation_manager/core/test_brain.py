"""
tests/conversation_manager/core/test_brain.py
==============================================

Unit tests for the BrainSpec data structure and build_brain_spec helper
in ``domains/brain.py``.

Covers:
- Plain-text state messages (no screenshots)
- Multimodal state messages with screenshot content parts
- Screenshot-to-utterance alignment in the multimodal output
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from unity.common.prompt_helpers import PromptParts
from unity.conversation_manager.domains.brain import BrainSpec


# =============================================================================
# Helpers
# =============================================================================

def _make_brain_spec(
    state_prompt: str = "<state>test</state>",
    screenshots: list[tuple[str, str, datetime]] | None = None,
) -> BrainSpec:
    """Create a minimal BrainSpec for testing."""
    from pydantic import BaseModel, Field, create_model

    DummyResponse = create_model(
        "DummyResponse",
        thoughts=(str, Field(..., description="Reasoning")),
    )

    parts = PromptParts()
    parts.add("You are a helpful assistant.")

    return BrainSpec(
        system_prompt=parts,
        state_prompt=state_prompt,
        response_model=DummyResponse,
        screenshots=screenshots or [],
    )


FAKE_B64 = "iVBORw0KGgoAAAANSUhEUg=="  # tiny valid-looking base64 stub


# =============================================================================
# Tests
# =============================================================================


class TestBrainSpecStateMessage:
    """Tests for BrainSpec.state_message() plain-text vs multimodal output."""

    def test_plain_text_without_screenshots(self):
        """Without screenshots the message is a plain text dict."""
        spec = _make_brain_spec(state_prompt="<state>hello</state>")
        msg = spec.state_message()

        assert msg["role"] == "user"
        assert isinstance(msg["content"], str)
        assert msg["content"] == "<state>hello</state>"
        assert msg["_cm_state_snapshot"] is True

    def test_multimodal_with_screenshots(self):
        """With screenshots the message content becomes a list of parts."""
        ts = datetime(2026, 2, 13, 12, 0, 0, tzinfo=timezone.utc)
        screenshots = [
            (FAKE_B64, "Click that button please", ts),
        ]
        spec = _make_brain_spec(screenshots=screenshots)
        msg = spec.state_message()

        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert msg["_cm_state_snapshot"] is True

        # First part is the text state prompt
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == spec.state_prompt

    def test_screenshot_header_present(self):
        """The multimodal message includes a header explaining the screenshots."""
        ts = datetime(2026, 2, 13, 12, 0, 0, tzinfo=timezone.utc)
        screenshots = [(FAKE_B64, "Do this", ts)]
        msg = _make_brain_spec(screenshots=screenshots).state_message()

        text_parts = [
            p for p in msg["content"] if p.get("type") == "text"
        ]
        header_texts = [
            p["text"] for p in text_parts
            if "screen_share_snapshots" in p["text"]
        ]
        assert len(header_texts) == 1
        assert "chronological order" in header_texts[0]

    def test_screenshot_utterance_alignment(self):
        """Each screenshot is preceded by a text block quoting the user utterance."""
        ts1 = datetime(2026, 2, 13, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 2, 13, 12, 0, 5, tzinfo=timezone.utc)
        screenshots = [
            (FAKE_B64, "First, click here", ts1),
            (FAKE_B64, "Then scroll down", ts2),
        ]
        msg = _make_brain_spec(screenshots=screenshots).state_message()
        content = msg["content"]

        # Find image_url parts
        image_parts = [
            (i, p) for i, p in enumerate(content) if p.get("type") == "image_url"
        ]
        assert len(image_parts) == 2

        # Each image should be preceded by a text part with the utterance
        for idx, img_part in image_parts:
            preceding = content[idx - 1]
            assert preceding["type"] == "text"

        # Verify the utterance text alignment
        assert 'User said: "First, click here"' in content[image_parts[0][0] - 1]["text"]
        assert 'User said: "Then scroll down"' in content[image_parts[1][0] - 1]["text"]

    def test_screenshot_numbering(self):
        """Screenshot labels include N/total numbering."""
        ts = datetime(2026, 2, 13, 12, 0, 0, tzinfo=timezone.utc)
        screenshots = [
            (FAKE_B64, "Step one", ts),
            (FAKE_B64, "Step two", ts),
            (FAKE_B64, "Step three", ts),
        ]
        msg = _make_brain_spec(screenshots=screenshots).state_message()

        text_parts = [
            p["text"] for p in msg["content"] if p.get("type") == "text"
        ]
        labels = [t for t in text_parts if t.strip().startswith("[Screenshot")]
        assert len(labels) == 3
        assert "[Screenshot 1/3]" in labels[0]
        assert "[Screenshot 2/3]" in labels[1]
        assert "[Screenshot 3/3]" in labels[2]

    def test_image_url_format(self):
        """Image parts use the data URI scheme with image/png MIME type."""
        ts = datetime(2026, 2, 13, 12, 0, 0, tzinfo=timezone.utc)
        screenshots = [(FAKE_B64, "Look at this", ts)]
        msg = _make_brain_spec(screenshots=screenshots).state_message()

        image_parts = [
            p for p in msg["content"] if p.get("type") == "image_url"
        ]
        assert len(image_parts) == 1
        url = image_parts[0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert url.endswith(FAKE_B64)

    def test_empty_screenshots_list_gives_plain_text(self):
        """An explicit empty screenshots list behaves like no screenshots."""
        spec = _make_brain_spec(screenshots=[])
        msg = spec.state_message()
        assert isinstance(msg["content"], str)
