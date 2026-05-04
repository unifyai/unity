from __future__ import annotations

from tests.destination_routing_helpers import (
    EVAL_SPACE_SUMMARIES,
    routing_decision_prompt,
)
from unity.common.accessible_spaces_block import build_accessible_spaces_block


def test_routing_eval_prompt_uses_product_accessible_spaces_block() -> None:
    """Eval prompts must not carry a second copy of shared-space routing policy."""

    prompt = routing_decision_prompt("Save the Patch-1 SOP for the team.")
    product_block = build_accessible_spaces_block(EVAL_SPACE_SUMMARIES)

    assert prompt.startswith(
        f"{product_block}\n\nAvailable write tools from the live manager docstrings:",
    )
    assert "FileManager.ingest_files" in prompt
    assert "DataManager.insert_rows" in prompt
    assert "BlackListManager.create_blacklist_entry" in prompt
