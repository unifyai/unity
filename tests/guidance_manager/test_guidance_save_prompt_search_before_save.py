from __future__ import annotations

from unity.common.llm_helpers import method_to_schema
from unity.guidance_manager.guidance_manager import GuidanceManager


def test_add_guidance_tool_description_mentions_search_before_save():
    gm = GuidanceManager.__new__(GuidanceManager)

    schema = method_to_schema(gm.add_guidance, "GuidanceManager_add_guidance")
    description = " ".join(schema["function"]["description"].split())

    assert "_near_duplicates" not in schema["function"]["parameters"]["properties"]
    assert "search existing Guidance" in description
    assert "adding a parallel rule" in description
    assert "contradicts an existing rule" in description
