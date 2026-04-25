from __future__ import annotations

from unity.common.llm_helpers import method_to_schema
from unity.knowledge_manager.knowledge_manager import KnowledgeManager


def test_add_rows_tool_description_mentions_search_before_save():
    km = KnowledgeManager.__new__(KnowledgeManager)

    schema = method_to_schema(km._add_rows, "_add_rows")
    description = " ".join(schema["function"]["description"].split())

    assert "_near_duplicates" not in schema["function"]["parameters"]["properties"]
    assert "shared Knowledge pool" in description
    assert "near-duplicates" in description
    assert "Create a new row only when nothing similar is already there" in description
