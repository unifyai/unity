from __future__ import annotations

from unity.common.llm_helpers import method_to_schema
from unity.file_manager.managers.file_manager import FileManager


def test_save_attachment_tool_description_mentions_outbound_routing():
    fm = FileManager.__new__(FileManager)

    schema = method_to_schema(fm.save_attachment, "save_attachment")
    description = " ".join(schema["function"]["description"].split())

    assert "local disk" in description
    assert "outbound channel" in description
    assert "upload the bytes to GCS" in description
    assert "working-memory drafts" in description
