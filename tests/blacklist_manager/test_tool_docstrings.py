from __future__ import annotations

from unity.blacklist_manager.base import BaseBlackListManager


def test_blacklist_write_tools_expose_destination_guidance():
    for method_name in (
        "create_blacklist_entry",
        "update_blacklist_entry",
        "delete_blacklist_entry",
        "clear",
    ):
        doc = (getattr(BaseBlackListManager, method_name).__doc__ or "").strip()

        assert "destination" in doc
        assert "Accessible" in doc
        assert "shared" in doc
        assert "spaces" in doc
        assert "space:<id>" in doc
        assert "personal" in doc
        assert "strictest-rule-wins" in doc.lower()
        assert "request_clarification" in doc
