"""Tests for broader-context assembly."""

from types import SimpleNamespace

from unity.memory_manager import broader_context
from unity.session_details import SESSION_DETAILS, SpaceSummary


class _ContactManager:
    def filter_contacts(self, *, filter: str, limit: int) -> dict:
        contacts = [
            SimpleNamespace(
                contact_id=0,
                first_name="Ava",
                surname="Assistant",
                bio="Assistant bio.",
                job_title="Operations",
            ),
            SimpleNamespace(
                contact_id=1,
                first_name="Boss",
                surname="User",
                bio="Boss bio.",
            ),
        ]
        if filter == "contact_id == 1":
            contacts = [contacts[1]]
        return {"contacts": contacts[:limit]}


def test_broader_context_includes_accessible_spaces_and_reset(monkeypatch):
    """The broader-context cache includes shared-space guidance and can reset."""

    from unity.memory_manager.memory_manager import MemoryManager

    monkeypatch.setattr(MemoryManager, "get_rolling_activity", lambda: "Recent work.")
    SESSION_DETAILS.reset()
    SESSION_DETAILS.space_summaries = [
        SpaceSummary(
            space_id=3,
            name="Repairs",
            description="South-East repairs patch daily operations.",
        ),
    ]
    broader_context.reset()

    first = broader_context.get_broader_context(contact_manager=_ContactManager())
    assert "Accessible shared spaces" in first
    assert '- space:3 "Repairs" - South-East repairs patch daily operations.' in first
    assert first.index("A bit about Boss User") < first.index(
        "Accessible shared spaces",
    )
    assert first.index("Accessible shared spaces") < first.index("# Activity Logs")

    SESSION_DETAILS.space_summaries = []
    assert (
        broader_context.get_broader_context(contact_manager=_ContactManager()) == first
    )

    broader_context.reset()
    refreshed = broader_context.get_broader_context(contact_manager=_ContactManager())
    assert "No shared spaces are currently available." in refreshed
