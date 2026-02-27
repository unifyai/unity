"""
Tests for DataManager context resolution logic.

DataManager handles multiple context types:
- Relative paths (resolved to Data/*)
- Absolute owned paths (Data/*)
- Foreign paths (Files/*, Knowledge/*, etc.)
"""

from __future__ import annotations


from unity.data_manager.simulated import SimulatedDataManager
from unity.common.context_registry import ContextRegistry

# ────────────────────────────────────────────────────────────────────────────
# Context path resolution
# ────────────────────────────────────────────────────────────────────────────


def test_relative_path_resolved_to_data_namespace():
    """Relative paths should be prefixed with Data/."""
    dm = SimulatedDataManager()

    # Create with relative path
    path = dm.create_table("myproject/users")

    assert path == "Data/myproject/users"


def test_absolute_data_path_unchanged():
    """Paths starting with Data/ should remain unchanged."""
    dm = SimulatedDataManager()

    # Create with absolute Data/ path
    path = dm.create_table("Data/myproject/users")

    assert path == "Data/myproject/users"


def test_foreign_context_accepted():
    """Foreign contexts (Files/, Knowledge/) should be accepted for queries."""
    dm = SimulatedDataManager()

    # Directly populate a foreign context for testing
    dm._tables["Files/Local/123/Content"] = [
        {"text": "Hello world", "line": 1},
    ]

    # Should be able to filter foreign context
    rows = dm.filter("Files/Local/123/Content")
    assert len(rows) == 1
    assert rows[0]["text"] == "Hello world"


def test_foreign_context_for_search():
    """Foreign contexts should work with search."""
    dm = SimulatedDataManager()

    dm._tables["Files/Local/456/Tables/Sheet1"] = [
        {"id": 1, "description": "Machine learning algorithms"},
        {"id": 2, "description": "Data analysis techniques"},
    ]

    results = dm.search(
        "Files/Local/456/Tables/Sheet1",
        references={"description": "ML algorithms"},
        k=5,
    )

    assert isinstance(results, list)


def test_fully_qualified_foreign_path_not_double_prefixed():
    """Fully-qualified foreign paths (e.g., org/asst/Contacts) must not be
    prepended with DataManager's own base context.

    When KnowledgeManager passes ContactManager's fully-qualified Contacts
    context (like "org123/42/Contacts") to DataManager, _resolve_context must
    recognise it as already absolute and return it unchanged — not produce
    "org123/42/Data/org123/42/Contacts".
    """
    from unity.data_manager.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    dm._base_ctx = "org123/42/Data"

    foreign_contexts = [
        "org123/42/Contacts",
        "org123/42/Tasks",
        "org123/42/Knowledge",
        "org123/42/Secrets",
        "org123/42/Images",
        "org123/42/Transcripts",
        "org123/42/Exchanges",
        "org123/42/BlackList",
    ]
    for ctx in foreign_contexts:
        resolved = dm._resolve_context(ctx)
        assert resolved == ctx, (
            f"Expected '{ctx}' to be returned as-is, "
            f"got '{resolved}' (double-prefixed with base context)"
        )


def test_context_resolution_for_reduce():
    """Context resolution should work for reduce operations."""
    dm = SimulatedDataManager()

    # Create with relative path
    dm.create_table("analytics/metrics", fields={"value": "float"})
    dm.insert_rows(
        "analytics/metrics",
        [
            {"value": 10.0},
            {"value": 20.0},
            {"value": 30.0},
        ],
    )

    # Query with relative path (columns= not column=)
    total = dm.reduce("analytics/metrics", metric="sum", columns="value")
    assert total == 60.0

    # Query with absolute path
    total2 = dm.reduce("Data/analytics/metrics", metric="sum", columns="value")
    assert total2 == 60.0


# ────────────────────────────────────────────────────────────────────────────
# ContextRegistry integration
# ────────────────────────────────────────────────────────────────────────────


def test_data_context_in_known_base_contexts():
    """Data should be a known base context."""
    ContextRegistry.clear()

    # Import to trigger registration
    from unity.data_manager.data_manager import DataManager  # noqa

    # Check Data is registered
    base_contexts = ContextRegistry.get_known_base_contexts()
    assert "Data" in base_contexts


def test_context_registry_get_known_base_contexts():
    """get_known_base_contexts should return all registered bases."""
    ContextRegistry.clear()

    # Import managers to trigger registration
    from unity.data_manager.data_manager import DataManager  # noqa

    bases = ContextRegistry.get_known_base_contexts()

    # Should include Data and potentially others
    assert isinstance(bases, list)
    assert "Data" in bases
