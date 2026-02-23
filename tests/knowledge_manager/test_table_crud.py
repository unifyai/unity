from tests.helpers import _handle_project
from unity.knowledge_manager.knowledge_manager import KnowledgeManager


@_handle_project
def test_create_table():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(name="MyTable")
    tables = knowledge_manager._tables_overview()
    assert len(tables) == 2
    assert set(tables.keys()) == {"Contacts", "MyTable"}


@_handle_project
def test_create_table_w_cols():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(
        name="MyTable",
        columns={"ColA": "int", "ColB": "str"},
    )
    tables = knowledge_manager._tables_overview(include_column_info=True)
    assert len(tables) == 2
    assert tables == {
        "Contacts": {
            "description": tables["Contacts"]["description"],
            "columns": tables["Contacts"]["columns"],
        },
        "MyTable": {
            "description": None,
            "columns": {"row_id": "int", "ColA": "int", "ColB": "str"},
        },
    }


@_handle_project
def test_create_table_w_desc():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(name="MyTable", description="For storing my data.")
    tables = knowledge_manager._tables_overview(include_column_info=False)
    assert len(tables) == 2
    assert tables == {
        "Contacts": {"description": tables["Contacts"]["description"]},
        "MyTable": {"description": "For storing my data."},
    }


@_handle_project
def test_list_tables():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(name="MyFirstTable")
    tables = knowledge_manager._tables_overview()
    assert len(tables) == 2
    assert "MyFirstTable" in tables
    knowledge_manager._create_table(name="MySecondTable")
    tables = knowledge_manager._tables_overview(include_column_info=False)
    assert len(tables) == 3
    assert tables == {
        "Contacts": {"description": tables["Contacts"]["description"]},
        "MyFirstTable": {"description": None},
        "MySecondTable": {"description": None},
    }


@_handle_project
def test_delete_tables():
    knowledge_manager = KnowledgeManager()

    # create
    knowledge_manager._create_table(name="MyTable")
    tables = knowledge_manager._tables_overview()
    assert len(tables) == 2
    assert "MyTable" in tables

    # delete
    knowledge_manager._delete_tables(tables="MyTable")
    tables = knowledge_manager._tables_overview()
    assert len(tables) == 1
    assert set(tables.keys()) == {"Contacts"}


@_handle_project
def test_delete_multiple_tables():
    """Explicitly delete several tables in a single call."""
    km = KnowledgeManager()

    # ── setup ───────────────────────────────────────────────────────────
    km._create_table(name="TableA")
    km._create_table(name="TableB")
    km._create_table(name="TableC")
    tabs = km._tables_overview()
    assert set(tabs.keys()) == {"Contacts", "TableA", "TableB", "TableC"}

    # ── action ──────────────────────────────────────────────────────────
    res = km._delete_tables(tables=["TableA", "TableC"])

    # ── assertions ──────────────────────────────────────────────────────
    # Two explicit deletions acknowledged …
    assert len(res) == 2
    # … and only the untouched table remains.
    tabs = km._tables_overview()
    assert set(tabs.keys()) == {"Contacts", "TableB"}


@_handle_project
def test_delete_tables_with_startswith():
    """Bulk-delete tables sharing a prefix via the *startswith* parameter."""
    km = KnowledgeManager()

    # ── setup ───────────────────────────────────────────────────────────
    km._create_table(name="_Private1")
    km._create_table(name="_Private2")
    km._create_table(name="Public")
    tabs = km._tables_overview()
    assert set(tabs.keys()) == {"Contacts", "_Private1", "_Private2", "Public"}

    # ── action ──────────────────────────────────────────────────────────
    res = km._delete_tables(tables=[], startswith="_")  # delete all "_…" tables

    # ── assertions ──────────────────────────────────────────────────────
    assert len(res) == 2  # two prefixed tables deleted
    tabs = km._tables_overview()
    assert set(tabs.keys()) == {"Contacts", "Public"}


@_handle_project
def test_delete_tables_mixed_explicit_and_startswith():
    """
    Combination: delete one explicit table *and* all prefixed tables
    in the same invocation.
    """
    km = KnowledgeManager()

    # ── setup ───────────────────────────────────────────────────────────
    km._create_table(name="_Tmp1")
    km._create_table(name="KeepMe")
    km._create_table(name="DeleteMe")
    tabs = km._tables_overview()
    assert set(tabs.keys()) == {"Contacts", "_Tmp1", "KeepMe", "DeleteMe"}

    # ── action ──────────────────────────────────────────────────────────
    res = km._delete_tables(tables="DeleteMe", startswith="_")

    # ── assertions ──────────────────────────────────────────────────────
    assert len(res) == 2  # _Tmp1 and DeleteMe removed
    tabs = km._tables_overview()
    assert set(tabs.keys()) == {"Contacts", "KeepMe"}


@_handle_project
def test_rename_table():
    knowledge_manager = KnowledgeManager()

    # create
    knowledge_manager._create_table(name="MyTable")
    tables = knowledge_manager._tables_overview()
    assert len(tables) == 2
    assert "MyTable" in tables

    # rename
    knowledge_manager._rename_table(old_name="MyTable", new_name="MyNewTable")
    tables = knowledge_manager._tables_overview()
    assert len(tables) == 2
    assert "MyNewTable" in tables


@_handle_project
def test_clear():
    km = KnowledgeManager()

    # Seed a couple of knowledge tables
    km._create_table(name="Alpha")
    km._create_table(name="Beta")

    # Sanity: tables present before clear
    tabs_before = km._tables_overview()
    assert set(tabs_before.keys()) == {"Contacts", "Alpha", "Beta"}

    # Execute clear
    km.clear()

    # After clear: Contacts linkage should be present again, other tables gone
    tabs_after = km._tables_overview()
    assert set(tabs_after.keys()) == {"Contacts"}
    assert "Alpha" not in tabs_after
    assert "Beta" not in tabs_after
