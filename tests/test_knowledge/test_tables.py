from tests.helpers import _handle_project
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
import pytest


@pytest.mark.unit
@_handle_project
def test_create_table():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(name="MyTable")
    tables = knowledge_manager._tables_overview()
    assert len(tables) == 1
    assert "MyTable" in tables


@pytest.mark.unit
@_handle_project
def test_create_table_w_cols():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(
        name="MyTable",
        columns={"ColA": "int", "ColB": "str"},
    )
    tables = knowledge_manager._tables_overview(include_column_info=True)
    assert len(tables) == 1
    assert tables == {
        "MyTable": {
            "description": None,
            "columns": {"row_id": "int", "ColA": "int", "ColB": "str"},
        },
    }


@pytest.mark.unit
@_handle_project
def test_create_table_w_desc():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(name="MyTable", description="For storing my data.")
    tables = knowledge_manager._tables_overview(include_column_info=False)
    assert len(tables) == 1
    assert tables == {
        "MyTable": {"description": "For storing my data."},
    }


@pytest.mark.unit
@_handle_project
def test_list_tables():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(name="MyFirstTable")
    tables = knowledge_manager._tables_overview()
    assert len(tables) == 1
    assert "MyFirstTable" in tables
    knowledge_manager._create_table(name="MySecondTable")
    tables = knowledge_manager._tables_overview(include_column_info=False)
    assert len(tables) == 2
    assert tables == {
        "MyFirstTable": {"description": None},
        "MySecondTable": {"description": None},
    }


@pytest.mark.unit
@_handle_project
def test_delete_tables():
    knowledge_manager = KnowledgeManager()

    # create
    knowledge_manager._create_table(name="MyTable")
    tables = knowledge_manager._tables_overview()
    assert len(tables) == 1
    assert "MyTable" in tables

    # delete
    knowledge_manager._delete_tables(tables="MyTable")
    tables = knowledge_manager._tables_overview()
    assert len(tables) == 0


@pytest.mark.unit
@_handle_project
def test_delete_multiple_tables():
    """Explicitly delete several tables in a single call."""
    km = KnowledgeManager()

    # ── setup ───────────────────────────────────────────────────────────
    km._create_table(name="TableA")
    km._create_table(name="TableB")
    km._create_table(name="TableC")
    assert set(km._tables_overview().keys()) == {"TableA", "TableB", "TableC"}

    # ── action ──────────────────────────────────────────────────────────
    res = km._delete_tables(tables=["TableA", "TableC"])

    # ── assertions ──────────────────────────────────────────────────────
    # Two explicit deletions acknowledged …
    assert len(res) == 2
    # … and only the untouched table remains.
    assert set(km._tables_overview().keys()) == {"TableB"}


@pytest.mark.unit
@_handle_project
def test_delete_tables_with_startswith():
    """Bulk-delete tables sharing a prefix via the *startswith* parameter."""
    km = KnowledgeManager()

    # ── setup ───────────────────────────────────────────────────────────
    km._create_table(name="_Private1")
    km._create_table(name="_Private2")
    km._create_table(name="Public")
    assert set(km._tables_overview().keys()) == {"_Private1", "_Private2", "Public"}

    # ── action ──────────────────────────────────────────────────────────
    res = km._delete_tables(tables=[], startswith="_")  # delete all "_…" tables

    # ── assertions ──────────────────────────────────────────────────────
    assert len(res) == 2  # two prefixed tables deleted
    assert set(km._tables_overview().keys()) == {"Public"}


@pytest.mark.unit
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
    assert set(km._tables_overview().keys()) == {"_Tmp1", "KeepMe", "DeleteMe"}

    # ── action ──────────────────────────────────────────────────────────
    res = km._delete_tables(tables="DeleteMe", startswith="_")

    # ── assertions ──────────────────────────────────────────────────────
    assert len(res) == 2  # _Tmp1 and DeleteMe removed
    assert set(km._tables_overview().keys()) == {"KeepMe"}


@pytest.mark.unit
@_handle_project
def test_rename_table():
    knowledge_manager = KnowledgeManager()

    # create
    knowledge_manager._create_table(name="MyTable")
    tables = knowledge_manager._tables_overview()
    assert len(tables) == 1
    assert "MyTable" in tables

    # rename
    knowledge_manager._rename_table(old_name="MyTable", new_name="MyNewTable")
    tables = knowledge_manager._tables_overview()
    assert len(tables) == 1
    assert "MyNewTable" in tables
