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
def test_delete_table():
    knowledge_manager = KnowledgeManager()

    # create
    knowledge_manager._create_table(name="MyTable")
    tables = knowledge_manager._tables_overview()
    assert len(tables) == 1
    assert "MyTable" in tables

    # delete
    knowledge_manager._delete_table(table="MyTable")
    tables = knowledge_manager._tables_overview()
    assert len(tables) == 0


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
