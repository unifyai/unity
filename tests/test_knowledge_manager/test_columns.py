from tests.helpers import _handle_project
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
import pytest


@pytest.mark.unit
@_handle_project
def test_create_empty_column():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(name="MyTable")
    knowledge_manager._create_empty_column(
        table="MyTable",
        column_name="MyCol",
        column_type="int",
    )
    tables = knowledge_manager._tables_overview(include_column_info=True)
    assert tables == {
        "Contacts": {
            "description": tables["Contacts"]["description"],
            "columns": tables["Contacts"]["columns"],
        },
        "MyTable": {"description": None, "columns": {"row_id": "int", "MyCol": "int"}},
    }


@pytest.mark.unit
@_handle_project
def test_create_derived_column():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(name="MyTable")
    knowledge_manager._add_rows(
        table="MyTable",
        rows=[{"x": 1, "y": 2}, {"x": 3, "y": 4}],
    )
    knowledge_manager._create_derived_column(
        table="MyTable",
        column_name="distance",
        equation="({x}**2 + {y}**2)**0.5",
    )
    data = knowledge_manager._filter()
    assert data == {
        "Contacts": [],
        "MyTable": [
            {"row_id": 1, "x": 3, "y": 4, "distance": (3**2 + 4**2) ** 0.5},
            {"row_id": 0, "x": 1, "y": 2, "distance": (1**2 + 2**2) ** 0.5},
        ],
    }


@pytest.mark.unit
@_handle_project
def test_delete_column():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(name="MyTable")
    knowledge_manager._add_rows(
        table="MyTable",
        rows=[{"x": 1, "y": 2}, {"x": 3, "y": 4}],
    )
    knowledge_manager._delete_column(table="MyTable", column_name="x")
    data = knowledge_manager._filter()
    assert data == {
        "Contacts": [],
        "MyTable": [
            {"row_id": 1, "y": 4},
            {"row_id": 0, "y": 2},
        ],
    }


@pytest.mark.unit
@_handle_project
def test_delete_empty_column():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(name="MyTable")
    knowledge_manager._create_empty_column(
        table="MyTable",
        column_name="x",
        column_type="int",
    )
    tables = knowledge_manager._tables_overview(include_column_info=True)
    assert tables == {
        "Contacts": {
            "description": tables["Contacts"]["description"],
            "columns": tables["Contacts"]["columns"],
        },
        "MyTable": {"description": None, "columns": {"row_id": "int", "x": "int"}},
    }
    knowledge_manager._delete_column(table="MyTable", column_name="x")
    tables = knowledge_manager._tables_overview(include_column_info=True)
    assert tables == {
        "Contacts": {
            "description": tables["Contacts"]["description"],
            "columns": tables["Contacts"]["columns"],
        },
        "MyTable": {"description": None, "columns": {"row_id": "int"}},
    }
    data = knowledge_manager._filter()
    assert data == {"Contacts": [], "MyTable": []}


@pytest.mark.unit
@_handle_project
def test_rename_column():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(name="MyTable")
    knowledge_manager._add_rows(
        table="MyTable",
        rows=[{"x": 1, "y": 2}, {"x": 3, "y": 4}],
    )
    knowledge_manager._rename_column(table="MyTable", old_name="x", new_name="X")
    data = knowledge_manager._filter()

    # Assert the expected keys are present (order not guaranteed)
    assert set(data.keys()) == {"Contacts", "MyTable"}
    assert list(data["MyTable"][0].keys()) == ["row_id", "X", "y"]
    assert list(data["MyTable"][1].keys()) == ["row_id", "X", "y"]

    # Assert the values
    assert data == {
        "Contacts": [],
        "MyTable": [
            {"row_id": 1, "X": 3, "y": 4},
            {"row_id": 0, "X": 1, "y": 2},
        ],
    }
