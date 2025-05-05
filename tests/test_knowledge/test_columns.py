from tests.helpers import _handle_project
from knowledge.knowledge_manager import KnowledgeManager


@_handle_project
def test_create_empty_column():
    knowledge_manager = KnowledgeManager()
    knowledge_manager.start()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._create_empty_column(
        table="MyTable",
        column_name="MyCol",
        column_type="int",
    )
    tables = knowledge_manager._list_tables(include_columns=True)
    assert tables == {"MyTable": {"description": None, "columns": {"MyCol": "int"}}}


@_handle_project
def test_create_derived_column():
    knowledge_manager = KnowledgeManager()
    knowledge_manager.start()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"x": 1, "y": 2}, {"x": 3, "y": 4}],
    )
    knowledge_manager._create_derived_column(
        table="MyTable",
        column_name="distance",
        equation="({x}**2 + {y}**2)**0.5",
    )
    data = knowledge_manager._search()
    assert data == {
        "MyTable": [
            {"x": 3, "y": 4, "distance": (3**2 + 4**2) ** 0.5},
            {"x": 1, "y": 2, "distance": (1**2 + 2**2) ** 0.5},
        ],
    }


@_handle_project
def test_delete_column():
    knowledge_manager = KnowledgeManager()
    knowledge_manager.start()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"x": 1, "y": 2}, {"x": 3, "y": 4}],
    )
    knowledge_manager._delete_column(table="MyTable", column_name="x")
    data = knowledge_manager._search()
    assert data == {
        "MyTable": [
            {"y": 4},
            {"y": 2},
        ],
    }


@_handle_project
def test_delete_empty_column():
    knowledge_manager = KnowledgeManager()
    knowledge_manager.start()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._create_empty_column(
        table="MyTable",
        column_name="x",
        column_type="int",
    )
    tables = knowledge_manager._list_tables(include_columns=True)
    assert tables == {"MyTable": {"description": None, "columns": {"x": "int"}}}
    knowledge_manager._delete_column(table="MyTable", column_name="x")
    tables = knowledge_manager._list_tables(include_columns=True)
    assert tables == {"MyTable": {"description": None, "columns": {}}}
    data = knowledge_manager._search()
    assert data == {"MyTable": []}


@_handle_project
def test_rename_column():
    knowledge_manager = KnowledgeManager()
    knowledge_manager.start()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"x": 1, "y": 2}, {"x": 3, "y": 4}],
    )
    knowledge_manager._rename_column(table="MyTable", old_name="x", new_name="X")
    data = knowledge_manager._search()
    assert data == {
        "MyTable": [
            {"y": 4, "X": 3},
            {"y": 2, "X": 1},
        ],
    }
