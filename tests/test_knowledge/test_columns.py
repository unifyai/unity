from tests.helpers import _handle_project
from knowledge.knowledge_manager import KnowledgeManager


@_handle_project
def test_create_empty_column():
    knowledge_manager = KnowledgeManager()
    knowledge_manager.start()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._create_empty_column("MyTable", "MyCol", "int")
    tables = knowledge_manager._list_tables(include_columns=True)
    assert tables == {"MyTable": {"description": None, "columns": {"MyCol": "int"}}}


@_handle_project
def test_create_derived_column():
    knowledge_manager = KnowledgeManager()
    knowledge_manager.start()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data("MyTable", [{"x": 1, "y": 2}, {"x": 3, "y": 4}])
    knowledge_manager._create_derived_column(
        "MyTable",
        "distance",
        "({x}**2 + {y}**2)**0.5",
    )
    data = knowledge_manager._search()
    assert data == {
        "MyTable": [
            {"x": 1, "y": 2, "distance": (1**2 + 2**2) ** 0.5},
            {"x": 3, "y": 4, "distance": (3**2 + 4**2) ** 0.5},
        ],
    }


@_handle_project
def test_delete_column():
    knowledge_manager = KnowledgeManager()
    knowledge_manager.start()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data("MyTable", [{"x": 1, "y": 2}, {"x": 3, "y": 4}])
    knowledge_manager._delete_column("MyTable", "x")
    data = knowledge_manager._search()
    assert data == {
        "MyTable": [
            {"y": 2},
            {"y": 4},
        ],
    }


@_handle_project
def test_transform_column():
    pass
