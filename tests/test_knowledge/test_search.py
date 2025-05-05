from tests.helpers import _handle_project
from knowledge.knowledge_manager import KnowledgeManager


@_handle_project
def test_search():
    knowledge_manager = KnowledgeManager()
    knowledge_manager.start()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"x": 0, "y": 1}, {"x": 2, "y": 3}],
    )
    data = knowledge_manager._search()
    assert data == {
        "MyTable": [
            {"x": 2, "y": 3},
            {"x": 0, "y": 1},
        ],
    }


@_handle_project
def test_search_specific_tables():
    knowledge_manager = KnowledgeManager()
    knowledge_manager.start()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"x": 0, "y": 1}, {"x": 2, "y": 3}],
    )
    knowledge_manager._create_table("MyOtherTable")
    knowledge_manager._add_data(
        table="MyOtherTable",
        data=[{"t": 0, "v": 3}, {"t": 1, "v": 2}],
    )
    data = knowledge_manager._search(tables=["MyTable"])
    assert data == {
        "MyTable": [
            {"x": 2, "y": 3},
            {"x": 0, "y": 1},
        ],
    }
    data = knowledge_manager._search(tables=["MyOtherTable"])
    assert data == {
        "MyOtherTable": [
            {"t": 1, "v": 2},
            {"t": 0, "v": 3},
        ],
    }


@_handle_project
def test_search_w_filter():
    knowledge_manager = KnowledgeManager()
    knowledge_manager.start()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"x": 0, "y": 1}, {"x": 1, "y": 2}, {"x": 2, "y": 3}, {"x": 3, "y": 4}],
    )
    data = knowledge_manager._search(filter="x > 1 and y < 4")
    assert data == {
        "MyTable": [
            {"x": 2, "y": 3},
        ],
    }
