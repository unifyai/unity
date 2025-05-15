from tests.helpers import _handle_project
from knowledge.knowledge_manager import KnowledgeManager
import pytest


@pytest.mark.unit
@_handle_project
def test_search_basic():
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


@pytest.mark.unit
@_handle_project
def test_search_filter():
    knowledge_manager = KnowledgeManager()
    knowledge_manager.start()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"x": 0, "y": 1}, {"x": 2, "y": 3}],
    )
    data = knowledge_manager._search(filter="x > 0")
    assert data == {
        "MyTable": [
            {"x": 2, "y": 3},
        ],
    }


@pytest.mark.unit
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
        data=[{"a": 9, "b": 10}],
    )
    # default
    data = knowledge_manager._search()
    assert data == {
        "MyTable": [
            {"x": 2, "y": 3},
            {"x": 0, "y": 1},
        ],
        "MyOtherTable": [
            {"a": 9, "b": 10},
        ],
    }
    # specific tables
    data = knowledge_manager._search(tables=["MyTable"])
    assert data == {
        "MyTable": [
            {"x": 2, "y": 3},
            {"x": 0, "y": 1},
        ],
    }


@pytest.mark.unit
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
