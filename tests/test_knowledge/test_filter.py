from tests.helpers import _handle_project
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
import pytest


@pytest.mark.unit
@_handle_project
def test_search_basic():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(name="MyTable")
    knowledge_manager._add_rows(
        table="MyTable",
        rows=[{"x": 0, "y": 1}, {"x": 2, "y": 3}],
    )
    data = knowledge_manager._filter()
    assert data == {
        "MyTable": [
            {"row_id": 1, "x": 2, "y": 3},
            {"row_id": 0, "x": 0, "y": 1},
        ],
    }


@pytest.mark.unit
@_handle_project
def test_search_filter():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(name="MyTable")
    knowledge_manager._add_rows(
        table="MyTable",
        rows=[{"x": 0, "y": 1}, {"x": 2, "y": 3}],
    )
    data = knowledge_manager._filter(filter="x > 0")
    assert data == {
        "MyTable": [
            {"row_id": 1, "x": 2, "y": 3},
        ],
    }


@pytest.mark.unit
@_handle_project
def test_search_specific_tables():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(name="MyTable")
    knowledge_manager._add_rows(
        table="MyTable",
        rows=[{"x": 0, "y": 1}, {"x": 2, "y": 3}],
    )
    knowledge_manager._create_table(name="MyOtherTable")
    knowledge_manager._add_rows(
        table="MyOtherTable",
        rows=[{"a": 9, "b": 10}],
    )
    # default
    data = knowledge_manager._filter()
    assert data == {
        "MyTable": [
            {"row_id": 1, "x": 2, "y": 3},
            {"row_id": 0, "x": 0, "y": 1},
        ],
        "MyOtherTable": [
            {"row_id": 0, "a": 9, "b": 10},
        ],
    }
    # specific tables
    data = knowledge_manager._filter(tables=["MyTable"])
    assert data == {
        "MyTable": [
            {"row_id": 1, "x": 2, "y": 3},
            {"row_id": 0, "x": 0, "y": 1},
        ],
    }


@pytest.mark.unit
@_handle_project
def test_search_w_filter():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(name="MyTable")
    knowledge_manager._add_rows(
        table="MyTable",
        rows=[{"x": 0, "y": 1}, {"x": 1, "y": 2}, {"x": 2, "y": 3}, {"x": 3, "y": 4}],
    )
    data = knowledge_manager._filter(filter="x > 1 and y < 4")
    assert data == {
        "MyTable": [
            {"row_id": 2, "x": 2, "y": 3},
        ],
    }
