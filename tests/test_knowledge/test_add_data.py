from tests.helpers import _handle_project
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
import pytest


@pytest.mark.unit
@_handle_project
def test_add_data():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
    )
    data = knowledge_manager._search()
    assert data == {
        "MyTable": [
            {"name": "Bob", "age": 25},
            {"name": "Alice", "age": 30},
        ],
    }


@pytest.mark.unit
@_handle_project
def test_add_more_data():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"name": "Alice", "age": 30}],
    )
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"name": "Bob", "age": 25}],
    )
    data = knowledge_manager._search()
    assert data == {
        "MyTable": [
            {"name": "Bob", "age": 25},
            {"name": "Alice", "age": 30},
        ],
    }
