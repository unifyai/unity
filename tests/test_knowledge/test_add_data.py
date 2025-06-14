from tests.helpers import _handle_project
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
import pytest


@pytest.mark.unit
@_handle_project
def test_add_data():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(name="MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"item": "WidgetA", "units": 30}, {"item": "WidgetB", "units": 25}],
    )
    data = knowledge_manager._search()
    assert data == {
        "MyTable": [
            {"item": "WidgetB", "units": 25},
            {"item": "WidgetA", "units": 30},
        ],
    }


@pytest.mark.unit
@_handle_project
def test_add_more_data():
    knowledge_manager = KnowledgeManager()
    knowledge_manager._create_table(name="MyTable")
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"item": "WidgetA", "units": 30}],
    )
    knowledge_manager._add_data(
        table="MyTable",
        data=[{"item": "WidgetB", "units": 25}],
    )
    data = knowledge_manager._search()
    assert data == {
        "MyTable": [
            {"item": "WidgetB", "units": 25},
            {"item": "WidgetA", "units": 30},
        ],
    }
