from tests.helpers import _handle_project
from knowledge.knowledge_manager import KnowledgeManager


@_handle_project
def test_add_data():
    knowledge_manager = KnowledgeManager()
    knowledge_manager.start()
    knowledge_manager._create_table("MyTable")
    knowledge_manager._add_data("MyTable", [{"x": 0, "y": 1}, {"x": 2, "y": 3}])
    data = knowledge_manager._search()
    assert data == {
        "MyTable": [
            {"x": 2, "y": 3},
            {"x": 0, "y": 1},
        ],
    }
