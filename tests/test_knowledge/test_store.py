import json
from tests.helpers import _handle_project
from knowledge.knowledge_manager import KnowledgeManager


@_handle_project
def test_store():
    knowledge_manager = KnowledgeManager()
    knowledge_manager.start()
    knowledge_manager.store("Please remember that Adrian was born in 1994")
    assert "1994" in json.dumps(knowledge_manager._list_tables())
