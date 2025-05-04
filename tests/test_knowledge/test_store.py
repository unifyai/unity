import json
from tests.helpers import _handle_project
from knowledge.knowledge_manager import KnowledgeManager


@_handle_project
def test_store_simple():
    knowledge_manager = KnowledgeManager()
    knowledge_manager.start()
    knowledge_manager.store("Please remember that Adrian was born in 1994")
    all_knowledge = knowledge_manager._search()
    assert "1994" in json.dumps(all_knowledge)
