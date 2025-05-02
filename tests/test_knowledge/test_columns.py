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
