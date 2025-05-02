from tests.helpers import _handle_project
from knowledge.knowledge_manager import KnowledgeManager


@_handle_project
def test_create_table():
    knowledge_manager = KnowledgeManager()
    knowledge_manager.start()
    knowledge_manager._create_table("MyTable")
    tables = knowledge_manager._list_tables()
    assert len(tables) == 1
    assert "MyTable" in tables


@_handle_project
def test_delete_table():
    knowledge_manager = KnowledgeManager()
    knowledge_manager.start()

    # create
    knowledge_manager._create_table("MyTable")
    tables = knowledge_manager._list_tables()
    assert len(tables) == 1
    assert "MyTable" in tables

    # delete
    knowledge_manager._delete_table("MyTable")
    tables = knowledge_manager._list_tables()
    assert len(tables) == 0


@_handle_project
def test_rename_table():
    knowledge_manager = KnowledgeManager()
    knowledge_manager.start()

    # create
    knowledge_manager._create_table("MyTable")
    tables = knowledge_manager._list_tables()
    assert len(tables) == 1
    assert "MyTable" in tables

    # rename
    knowledge_manager._rename_table("MyTable", "MyNewTable")
    tables = knowledge_manager._list_tables()
    assert len(tables) == 1
    assert tables[0] == "MyNewTable"
