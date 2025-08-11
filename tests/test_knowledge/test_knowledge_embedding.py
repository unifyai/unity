from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from tests.helpers import _handle_project
import pytest


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_knowledge_embedding():
    # Initialize and start the KnowledgeManager thread
    manager = KnowledgeManager()

    # Define table name and schema
    table_name = "ContactPrefs"
    columns = {"content": "str"}
    manager._create_table(name=table_name, columns=columns)

    # Semantically related entries without substring overlap
    entries = [
        {"content": "I email and phone sometimes."},
        {"content": "Text messaging is my go-to communication method."},
        {"content": "I love taking the train to work."},
    ]
    manager._add_rows(table=table_name, rows=entries)

    # Keyword-based search should find no hits for the term 'preferences'
    keyword_results = manager._filter(
        filter="'preferences' in content",
        tables=[table_name],
    )[table_name]
    assert isinstance(keyword_results, list)
    assert len(keyword_results) == 0

    # Embedding-based nearest search for k=1 should return the most relevant entry
    query = "favorite means of communication"
    emb_results_k1 = manager._search(
        tables=[table_name],
        source="content",
        text=query,
        k=1,
    )[table_name]
    assert len(emb_results_k1) == 1
    assert emb_results_k1[0]["content"] == entries[1]["content"]

    # Embedding-based nearest search for k=2 should respect ordering and limit
    emb_results_k2 = manager._search(
        tables=[table_name],
        source="content",
        text=query,
        k=2,
    )[table_name]
    assert len(emb_results_k2) == 2
    # First result should match the top-1, second should be different
    assert emb_results_k2[0]["content"] == emb_results_k1[0]["content"]
    assert emb_results_k2[1]["content"] in [
        e["content"] for e in entries if e["content"] != emb_results_k1[0]["content"]
    ]
