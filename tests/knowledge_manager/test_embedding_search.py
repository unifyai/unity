from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from tests.helpers import _handle_project
import pytest


@pytest.mark.requires_real_unify
@_handle_project
def test_embedding_search_basic():
    # Initialize and start the KnowledgeManager thread
    manager = KnowledgeManager()

    # Define table name and schema
    table_name = "ContactPrefs"
    columns = {"content": "str", "channel": "str"}
    manager._create_table(name=table_name, columns=columns)

    # Semantically related entries without substring overlap
    entries = [
        {"content": "I email and phone sometimes.", "channel": "email"},
        {
            "content": "Text messaging is my go-to communication method.",
            "channel": "sms",
        },
        {"content": "I love taking the train to work.", "channel": "travel"},
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
        table=table_name,
        references={"content": query},
        k=1,
    )
    assert len(emb_results_k1) == 1
    assert emb_results_k1[0]["content"] == entries[1]["content"]

    # Embedding-based nearest search for k=2 should respect ordering and limit
    emb_results_k2 = manager._search(
        table=table_name,
        references={"content": query},
        k=2,
    )
    assert len(emb_results_k2) == 2
    # First result should match the top-1, second should be different
    assert emb_results_k2[0]["content"] == emb_results_k1[0]["content"]
    assert emb_results_k2[1]["content"] in [
        e["content"] for e in entries if e["content"] != emb_results_k1[0]["content"]
    ]

    # Multi-expression search (sum of cosine) across multiple columns
    multi_refs = {"content": "texting", "channel": "sms"}
    multi_results = manager._search(table=table_name, references=multi_refs, k=2)
    assert len(multi_results) >= 1
    assert multi_results[0]["channel"] == "sms"
