import pytest

from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from tests.helpers import _handle_project


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_knowledge_search_single_reference_basic():
    km = KnowledgeManager()

    table = "KB_Single"
    km._create_table(name=table)

    entries = [
        {
            "title": "Quick reference for Linux",
            "content": "Short tips and cheatsheets for terminal usage",
        },
        {
            "title": "Messaging protocols overview",
            "content": "Long-form article about message queues and brokers",
        },
        {
            "title": "Deep learning tutorial",
            "content": "Comprehensive guide to neural networks and training",
        },
        {
            "title": "Version control fundamentals",
            "content": "Introductory material on branching and merging",
        },
    ]
    km._add_rows(table=table, rows=entries)

    query = "short tips cheatsheet terminal"
    results = km._search(table=table, references={"content": query}, k=3)

    assert results[0]["title"] == "Quick reference for Linux"

    cols = km._get_columns(table=table)
    assert "_content_emb" in cols


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_knowledge_search_multi_columns_json_and_vec_created():
    km = KnowledgeManager()

    table = "KB_MultiCols"
    km._create_table(name=table)

    km._add_rows(
        table=table,
        rows=[
            {
                "title": "Compose LaTeX quickly",
                "content": "Short notes with snippets for equations and symbols",
            },
            {
                "title": "Logging problems",
                "content": "Comprehensive debugging guide with verbose traces",
            },
            {
                "title": "Text processing toolkit",
                "content": "Prefers regex for quick text processing in pipelines",
            },
        ],
    )

    query = "quick text snippets"
    refs = {"content": query, "title": "irrelevant"}
    results = km._search(table=table, references=refs, k=2)

    assert len(results) == 2
    assert results[0]["title"] in {"Text processing toolkit", "Compose LaTeX quickly"}

    cols = km._get_columns(table=table)
    assert "_content_emb" in cols
    assert "_title_emb" in cols


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_knowledge_search_all_columns_default_derivation():
    km = KnowledgeManager()

    table = "KB_Expr"
    km._create_table(name=table)

    km._add_rows(
        table=table,
        rows=[
            {
                "title": "Reading list summary",
                "content": "Prepare a summary of recent research papers",
                "category": "Research",
                "keywords": "summaries literature",
            },
            {
                "title": "Email notifications config",
                "content": "Best practices for email deliverability and setup",
                "category": "Operations",
                "keywords": "email smtp dkim",
            },
            {
                "title": "SMS templates",
                "content": "Create message templates for quick outreach",
                "category": "Communications",
                "keywords": "sms texting",
            },
        ],
    )

    expr = "str({title}) + ' ' + str({content}) + ' ' + str({category}) + ' ' + str({keywords})"
    query = "best practices for email"
    results = km._search(table=table, references={expr: query}, k=2)

    assert len(results) >= 1
    assert results[0]["title"] == "Email notifications config"

    cols = km._get_columns(table=table)
    assert any(k.startswith("_expr_") and k.endswith("_emb") for k in cols.keys())


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_knowledge_search_sum_of_cosine_ranking():
    km = KnowledgeManager()

    table = "KB_SumCos"
    km._create_table(name=table)

    km._add_rows(
        table=table,
        rows=[
            {
                "title": "Neural network training guide",
                "content": "Getting started guide for training deep networks",
                "tags": "deep learning, tutorial",
            },
            {
                "title": "Onboarding process",
                "content": "Haven't started yet",
                "tags": "tutorial",
            },
            {
                "title": "Tax preparation tips",
                "content": "Getting started with your taxes",
                "tags": "finance, guide",
            },
        ],
    )

    refs = {"tags": "deep learning", "content": "getting started guide"}
    results = km._search(table=table, references=refs, k=3)

    assert len(results) == 3
    titles = [r["title"] for r in results]
    assert titles[0] == "Neural network training guide"
    assert titles.index("Neural network training guide") < titles.index(
        "Onboarding process",
    )
    assert titles.index("Neural network training guide") < titles.index(
        "Tax preparation tips",
    )

    cols = km._get_columns(table=table)
    assert "_tags_emb" in cols
    assert "_content_emb" in cols
    assert any(k.startswith("_sum_cos_") for k in cols.keys())


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_knowledge_search_backfills_when_insufficient_similarity_results():
    km = KnowledgeManager()

    table = "KB_Backfill"
    km._create_table(name=table)

    # Create several rows where only one has the target signal in 'content'
    km._add_rows(
        table=table,
        rows=[
            {"title": "Alpha"},
            {"title": "Beta"},
            {"title": "Gamma", "content": "needle in haystack"},  # single match
            {"title": "Delta"},
            {"title": "Epsilon"},
            {"title": "Zeta"},
        ],
    )

    k = 4
    results = km._search(table=table, references={"content": "needle"}, k=k)

    assert len(results) == k
    titles = [r["title"] for r in results]
    # Gamma should be the top semantic match
    assert titles[0] == "Gamma"
    # Remaining should be backfilled from latest creation order without duplicates
    assert titles[1:4] == ["Zeta", "Epsilon", "Delta"]
