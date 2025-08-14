import pytest

from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from tests.helpers import _handle_project


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_knowledge_search_join_single_reference_basic():
    km = KnowledgeManager()

    articles = "J_Single_Articles"
    meta = "J_Single_Meta"
    km._create_table(
        name=articles,
        columns={
            "article_id": "int",
            "title": "str",
            "content": "str",
            "meta_id": "int",
        },
    )
    km._create_table(
        name=meta,
        columns={
            "meta_id": "int",
            "tags": "str",
            "category": "str",
        },
    )

    km._add_rows(
        table=meta,
        rows=[
            {"meta_id": 1, "tags": "linux, cheatsheet", "category": "Ops"},
            {"meta_id": 2, "tags": "messaging, queues", "category": "Systems"},
            {"meta_id": 3, "tags": "neural networks", "category": "Research"},
        ],
    )
    km._add_rows(
        table=articles,
        rows=[
            {
                "article_id": 101,
                "title": "Quick reference for Linux",
                "content": "Short tips and cheatsheets for terminal usage",
                "meta_id": 1,
            },
            {
                "article_id": 102,
                "title": "Messaging protocols overview",
                "content": "Long-form article about message queues and brokers",
                "meta_id": 2,
            },
            {
                "article_id": 103,
                "title": "Deep learning tutorial",
                "content": "Comprehensive guide to neural networks and training",
                "meta_id": 3,
            },
        ],
    )

    query = "short tips cheatsheet terminal"
    results = km._search_join(
        tables=[articles, meta],
        join_expr=f"{articles}.meta_id == {meta}.meta_id",
        select={
            f"{articles}.title": "title",
            f"{articles}.content": "content",
            f"{meta}.tags": "tags",
            f"{meta}.category": "category",
        },
        references={"content": query},
        k=3,
    )

    assert results[0]["title"] == "Quick reference for Linux"


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_knowledge_search_join_multi_columns_sum():
    km = KnowledgeManager()

    articles = "J_MultiCols_Articles"
    meta = "J_MultiCols_Meta"
    km._create_table(
        name=articles,
        columns={
            "article_id": "int",
            "title": "str",
            "content": "str",
            "meta_id": "int",
        },
    )
    km._create_table(
        name=meta,
        columns={
            "meta_id": "int",
            "tags": "str",
        },
    )

    km._add_rows(
        table=meta,
        rows=[
            {"meta_id": 1, "tags": "latex"},
            {"meta_id": 2, "tags": "debugging"},
            {"meta_id": 3, "tags": "regex"},
        ],
    )
    km._add_rows(
        table=articles,
        rows=[
            {
                "article_id": 201,
                "title": "Compose LaTeX quickly",
                "content": "Short notes with snippets for equations and symbols",
                "meta_id": 1,
            },
            {
                "article_id": 202,
                "title": "Logging problems",
                "content": "Comprehensive debugging guide with verbose traces",
                "meta_id": 2,
            },
            {
                "article_id": 203,
                "title": "Text processing toolkit",
                "content": "Prefers regex for quick text processing in pipelines",
                "meta_id": 3,
            },
        ],
    )

    query = "quick text snippets"
    references = {"content": query, "title": "irrelevant"}
    results = km._search_join(
        tables=[articles, meta],
        join_expr=f"{articles}.meta_id == {meta}.meta_id",
        select={
            f"{articles}.title": "title",
            f"{articles}.content": "content",
            f"{meta}.tags": "tags",
        },
        references=references,
        k=2,
    )

    assert len(results) == 2
    assert results[0]["title"] in {"Text processing toolkit", "Compose LaTeX quickly"}


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_knowledge_search_join_all_columns_default_derivation():
    km = KnowledgeManager()

    articles = "J_Expr_Articles"
    meta = "J_Expr_Meta"
    km._create_table(
        name=articles,
        columns={
            "article_id": "int",
            "title": "str",
            "content": "str",
            "meta_id": "int",
            "keywords": "str",
        },
    )
    km._create_table(
        name=meta,
        columns={
            "meta_id": "int",
            "category": "str",
        },
    )

    km._add_rows(
        table=meta,
        rows=[
            {"meta_id": 10, "category": "Research"},
            {"meta_id": 20, "category": "Operations"},
            {"meta_id": 30, "category": "Communications"},
        ],
    )
    km._add_rows(
        table=articles,
        rows=[
            {
                "article_id": 301,
                "title": "Reading list summary",
                "content": "Prepare a summary of recent research papers",
                "keywords": "summaries literature",
                "meta_id": 10,
            },
            {
                "article_id": 302,
                "title": "Email notifications config",
                "content": "Best practices for email deliverability and setup",
                "keywords": "email smtp dkim",
                "meta_id": 20,
            },
            {
                "article_id": 303,
                "title": "SMS templates",
                "content": "Create message templates for quick outreach",
                "keywords": "sms texting",
                "meta_id": 30,
            },
        ],
    )

    expr = "str({title}) + ' ' + str({content}) + ' ' + str({category}) + ' ' + str({keywords})"
    query = "best practices for email"
    results = km._search_join(
        tables=[articles, meta],
        join_expr=f"{articles}.meta_id == {meta}.meta_id",
        select={
            f"{articles}.title": "title",
            f"{articles}.content": "content",
            f"{articles}.keywords": "keywords",
            f"{meta}.category": "category",
        },
        references={expr: query},
        k=2,
    )

    assert len(results) >= 1
    assert results[0]["title"] == "Email notifications config"


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_knowledge_search_join_sum_of_cosine_ranking():
    km = KnowledgeManager()

    articles = "J_SumCos_Articles"
    meta = "J_SumCos_Meta"
    km._create_table(
        name=articles,
        columns={
            "article_id": "int",
            "title": "str",
            "content": "str",
            "meta_id": "int",
        },
    )
    km._create_table(
        name=meta,
        columns={
            "meta_id": "int",
            "tags": "str",
        },
    )

    km._add_rows(
        table=meta,
        rows=[
            {"meta_id": 1, "tags": "deep learning, tutorial"},
            {"meta_id": 2, "tags": "tutorial"},
            {"meta_id": 3, "tags": "finance, guide"},
        ],
    )
    km._add_rows(
        table=articles,
        rows=[
            {
                "article_id": 401,
                "title": "Neural network training guide",
                "content": "Getting started guide for training deep networks",
                "meta_id": 1,
            },
            {
                "article_id": 402,
                "title": "Onboarding process",
                "content": "Haven't started yet",
                "meta_id": 2,
            },
            {
                "article_id": 403,
                "title": "Tax preparation tips",
                "content": "Getting started with your taxes",
                "meta_id": 3,
            },
        ],
    )

    references = {"tags": "deep learning", "content": "getting started guide"}
    results = km._search_join(
        tables=[articles, meta],
        join_expr=f"{articles}.meta_id == {meta}.meta_id",
        select={
            f"{articles}.title": "title",
            f"{articles}.content": "content",
            f"{meta}.tags": "tags",
        },
        references=references,
        k=3,
    )

    assert len(results) == 3
    titles = [r["title"] for r in results]
    assert titles[0] == "Neural network training guide"
    assert titles.index("Neural network training guide") < titles.index(
        "Onboarding process",
    )
    assert titles.index("Neural network training guide") < titles.index(
        "Tax preparation tips",
    )


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_knowledge_search_join_backfills_when_insufficient_similarity_results():
    km = KnowledgeManager()

    left = "KBJ_Left"
    right = "KBJ_Right"
    km._create_table(name=left)
    km._create_table(name=right)

    # Create matching user_ids across left/right; only one left row contains the signal
    km._add_rows(
        table=left,
        rows=[
            {"user_id": 1, "title": "Alpha"},
            {"user_id": 2, "title": "Beta"},
            {"user_id": 3, "title": "Gamma", "content": "needle in haystack"},
            {"user_id": 4, "title": "Delta"},
            {"user_id": 5, "title": "Epsilon"},
            {"user_id": 6, "title": "Zeta"},
        ],
    )

    km._add_rows(
        table=right,
        rows=[
            {"user_id": 1, "note": "n/a"},
            {"user_id": 2, "note": "n/a"},
            {"user_id": 3, "note": "n/a"},
            {"user_id": 4, "note": "n/a"},
            {"user_id": 5, "note": "n/a"},
            {"user_id": 6, "note": "n/a"},
        ],
    )

    # Perform join search: only one semantic match; ensure backfill to k
    k = 4
    results = km._search_join(
        tables=[left, right],
        join_expr=f"{left}.user_id == {right}.user_id",
        select={
            f"{left}.user_id": "user_id",
            f"{left}.title": "title",
            f"{left}.content": "content",
            f"{right}.note": "note",
        },
        references={"content": "needle"},
        k=k,
    )

    assert len(results) == k
    titles = [r.get("title") for r in results]
    # The joined row containing the semantic signal should be first
    assert titles[0] == "Gamma"
    # Remaining should be backfilled from latest creation order without duplicates
    assert titles[1:4] == ["Zeta", "Epsilon", "Delta"]

    # When references is None/empty, skip semantic search and return most recent joined rows
    recent_only = km._search_join(
        tables=[left, right],
        join_expr=f"{left}.user_id == {right}.user_id",
        select={
            f"{left}.user_id": "user_id",
            f"{left}.title": "title",
            f"{left}.content": "content",
            f"{right}.note": "note",
        },
        references=None,
        k=3,
    )
    assert [r.get("title") for r in recent_only] == ["Zeta", "Epsilon", "Delta"]

    recent_only_empty = km._search_join(
        tables=[left, right],
        join_expr=f"{left}.user_id == {right}.user_id",
        select={
            f"{left}.user_id": "user_id",
            f"{left}.title": "title",
            f"{left}.content": "content",
            f"{right}.note": "note",
        },
        references={},
        k=3,
    )
    assert [r.get("title") for r in recent_only_empty] == ["Zeta", "Epsilon", "Delta"]
