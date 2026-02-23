import pytest

from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from tests.helpers import _handle_project


@pytest.mark.requires_real_unify
@_handle_project
def test_multi_join_single_reference_basic():
    km = KnowledgeManager()

    authors = "MJ_Single_Authors"
    books = "MJ_Single_Books"
    reviews = "MJ_Single_Reviews"

    km._create_table(
        name=authors,
        columns={"author_id": "int", "author_name": "str"},
    )
    km._create_table(
        name=books,
        columns={
            "book_id": "int",
            "author_id": "int",
            "title": "str",
            "content": "str",
        },
    )
    km._create_table(
        name=reviews,
        columns={"review_id": "int", "book_id": "int", "review_text": "str"},
    )

    km._add_rows(
        table=authors,
        rows=[
            {"author_id": 1, "author_name": "A1"},
            {"author_id": 2, "author_name": "A2"},
            {"author_id": 3, "author_name": "A3"},
        ],
    )
    km._add_rows(
        table=books,
        rows=[
            {
                "book_id": 101,
                "author_id": 1,
                "title": "Quick reference for Linux",
                "content": "Short tips and cheatsheets for terminal usage",
            },
            {
                "book_id": 102,
                "author_id": 2,
                "title": "Messaging protocols overview",
                "content": "Long-form article about message queues and brokers",
            },
            {
                "book_id": 103,
                "author_id": 3,
                "title": "Deep learning tutorial",
                "content": "Comprehensive guide to neural networks and training",
            },
        ],
    )
    km._add_rows(
        table=reviews,
        rows=[
            {"review_id": 1001, "book_id": 101, "review_text": "solid reference"},
            {"review_id": 1002, "book_id": 102, "review_text": "systems"},
            {"review_id": 1003, "book_id": 103, "review_text": "ml"},
        ],
    )

    query = "short tips cheatsheet terminal"
    pipeline = [
        {
            "tables": [authors, books],
            "join_expr": f"{authors}.author_id == {books}.author_id",
            "select": {
                f"{books}.title": "title",
                f"{books}.content": "content",
                f"{books}.book_id": "book_id",
            },
        },
        {
            "tables": ["$prev", reviews],
            "join_expr": f"$prev.book_id == {reviews}.book_id",
            "select": {
                "$prev.title": "title",
                "$prev.content": "content",
                f"{reviews}.review_id": "review_id",
            },
        },
    ]

    results = km._search_multi_join(joins=pipeline, references={"content": query}, k=3)

    assert results[0]["title"] == "Quick reference for Linux"


@pytest.mark.requires_real_unify
@_handle_project
def test_multi_join_multi_columns_sum():
    km = KnowledgeManager()

    authors = "MJ_MultiCols_Authors"
    books = "MJ_MultiCols_Books"
    reviews = "MJ_MultiCols_Reviews"

    km._create_table(
        name=authors,
        columns={"author_id": "int", "author_name": "str"},
    )
    km._create_table(
        name=books,
        columns={
            "book_id": "int",
            "author_id": "int",
            "title": "str",
            "content": "str",
        },
    )
    km._create_table(
        name=reviews,
        columns={"review_id": "int", "book_id": "int", "review_text": "str"},
    )

    km._add_rows(
        table=authors,
        rows=[
            {"author_id": 1, "author_name": "A1"},
            {"author_id": 2, "author_name": "A2"},
            {"author_id": 3, "author_name": "A3"},
        ],
    )
    km._add_rows(
        table=books,
        rows=[
            {
                "book_id": 201,
                "author_id": 1,
                "title": "Compose LaTeX quickly",
                "content": "Short notes with snippets for equations and symbols",
            },
            {
                "book_id": 202,
                "author_id": 2,
                "title": "Logging problems",
                "content": "Comprehensive debugging guide with verbose traces",
            },
            {
                "book_id": 203,
                "author_id": 3,
                "title": "Text processing toolkit",
                "content": "Prefers regex for quick text processing in pipelines",
            },
        ],
    )
    km._add_rows(
        table=reviews,
        rows=[
            {"review_id": 2001, "book_id": 201, "review_text": "latex"},
            {"review_id": 2002, "book_id": 202, "review_text": "debugging"},
            {"review_id": 2003, "book_id": 203, "review_text": "regex"},
        ],
    )

    pipeline = [
        {
            "tables": [authors, books],
            "join_expr": f"{authors}.author_id == {books}.author_id",
            "select": {
                f"{books}.title": "title",
                f"{books}.content": "content",
                f"{books}.book_id": "book_id",
            },
        },
        {
            "tables": ["$prev", reviews],
            "join_expr": f"$prev.book_id == {reviews}.book_id",
            "select": {
                "$prev.title": "title",
                "$prev.content": "content",
                f"{reviews}.review_text": "review_text",
            },
        },
    ]

    query = "quick text snippets"
    references = {"content": query, "title": "irrelevant"}
    results = km._search_multi_join(joins=pipeline, references=references, k=2)

    assert len(results) == 2
    assert results[0]["title"] in {"Text processing toolkit", "Compose LaTeX quickly"}


@pytest.mark.requires_real_unify
@_handle_project
def test_multi_join_all_columns_default_derivation():
    km = KnowledgeManager()

    authors = "MJ_Expr_Authors"
    books = "MJ_Expr_Books"
    reviews = "MJ_Expr_Reviews"

    km._create_table(
        name=authors,
        columns={"author_id": "int", "author_name": "str"},
    )
    km._create_table(
        name=books,
        columns={
            "book_id": "int",
            "author_id": "int",
            "title": "str",
            "content": "str",
            "keywords": "str",
        },
    )
    km._create_table(
        name=reviews,
        columns={"review_id": "int", "book_id": "int", "review_text": "str"},
    )

    km._add_rows(
        table=authors,
        rows=[
            {"author_id": 10, "author_name": "Researcher"},
            {"author_id": 20, "author_name": "Operator"},
            {"author_id": 30, "author_name": "Communicator"},
        ],
    )
    km._add_rows(
        table=books,
        rows=[
            {
                "book_id": 301,
                "author_id": 10,
                "title": "Reading list summary",
                "content": "Prepare a summary of recent research papers",
                "keywords": "summaries literature",
            },
            {
                "book_id": 302,
                "author_id": 20,
                "title": "Email notifications config",
                "content": "Best practices for email deliverability and setup",
                "keywords": "email smtp dkim",
            },
            {
                "book_id": 303,
                "author_id": 30,
                "title": "SMS templates",
                "content": "Create message templates for quick outreach",
                "keywords": "sms texting",
            },
        ],
    )
    km._add_rows(
        table=reviews,
        rows=[
            {"review_id": 3001, "book_id": 301, "review_text": "ok"},
            {"review_id": 3002, "book_id": 302, "review_text": "great"},
            {"review_id": 3003, "book_id": 303, "review_text": "fine"},
        ],
    )

    pipeline = [
        {
            "tables": [authors, books],
            "join_expr": f"{authors}.author_id == {books}.author_id",
            "select": {
                f"{books}.title": "title",
                f"{books}.content": "content",
                f"{books}.keywords": "keywords",
                f"{books}.book_id": "book_id",
                f"{authors}.author_name": "author",
            },
        },
        {
            "tables": ["$prev", reviews],
            "join_expr": f"$prev.book_id == {reviews}.book_id",
            "select": {
                "$prev.title": "title",
                "$prev.content": "content",
                "$prev.keywords": "keywords",
                "$prev.author": "author",
            },
        },
    ]

    expr = "str({title}) + ' ' + str({content}) + ' ' + str({author}) + ' ' + str({keywords})"
    query = "best practices for email"
    results = km._search_multi_join(joins=pipeline, references={expr: query}, k=2)

    assert len(results) >= 1
    assert results[0]["title"] == "Email notifications config"


@pytest.mark.requires_real_unify
@_handle_project
def test_multi_join_mean_of_cosine_ranking():
    km = KnowledgeManager()

    authors = "MJ_SumCos_Authors"
    books = "MJ_SumCos_Books"
    reviews = "MJ_SumCos_Reviews"

    km._create_table(
        name=authors,
        columns={"author_id": "int", "author_name": "str"},
    )
    km._create_table(
        name=books,
        columns={
            "book_id": "int",
            "author_id": "int",
            "title": "str",
            "content": "str",
        },
    )
    km._create_table(
        name=reviews,
        columns={"review_id": "int", "book_id": "int", "review_text": "str"},
    )

    km._add_rows(
        table=authors,
        rows=[
            {"author_id": 1, "author_name": "A1"},
            {"author_id": 2, "author_name": "A2"},
            {"author_id": 3, "author_name": "A3"},
        ],
    )
    km._add_rows(
        table=books,
        rows=[
            {
                "book_id": 401,
                "author_id": 1,
                "title": "Neural network training guide",
                "content": "Getting started guide for training deep networks",
            },
            {
                "book_id": 402,
                "author_id": 2,
                "title": "Onboarding process",
                "content": "Haven't started yet",
            },
            {
                "book_id": 403,
                "author_id": 3,
                "title": "Tax preparation tips",
                "content": "Getting started with your taxes",
            },
        ],
    )
    km._add_rows(
        table=reviews,
        rows=[
            {
                "review_id": 4001,
                "book_id": 401,
                "review_text": "deep learning, tutorial",
            },
            {"review_id": 4002, "book_id": 402, "review_text": "tutorial"},
            {"review_id": 4003, "book_id": 403, "review_text": "finance, guide"},
        ],
    )

    pipeline = [
        {
            "tables": [authors, books],
            "join_expr": f"{authors}.author_id == {books}.author_id",
            "select": {
                f"{books}.title": "title",
                f"{books}.content": "content",
                f"{books}.book_id": "book_id",
            },
        },
        {
            "tables": ["$prev", reviews],
            "join_expr": f"$prev.book_id == {reviews}.book_id",
            "select": {
                "$prev.title": "title",
                "$prev.content": "content",
                f"{reviews}.review_text": "review_text",
            },
        },
    ]

    references = {"review_text": "deep learning", "content": "getting started guide"}
    results = km._search_multi_join(joins=pipeline, references=references, k=3)

    assert len(results) == 3
    titles = [r["title"] for r in results]
    assert titles[0] == "Neural network training guide"
    assert titles.index("Neural network training guide") < titles.index(
        "Onboarding process",
    )
    assert titles.index("Neural network training guide") < titles.index(
        "Tax preparation tips",
    )


@pytest.mark.requires_real_unify
@_handle_project
def test_multi_join_backfills_when_insufficient_similarity_results():
    km = KnowledgeManager()

    left = "KBMJ_Left"
    right = "KBMJ_Right"
    third = "KBMJ_Third"
    km._create_table(name=left)
    km._create_table(name=right)
    km._create_table(name=third)

    # Left table: only one row has the semantic signal in 'content'
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

    # Right table: simple attributes to join on user_id
    km._add_rows(
        table=right,
        rows=[
            {"user_id": 1, "note": "r1"},
            {"user_id": 2, "note": "r2"},
            {"user_id": 3, "note": "r3"},
            {"user_id": 4, "note": "r4"},
            {"user_id": 5, "note": "r5"},
            {"user_id": 6, "note": "r6"},
        ],
    )

    # Third table: second hop join on user_id (acts as a pass-through)
    km._add_rows(
        table=third,
        rows=[
            {"user_id": 1, "tag": "t1"},
            {"user_id": 2, "tag": "t2"},
            {"user_id": 3, "tag": "t3"},
            {"user_id": 4, "tag": "t4"},
            {"user_id": 5, "tag": "t5"},
            {"user_id": 6, "tag": "t6"},
        ],
    )

    k = 4
    joins = [
        {
            "tables": [left, right],
            "join_expr": f"{left}.user_id == {right}.user_id",
            "select": {
                f"{left}.user_id": "user_id",
                f"{left}.title": "title",
                f"{left}.content": "content",
                f"{right}.note": "note",
            },
        },
        {
            "tables": ["$prev", third],
            "join_expr": f"_.user_id == {third}.user_id",
            "select": {
                f"_.user_id": "user_id",
                f"_.title": "title",
                f"_.content": "content",
                f"_.note": "note",
                f"{third}.tag": "tag",
            },
        },
    ]

    results = km._search_multi_join(joins=joins, references={"content": "needle"}, k=k)

    assert len(results) == k
    titles = [r.get("title") for r in results]
    # The row containing the semantic signal should be first
    assert titles[0] == "Gamma"
    # Remaining should be backfilled from latest creation order without duplicates
    assert titles[1:4] == ["Zeta", "Epsilon", "Delta"]

    # When references is None/empty, skip semantic search and return most recent rows from final joined context
    recent_only = km._search_multi_join(joins=joins, references=None, k=3)
    assert [r.get("title") for r in recent_only] == ["Zeta", "Epsilon", "Delta"]

    recent_only_empty = km._search_multi_join(joins=joins, references={}, k=3)
    assert [r.get("title") for r in recent_only_empty] == ["Zeta", "Epsilon", "Delta"]
