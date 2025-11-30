import pytest

from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from tests.helpers import _handle_project


@pytest.mark.requires_real_unify
@_handle_project
def test_filter_orders_and_early_stop():
    km = KnowledgeManager(grouped=True)

    table = "KB_Grouped_Filter"
    km._create_table(name=table)

    km._add_rows(
        table=table,
        rows=[
            {"country": "US", "city": "NYC", "idnum": 1},
            {"country": "US", "city": "SF", "idnum": 2},
            {"country": "UK", "city": "London", "idnum": 3},
            {"country": "UK", "city": "Leeds", "idnum": 4},
            {"country": "UK", "city": "London", "idnum": 5},
        ],
    )

    results = km._filter(tables=[table], limit=100)

    assert isinstance(results, dict)
    grouped = results[table]
    assert isinstance(grouped, dict)
    assert "country" in grouped

    country_node = grouped["country"]
    assert isinstance(country_node, dict)
    assert country_node.get("count") == 5
    assert country_node.get("group_count") == 2

    # Top-level groups should be by country with stable order of first occurrence
    groups = {g["key"]: g["value"] for g in country_node["group"]}
    assert set(groups.keys()) == {"US", "UK"}

    # Early stop applies within US (subset size 2 → next column would be one-to-one)
    us_val = groups["US"]
    assert isinstance(us_val, list)
    assert len(us_val) == 2
    assert sorted({r["city"] for r in us_val}) == ["NYC", "SF"]

    # For UK (subset size 3), next column 'city' is not one-to-one → nested grouping continues
    uk_val = groups["UK"]
    assert isinstance(uk_val, dict)
    assert "city" in uk_val

    city_node = uk_val["city"]
    assert city_node.get("count") == 3
    assert city_node.get("group_count") == 2
    city_groups = {g["key"]: g["value"] for g in city_node["group"]}
    assert set(city_groups.keys()) == {"London", "Leeds"}
    assert isinstance(city_groups["Leeds"], list) and len(city_groups["Leeds"]) == 1
    assert isinstance(city_groups["London"], list) and len(city_groups["London"]) == 2


@pytest.mark.requires_real_unify
@_handle_project
def test_search_groups_on_low_cardinality_field():
    km = KnowledgeManager(grouped=True)

    table = "KB_Grouped_Search"
    km._create_table(name=table)

    km._add_rows(
        table=table,
        rows=[
            {"title": "A1", "content": "alpha target", "category": "A"},
            {"title": "A2", "content": "alpha something", "category": "A"},
            {"title": "B1", "content": "beta target", "category": "B"},
            {"title": "B2", "content": "beta something", "category": "B"},
        ],
    )

    # Ask for 4 rows; grouping should choose 'category' first (2 uniques)
    results = km._search(table=table, references=None, k=4)

    assert isinstance(results, dict)
    # Top-level column should be the one with smallest unique values (category)
    assert "category" in results
    groups = {g["key"]: g["value"] for g in results["category"]["group"]}
    assert set(groups.keys()) == {"A", "B"}


@pytest.mark.requires_real_unify
@_handle_project
def test_filter_join_nested_structure():
    km = KnowledgeManager(grouped=True)

    left = "KB_G_Left"
    right = "KB_G_Right"
    km._create_table(name=left)
    km._create_table(name=right)

    km._add_rows(
        table=left,
        rows=[
            {"k": 1, "grp": "X"},
            {"k": 2, "grp": "X"},
            {"k": 3, "grp": "Y"},
        ],
    )
    km._add_rows(
        table=right,
        rows=[
            {"k": 1, "kind": "foo"},
            {"k": 2, "kind": "bar"},
            {"k": 3, "kind": "bar"},
        ],
    )

    rows = km._filter_join(
        tables=[left, right],
        join_expr=f"{left}.k == {right}.k",
        select={f"{left}.grp": "grp", f"{right}.kind": "kind"},
        mode="inner",
        result_limit=10,
    )

    assert isinstance(rows, dict)
    assert "grp" in rows
    grp_node = rows["grp"]
    assert grp_node["group_count"] == 2 and grp_node["count"] == 3
    groups = {g["key"]: g["value"] for g in grp_node["group"]}
    assert set(groups.keys()) == {"X", "Y"}
    # For X (subset size 2), next column 'kind' has 2 uniques → early stop
    assert isinstance(groups["X"], list) and len(groups["X"]) == 2
    # For Y (subset size 1), nested is just the list
    assert isinstance(groups["Y"], list) and len(groups["Y"]) == 1


@pytest.mark.requires_real_unify
@_handle_project
def test_filter_multi_join_chained():
    km = KnowledgeManager(grouped=True)

    a = "KB_G_A"
    b = "KB_G_B"
    c = "KB_G_C"
    km._create_table(name=a)
    km._create_table(name=b)
    km._create_table(name=c)

    km._add_rows(
        table=a,
        rows=[{"k": 1, "grp": "X"}, {"k": 2, "grp": "X"}, {"k": 3, "grp": "Y"}],
    )
    km._add_rows(
        table=b,
        rows=[{"k": 1, "sub": "s1"}, {"k": 2, "sub": "s2"}, {"k": 3, "sub": "s1"}],
    )
    km._add_rows(table=c, rows=[{"sub": "s1", "tag": "T"}, {"sub": "s2", "tag": "U"}])

    joins = [
        {
            "tables": [a, b],
            "join_expr": f"{a}.k == {b}.k",
            "select": {f"{a}.grp": "grp", f"{b}.sub": "sub"},
            "mode": "inner",
        },
        {
            "tables": ["$prev", c],
            "join_expr": f"_.sub == {c}.sub",
            "select": {"_.grp": "grp", "_.sub": "sub", f"{c}.tag": "tag"},
            "mode": "inner",
        },
    ]

    rows = km._filter_multi_join(joins=joins, result_limit=10)

    assert isinstance(rows, dict)
    # Lowest-cardinality first: grp (2) vs tag (2) vs sub (2) → tie break by name → grp
    assert "grp" in rows
    grp_node = rows["grp"]
    assert grp_node["group_count"] == 2 and grp_node["count"] == 3
    groups = {g["key"]: g["value"] for g in grp_node["group"]}
    assert set(groups.keys()) == {"X", "Y"}
    # For X subset (2 rows), next col could be sub or tag (both 2 uniques) → early stop
    assert isinstance(groups["X"], list) and len(groups["X"]) == 2


@pytest.mark.requires_real_unify
@_handle_project
def test_search_join_groups():
    km = KnowledgeManager(grouped=True)

    left = "KB_GJ_Left"
    right = "KB_GJ_Right"
    km._create_table(name=left)
    km._create_table(name=right)

    km._add_rows(
        table=left,
        rows=[
            {"k": 1, "grp": "A"},
            {"k": 2, "grp": "A"},
            {"k": 3, "grp": "B"},
        ],
    )
    km._add_rows(
        table=right,
        rows=[
            {"k": 1, "kind": "x"},
            {"k": 2, "kind": "y"},
            {"k": 3, "kind": "y"},
        ],
    )

    rows = km._search_join(
        tables=[left, right],
        join_expr=f"{left}.k == {right}.k",
        select={f"{left}.grp": "grp", f"{right}.kind": "kind"},
        mode="inner",
        references=None,
        k=3,
    )

    assert isinstance(rows, dict)
    assert "grp" in rows
    grp_node = rows["grp"]
    assert grp_node["group_count"] == 2 and grp_node["count"] == 3
