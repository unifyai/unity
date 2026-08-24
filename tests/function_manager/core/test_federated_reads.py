from __future__ import annotations

from types import SimpleNamespace

from unify.common.builtins import builtins_project
from unify.function_manager.function_manager import FunctionManager


def _manager_stub(*, include_primitives: bool = True) -> FunctionManager:
    fm = object.__new__(FunctionManager)
    fm._include_primitives = include_primitives
    fm._filter_scope = None
    fm._exclude_compositional_ids = None
    fm._exclude_primitive_ids = None
    fm._primitive_scope = object()
    fm._primitives_ctx = "Functions/Primitives"
    fm._registry = SimpleNamespace(
        primitive_row_filter=lambda _scope: "primitive_class == 'Primitives'",
    )
    fm._read_compositional_contexts = lambda: ["Functions/Compositional"]
    return fm


def test_filter_functions_delegates_federated_read_to_server(monkeypatch):
    fm = _manager_stub()
    fm._filter_scope = "language == 'python'"
    fm._exclude_primitive_ids = frozenset({99})
    builtins = builtins_project()
    calls = []

    def fake_get_logs_federated(**kwargs):
        calls.append(kwargs)
        return {
            "logs": [
                {
                    "name": "comp-2",
                    "implementation": "def comp_2(): pass",
                    "_federated_source": "compositional",
                },
                {
                    "name": "prim-1",
                    "implementation": None,
                    "_federated_source": "primitives",
                },
            ],
            "count": 5,
            "counts": {"compositional": 2, "primitives": 3},
        }

    monkeypatch.setattr("unisdk.get_logs_federated", fake_get_logs_federated)
    monkeypatch.setattr(
        "unify.function_manager.function_manager.list_private_fields",
        lambda *_args, **_kwargs: ["_embedding"],
    )

    rows = fm.filter_functions(
        filter="'tool' in docstring",
        offset=1,
        limit=2,
        include_implementations=False,
    )

    assert [row["name"] for row in rows] == ["comp-2", "prim-1"]
    assert all("implementation" not in row for row in rows)
    assert [row["_federated_source"] for row in rows] == [
        "compositional",
        "primitives",
    ]

    scoped_primitive_filter = (
        "(primitive_class == 'Primitives') and (function_id != 99)"
    )
    assert len(calls) == 1
    call = calls[0]
    assert call["filter"] == "'tool' in docstring"
    assert call["offset"] == 1
    assert call["limit"] == 2
    assert call["contexts"] == [
        {
            "context": "Functions/Compositional",
            "source": "compositional",
            "filter": "language == 'python'",
            "exclude_fields": ["_embedding"],
        },
        {
            "context": "Functions/Primitives",
            "source": "primitives",
            "filter": scoped_primitive_filter,
            "exclude_fields": ["_embedding"],
            "project_name": builtins,
        },
        {
            "context": "Functions/Primitives",
            "source": "primitives",
            "filter": f"({scoped_primitive_filter}) "
            'and metadata["source"] == "provider_backed"',
            "exclude_fields": ["_embedding"],
        },
    ]


def test_filter_functions_skips_primitive_contexts_when_disabled(monkeypatch):
    fm = _manager_stub(include_primitives=False)
    calls = []

    def fake_get_logs_federated(**kwargs):
        calls.append(kwargs)
        return {"logs": [{"name": "comp-1"}], "count": 1, "counts": {}}

    monkeypatch.setattr("unisdk.get_logs_federated", fake_get_logs_federated)
    monkeypatch.setattr(
        "unify.function_manager.function_manager.list_private_fields",
        lambda *_args, **_kwargs: [],
    )

    rows = fm.filter_functions(limit=5)

    assert [row["name"] for row in rows] == ["comp-1"]
    assert [spec["context"] for spec in calls[0]["contexts"]] == [
        "Functions/Compositional",
    ]


def test_search_functions_uses_federated_ranked_search_contexts(monkeypatch):
    fm = _manager_stub()
    fm._filter_scope = "language == 'python'"
    fm._exclude_compositional_ids = frozenset({1})
    builtins = builtins_project()
    captured = {}

    def fake_ranked_search(contexts, references, *, limit, **kwargs):
        captured["contexts"] = contexts
        captured["references"] = references
        captured["limit"] = limit
        captured["kwargs"] = kwargs
        return [
            {
                "name": "ranked",
                "implementation": "def ranked(): pass",
                "_federated_score": 0.1,
            },
        ]

    monkeypatch.setattr(
        "unify.function_manager.function_manager.federated_ranked_search",
        fake_ranked_search,
    )

    rows = fm.search_functions(
        query="rank useful functions",
        n=7,
        include_implementations=False,
    )

    # The activation pass annotates each surviving row with its ranking
    # components; the federated fields themselves must pass through intact.
    assert len(rows) == 1
    assert rows[0]["name"] == "ranked"
    assert rows[0]["_federated_score"] == 0.1
    assert {"_similarity", "_standing", "_retrieval_score"} <= set(rows[0])
    assert captured["references"] == {"embedding_text": "rank useful functions"}
    # Search overfetches beyond n so scope-dropped rows cannot leave the
    # caller short; the default policy is max(n + 4, n * 3) capped at 50.
    assert captured["limit"] == 21
    assert captured["kwargs"] == {"unique_id_field": "function_id", "backfill": True}

    contexts = captured["contexts"]
    assert [(spec.context, spec.project) for spec in contexts] == [
        ("Functions/Compositional", None),
        ("Functions/Primitives", builtins),
        ("Functions/Primitives", None),
    ]
    assert [spec.source for spec in contexts] == [
        "compositional",
        "primitives",
        "primitives",
    ]
    assert contexts[0].row_filter == "(language == 'python') and (function_id != 1)"
    assert contexts[1].row_filter == "primitive_class == 'Primitives'"
    assert contexts[2].row_filter == (
        "(primitive_class == 'Primitives') "
        'and metadata["source"] == "provider_backed"'
    )
    assert "embedding_text" in contexts[0].allowed_fields
    assert contexts[0].allowed_fields == contexts[1].allowed_fields
