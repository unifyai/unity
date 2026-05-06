"""
Tests for DataManager ops builder helpers.

Plot and table-view ops tests were removed alongside ``plot()`` / ``table_view()``
on DataManager.
"""

from __future__ import annotations

import pytest
import requests
from unify.utils.http import RequestError

from unity.data_manager.ops import ingest_ops, mutation_ops


def _duplicate_key_error() -> RequestError:
    response = requests.Response()
    response.status_code = 400
    response._content = b'{"detail":"Duplicate composite key already exists"}'
    response.url = "https://api.unify.ai/v0/logs"
    return RequestError("https://api.unify.ai/v0/logs", "POST", response)


def test_insert_rows_duplicate_key_errors_raise_by_default(monkeypatch):
    def fake_create_logs(**_kwargs):
        raise _duplicate_key_error()

    monkeypatch.setattr(mutation_ops, "unify_create_logs", fake_create_logs)

    with pytest.raises(RequestError, match="Duplicate composite key"):
        mutation_ops.insert_rows_impl("Data/test", [{"key": "a"}])


def test_insert_rows_can_treat_duplicate_private_keys_as_committed(monkeypatch):
    def fake_create_logs(**_kwargs):
        raise _duplicate_key_error()

    monkeypatch.setattr(mutation_ops, "unify_create_logs", fake_create_logs)

    assert (
        mutation_ops.insert_rows_impl(
            "Data/test",
            [{"key": "a"}],
            ignore_duplicate_composite_key_errors=True,
        )
        == []
    )


def test_insert_chunk_forwards_duplicate_key_replay_flag(monkeypatch):
    captured = {}

    def fake_insert_rows_impl(context, rows, **kwargs):
        captured["context"] = context
        captured["rows"] = rows
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(ingest_ops, "insert_rows_impl", fake_insert_rows_impl)

    insert = ingest_ops._make_insert_chunk_func(
        "Data/test",
        [{"key": "a"}],
        ignore_duplicate_composite_key_errors=True,
    )

    assert insert() == {"inserted_ids": [], "row_count": 1}
    assert captured == {
        "context": "Data/test",
        "rows": [{"key": "a"}],
        "kwargs": {
            "add_to_all_context": False,
            "ignore_duplicate_composite_key_errors": True,
        },
    }
