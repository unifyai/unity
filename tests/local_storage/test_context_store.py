from __future__ import annotations

import pytest

import unify
from unity.common.context_store import TableStore


@pytest.fixture(autouse=True)
def _reset_ensured_cache():
    """Ensure TableStore's memoized ensure set is clear for each test."""
    TableStore._ENSURED.clear()
    yield
    TableStore._ENSURED.clear()


def test_ensure_creates_and_idempotent(monkeypatch):
    # Arrange: stable project and call counters
    monkeypatch.setattr(unify, "active_project", lambda: "proj-ctx")

    # Simulate 404 so ensure_context proceeds to creation
    def _fail_get(*args, **kwargs):
        raise Exception("Context not found")

    monkeypatch.setattr(unify, "get_context", _fail_get)

    calls = {"create_context": 0, "create_fields": 0}

    def _create_context(
        ctx,
        *,
        unique_keys=None,
        auto_counting=None,
        description=None,
        foreign_keys=None,
    ):
        calls["create_context"] += 1
        # Basic argument sanity
        assert ctx == "Test/Contacts"
        assert unique_keys == {"contact_id": "int"}
        assert auto_counting == {"contact_id": None}
        assert description == "Contacts table"

    def _create_fields(fields, *, context):
        calls["create_fields"] += 1
        assert context == "Test/Contacts"
        assert fields == {"first_name": {"type": "str"}, "surname": {"type": "str"}}

    monkeypatch.setattr(unify, "create_context", _create_context)
    monkeypatch.setattr(unify, "create_fields", _create_fields)

    store = TableStore(
        "Test/Contacts",
        unique_keys={"contact_id": "int"},
        auto_counting={"contact_id": None},
        description="Contacts table",
        fields={"first_name": {"type": "str"}, "surname": {"type": "str"}},
    )

    # Act: first ensure should create context and fields
    store.ensure_context()
    # Assert
    assert calls == {"create_context": 1, "create_fields": 1}

    # Act: second ensure should be a no-op due to memoization
    store.ensure_context()
    # Assert unchanged
    assert calls == {"create_context": 1, "create_fields": 1}


def test_get_columns_transforms(monkeypatch):
    # Arrange stable project and capture parameters
    monkeypatch.setattr(unify, "active_project", lambda: "proj-Z")
    seen = {"project_name": None, "context": None}

    def _get_fields(*, project, context):
        seen["project_name"] = project
        seen["context"] = context
        return {
            "first_name": {"data_type": "str"},
            "contact_id": {"data_type": "int"},
            "_internal": {
                "data_type": "dict",
            },  # still returned by backend – consumer filters
        }

    monkeypatch.setattr(unify, "get_fields", _get_fields)

    store = TableStore("Org/Contacts")
    cols = store.get_columns()

    # Assert mapping and the exact call arguments
    assert cols == {"first_name": "str", "contact_id": "int", "_internal": "dict"}
    assert seen == {"project_name": "proj-Z", "context": "Org/Contacts"}
