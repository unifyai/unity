from __future__ import annotations

import pytest

import unify
from unity.common.data_store import DataStore


@pytest.fixture(autouse=True)
def _stable_project_and_clean_registry(monkeypatch):
    # Ensure a stable project id per test and clear singleton registry
    monkeypatch.setattr(unify, "active_project", lambda: "proj-test")
    DataStore._REGISTRY.clear()
    yield
    DataStore._REGISTRY.clear()


@pytest.mark.unit
def test_singleton_per_project_and_context(monkeypatch):
    monkeypatch.setattr(unify, "active_project", lambda: "P1")

    a = DataStore.for_context("Ctx/A", key_fields=("id",))
    b = DataStore.for_context("Ctx/A", key_fields=("id",))
    c = DataStore.for_context("Ctx/B", key_fields=("id",))
    d = DataStore.for_context("Ctx/A", key_fields=("id",), project="P2")

    assert a is b, "Same (project, context) should return the same instance"
    assert a is not c, "Different context should yield a different instance"
    assert (
        a is not d and c is not d
    ), "Different project should yield a different instance"


@pytest.mark.unit
def test_single_key_put_get_update_delete():
    ds = DataStore.for_context("C/Contacts", key_fields=("contact_id",))

    # Put with private fields and vector columns – they should be filtered out
    ds.put(
        {
            "contact_id": 2,
            "first_name": "Jane",
            "bio": "Hello",
            "_internal": 123,
            "notes_emb": [0.1, 0.2],
        },
    )

    row = ds[2]
    assert row == {"contact_id": 2, "first_name": "Jane", "bio": "Hello"}

    # Update replaces/merges top-level keys; private keys in updates are ignored
    ds.update(2, {"surname": "Roe", "_cfg": True, "notes_emb": [0.3]})
    row2 = ds[(2,)]  # 1-tuple form
    assert row2 == {
        "contact_id": 2,
        "first_name": "Jane",
        "bio": "Hello",
        "surname": "Roe",
    }

    # Delete and verify
    ds.delete(2)
    assert 2 not in ds
    assert len(ds) == 0

    # Miss behavior
    with pytest.raises(KeyError):
        _ = ds[2]
    assert ds.get(2) is None
    assert ds.get(2, default={"x": 1}) == {"x": 1}


@pytest.mark.unit
def test_composite_key_multiple_forms_and_snapshot():
    ds = DataStore.for_context("C/Tasks", key_fields=("task_id", "instance_id"))

    ds.put({"task_id": 10, "instance_id": 0, "status": "queued", "_meta": "x"})

    r1 = ds[(10, 0)]
    r2 = ds["10.0"]
    r3 = ds[[10, 0]]
    assert r1 == r2 == r3 == {"task_id": 10, "instance_id": 0, "status": "queued"}

    assert (10, 0) in ds
    assert "10.0" in ds

    snap = ds.snapshot()
    assert snap == {"10.0": {"task_id": 10, "instance_id": 0, "status": "queued"}}


@pytest.mark.unit
def test_miss_update_and_delete_raise_keyerror():
    ds = DataStore.for_context("C/Tasks", key_fields=("task_id", "instance_id"))

    with pytest.raises(KeyError):
        ds.update((99, 1), {"status": "queued"})

    with pytest.raises(KeyError):
        ds.delete("99.1")


@pytest.mark.unit
def test_clear_and_len_and_repr():
    ds = DataStore.for_context("C/Contacts", key_fields=("contact_id",))
    ds.put({"contact_id": 1, "first_name": "A"})
    ds.put({"contact_id": 2, "first_name": "B"})
    assert len(ds) == 2
    s = repr(ds)
    assert "DataStore(" in s and "keys=2" in s
    ds.clear()
    assert len(ds) == 0


@pytest.mark.unit
def test_key_coercion_of_negative_numbers_in_strings():
    ds = DataStore.for_context("C/Contacts", key_fields=("contact_id",))
    ds.put({"contact_id": -12, "first_name": "Neg"})

    # Lookup via various string forms
    assert ds["-12"]["first_name"] == "Neg"
    assert ds[" -12 "]["first_name"] == "Neg"
