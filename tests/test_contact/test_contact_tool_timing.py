from __future__ import annotations

import time
import pytest

from unity.contact_manager.contact_manager import ContactManager
from tests.helpers import _handle_project


# One test per tool, with lightweight assertions and commented timing bounds.


@pytest.mark.unit
@_handle_project
def test_tool_list_columns_timing():
    cm = ContactManager()
    t0 = time.perf_counter()
    cols = cm._list_columns()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(cols, dict) and cols
    assert elapsed_ms < 2250
    print(f"elapsed: {elapsed_ms} < 2250")


@pytest.mark.unit
@_handle_project
def test_tool_create_contact_timing():
    cm = ContactManager()
    t0 = time.perf_counter()
    out = cm._create_contact(first_name="PerfCreate")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out["outcome"]
    assert isinstance(out.get("details", {}).get("contact_id"), int)
    assert elapsed_ms < 1500
    print(f"elapsed: {elapsed_ms} < 1500")


@pytest.mark.unit
@_handle_project
def test_tool_update_contact_timing():
    cm = ContactManager()
    cid = cm._create_contact(first_name="PerfUpd")["details"]["contact_id"]
    t0 = time.perf_counter()
    out = cm._update_contact(contact_id=cid, surname="Case")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out["outcome"] and out["details"]["contact_id"] == cid
    assert elapsed_ms < 2300
    print(f"elapsed: {elapsed_ms} < 2300")


@pytest.mark.unit
@_handle_project
def test_tool_filter_contacts_timing():
    cm = ContactManager()
    cid = cm._create_contact(first_name="PerfFilter")["details"]["contact_id"]
    t0 = time.perf_counter()
    rows = cm._filter_contacts(filter=f"contact_id == {cid}")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert rows and rows[0].contact_id == cid
    assert elapsed_ms < 1800
    print(f"elapsed: {elapsed_ms} < 1800")


@pytest.mark.unit
@_handle_project
def test_tool_delete_contact_timing():
    cm = ContactManager()
    cid = cm._create_contact(first_name="PerfDelete")["details"]["contact_id"]
    t0 = time.perf_counter()
    out = cm._delete_contact(contact_id=cid)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out["outcome"] and out["details"]["contact_id"] == cid
    assert elapsed_ms < 3300
    print(f"elapsed: {elapsed_ms} < 3300")


@pytest.mark.unit
@_handle_project
def test_tool_create_custom_column_timing():
    cm = ContactManager()
    col = "timing_nickname"
    # ensure it's not present
    if col in cm._list_columns():
        cm._delete_custom_column(column_name=col)
    t0 = time.perf_counter()
    resp = cm._create_custom_column(column_name=col, column_type="str")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(resp, dict)
    assert col in cm._list_columns()
    assert elapsed_ms < 1200
    print(f"elapsed: {elapsed_ms} < 1200")


@pytest.mark.unit
@_handle_project
def test_tool_delete_custom_column_timing():
    cm = ContactManager()
    col = "timing_delete_me"
    try:
        cm._create_custom_column(column_name=col, column_type="str")
    except Exception:
        pass
    t0 = time.perf_counter()
    resp = cm._delete_custom_column(column_name=col)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(resp, dict)
    assert col not in cm._list_columns()
    assert elapsed_ms < 1800
    print(f"elapsed: {elapsed_ms} < 1800")


@pytest.mark.unit
@_handle_project
def test_tool_merge_contacts_timing():
    cm = ContactManager()
    c1 = cm._create_contact(first_name="John", surname="Doe")["details"]["contact_id"]
    c2 = cm._create_contact(first_name="Johnny", surname="Roe")["details"]["contact_id"]
    t0 = time.perf_counter()
    out = cm._merge_contacts(
        contact_id_1=c1,
        contact_id_2=c2,
        overrides={"contact_id": 1, "first_name": 2, "surname": 2},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out["outcome"] and out["details"]["kept_contact_id"] == c1
    assert elapsed_ms < 6000
    print(f"elapsed: {elapsed_ms} < 6000")


@pytest.mark.unit
@_handle_project
def test_tool_search_contacts_timing():
    cm = ContactManager()
    cm._create_contact(first_name="Alice", bio="enjoys emails")
    cm._create_contact(first_name="Bob", bio="prefers texts")
    t0 = time.perf_counter()
    results = cm._search_contacts(references={"bio": "emails"}, k=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(results, list) and results
    assert elapsed_ms < 6800
    print(f"elapsed: {elapsed_ms} < 6800")
