from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.contact_manager.contact_manager import ContactManager


@pytest.mark.requires_real_unify
@_handle_project
def test_contact_reduce_param_shapes():
    cm = ContactManager()

    # Seed a few contacts so metrics have real data to aggregate
    cm._create_contact(first_name="Alice", should_respond=True)
    cm._create_contact(first_name="Bob", should_respond=False)
    cm._create_contact(first_name="Carol", should_respond=True)

    # Single key, no grouping
    scalar = cm._reduce(metric="sum", keys="contact_id")
    assert isinstance(scalar, (int, float))

    # Multiple keys, no grouping
    multi = cm._reduce(metric="max", keys=["contact_id"])
    assert isinstance(multi, dict)
    assert set(multi.keys()) == {"contact_id"}

    # Single key, group_by string
    grouped_str = cm._reduce(
        metric="sum",
        keys="contact_id",
        group_by="should_respond",
    )
    assert isinstance(grouped_str, dict)

    # Multiple keys, group_by string
    grouped_str_multi = cm._reduce(
        metric="min",
        keys=["contact_id"],
        group_by="should_respond",
    )
    assert isinstance(grouped_str_multi, dict)

    # Single key, group_by list
    grouped_list = cm._reduce(
        metric="sum",
        keys="contact_id",
        group_by=["should_respond", "contact_id"],
    )
    assert isinstance(grouped_list, dict)

    # Multiple keys, group_by list
    grouped_list_multi = cm._reduce(
        metric="mean",
        keys=["contact_id"],
        group_by=["should_respond", "contact_id"],
    )
    assert isinstance(grouped_list_multi, dict)

    # Filter as string
    filtered_scalar = cm._reduce(
        metric="sum",
        keys="contact_id",
        filter="contact_id >= 0",
    )
    assert isinstance(filtered_scalar, (int, float))

    # Filter as per-key dict
    filtered_multi = cm._reduce(
        metric="sum",
        keys=["contact_id"],
        filter={"contact_id": "contact_id >= 0"},
    )
    assert isinstance(filtered_multi, dict)
