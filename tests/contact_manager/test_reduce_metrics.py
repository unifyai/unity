from __future__ import annotations

import pytest
import time
import unify

from tests.helpers import _handle_project
from unity.common.context_registry import ContextRegistry
from unity.contact_manager.contact_manager import ContactManager
from unity.session_details import SESSION_DETAILS


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


@pytest.mark.requires_real_unify
@_handle_project
def test_contact_reduce_reads_personal_and_accessible_space_roots():
    space_id = time.time_ns()
    SESSION_DETAILS.space_ids = [space_id]

    try:
        cm = ContactManager()
        marker = f"reduce-marker-{space_id}"
        cm._create_contact(first_name="Personal Reduce", bio=marker)
        cm._create_contact(
            first_name="Shared Reduce",
            bio=marker,
            destination=f"space:{space_id}",
        )

        count = cm._reduce(
            metric="count",
            keys="contact_id",
            filter=f"bio == '{marker}'",
        )
        assert count == 2

        grouped = cm._reduce(
            metric="count",
            keys="contact_id",
            filter=f"bio == '{marker}'",
            group_by="first_name",
        )
        assert grouped == {"Personal Reduce": 1, "Shared Reduce": 1}

        SESSION_DETAILS.space_ids = []
        ContextRegistry.clear()
        assert (
            cm._reduce(
                metric="count",
                keys="contact_id",
                filter=f"bio == '{marker}'",
            )
            == 1
        )
    finally:
        try:
            unify.delete_context(f"Spaces/{space_id}/Contacts")
        except Exception:
            pass
        SESSION_DETAILS.space_ids = []
        ContextRegistry.clear()
