from __future__ import annotations

import pytest

from unity.guidance_manager.guidance_manager import GuidanceManager
from tests.helpers import _handle_project


@pytest.mark.unit
@_handle_project
def test_create_guidance():
    gm = GuidanceManager()
    out = gm._add_guidance(
        title="Setup demo",
        content="Steps to set up the product demo.",
    )
    gid = out["details"]["guidance_id"]

    rows = gm._filter(filter=f"guidance_id == {gid}")
    assert rows and rows[0].guidance_id == gid
    assert rows[0].title == "Setup demo"
    assert rows[0].content.startswith("Steps to set up")
    assert isinstance(rows[0].images, dict) and rows[0].images == {}


@pytest.mark.unit
@_handle_project
def test_update_guidance():
    gm = GuidanceManager()
    gid = gm._add_guidance(
        title="Onboarding Overview",
        content="We walk through onboarding steps.",
    )["details"]["guidance_id"]

    gm._update_guidance(
        guidance_id=gid,
        content="Updated walkthrough of onboarding steps for new users.",
        images={"[0:8]": 12},
    )

    rows = gm._filter(filter=f"guidance_id == {gid}")
    assert rows and rows[0].guidance_id == gid
    assert "Updated walkthrough" in rows[0].content
    assert rows[0].images == {"[0:8]": 12}


@pytest.mark.unit
@_handle_project
def test_delete_guidance():
    gm = GuidanceManager()
    gid = gm._add_guidance(
        title="Billing",
        content="Explains invoices and payment flows.",
    )["details"]["guidance_id"]

    # ensure present
    assert gm._filter(filter=f"guidance_id == {gid}")

    gm._delete_guidance(guidance_id=gid)
    assert len(gm._filter(filter=f"guidance_id == {gid}")) == 0


@pytest.mark.unit
@_handle_project
def test_list_columns_and_filter_title():
    gm = GuidanceManager()
    cols = gm._list_columns()
    # Basic schema keys should be present
    for key in ("guidance_id", "title", "content", "images"):
        assert key in cols

    gm._add_guidance(title="Comms", content="Prefer emails for updates")
    gm._add_guidance(title="Ops", content="Runbooks and SOPs")

    rows = gm._filter(filter="title == 'Comms'")
    assert rows and rows[0].title == "Comms"


@pytest.mark.unit
@_handle_project
def test_update_guidance_images_validation():
    gm = GuidanceManager()
    gid = gm._add_guidance(
        title="Docs",
        content="Documentation structure and guidelines.",
    )["details"]["guidance_id"]

    with pytest.raises(ValueError):
        gm._update_guidance(guidance_id=gid, images={"bad": 1})


@pytest.mark.unit
@_handle_project
def test_guidance_manager_clear():
    gm = GuidanceManager()

    # Seed a couple of guidance entries
    out1 = gm._add_guidance(title="Alpha", content="First entry")
    out2 = gm._add_guidance(title="Beta", content="Second entry")
    gid1 = out1["details"]["guidance_id"]
    gid2 = out2["details"]["guidance_id"]

    # Sanity: entries present before clear
    assert gm._filter(filter=f"guidance_id == {gid1}")
    assert gm._filter(filter=f"guidance_id == {gid2}")

    # Execute clear
    gm.clear()

    # After clear: schema should be present again
    cols = gm._list_columns()
    for key in ("guidance_id", "title", "content", "images"):
        assert key in cols

    # All prior guidance entries should be gone
    remaining_1 = gm._filter(filter=f"guidance_id == {gid1}")
    remaining_2 = gm._filter(filter=f"guidance_id == {gid2}")
    assert len(remaining_1) == 0
    assert len(remaining_2) == 0
