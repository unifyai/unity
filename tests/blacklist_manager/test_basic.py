import pytest

from unity.blacklist_manager.blacklist_manager import BlackListManager
from unity.conversation_manager.cm_types import Medium
from tests.helpers import _handle_project


@_handle_project
def test_create_and_filter():
    bl = BlackListManager()
    bl.clear()

    # Create a few entries across different media
    r1 = bl.create_blacklist_entry(
        medium=Medium.EMAIL,
        contact_detail="spam@example.com",
        reason="Phishing emails",
    )
    r2 = bl.create_blacklist_entry(
        medium=Medium.SMS_MESSAGE,
        contact_detail="+15551234567",
        reason="Spam SMS",
    )
    assert isinstance(r1["details"]["blacklist_id"], int)
    assert isinstance(r2["details"]["blacklist_id"], int)
    assert r1["details"]["blacklist_id"] != r2["details"]["blacklist_id"]

    # List all
    res = bl.filter_blacklist()
    entries = res["entries"]
    assert len(entries) == 2
    mediums = {e.medium for e in entries}
    details = {e.contact_detail for e in entries}
    assert Medium.EMAIL in mediums and Medium.SMS_MESSAGE in mediums
    assert "spam@example.com" in details and "+15551234567" in details

    # Filter by medium
    only_email = bl.filter_blacklist(filter="medium == 'email'")["entries"]
    assert len(only_email) == 1
    assert only_email[0].medium == Medium.EMAIL
    assert only_email[0].contact_detail == "spam@example.com"

    # Filter by exact contact_detail
    only_phone = bl.filter_blacklist(filter="contact_detail == '+15551234567'")[
        "entries"
    ]
    assert len(only_phone) == 1
    assert only_phone[0].medium == Medium.SMS_MESSAGE


@_handle_project
def test_update_entry():
    bl = BlackListManager()
    bl.clear()

    r = bl.create_blacklist_entry(
        medium=Medium.EMAIL,
        contact_detail="annoying@spam.com",
        reason="Initial",
    )
    bid = r["details"]["blacklist_id"]

    # Update reason
    bl.update_blacklist_entry(blacklist_id=bid, reason="Updated reason")
    after = bl.filter_blacklist(filter=f"blacklist_id == {bid}")["entries"]
    assert len(after) == 1
    assert after[0].reason == "Updated reason"

    # Update medium and contact_detail together
    bl.update_blacklist_entry(
        blacklist_id=bid,
        medium=Medium.SMS_MESSAGE,
        contact_detail="+441234567890",
    )
    after2 = bl.filter_blacklist(filter=f"blacklist_id == {bid}")["entries"]
    assert len(after2) == 1
    assert after2[0].medium == Medium.SMS_MESSAGE
    assert after2[0].contact_detail == "+441234567890"


@_handle_project
def test_update_requires_field():
    bl = BlackListManager()
    bl.clear()

    r = bl.create_blacklist_entry(
        medium=Medium.EMAIL,
        contact_detail="x@y.com",
        reason="x",
    )
    bid = r["details"]["blacklist_id"]
    with pytest.raises(ValueError):
        bl.update_blacklist_entry(blacklist_id=bid)


@_handle_project
def test_delete_entry():
    bl = BlackListManager()
    bl.clear()

    r1 = bl.create_blacklist_entry(
        medium=Medium.EMAIL,
        contact_detail="a@b.com",
        reason="r1",
    )
    r2 = bl.create_blacklist_entry(
        medium=Medium.SMS_MESSAGE,
        contact_detail="+11111111111",
        reason="r2",
    )
    id1 = r1["details"]["blacklist_id"]
    id2 = r2["details"]["blacklist_id"]
    assert id1 != id2

    # Delete the first
    bl.delete_blacklist_entry(blacklist_id=id1)
    all_after = bl.filter_blacklist()["entries"]
    assert len(all_after) == 1
    assert all_after[0].blacklist_id == id2


@_handle_project
def test_clear_resets_context():
    bl = BlackListManager()
    bl.clear()
    bl.create_blacklist_entry(
        medium=Medium.SMS_MESSAGE,
        contact_detail="+22222222222",
        reason="SMS spam",
    )
    assert len(bl.filter_blacklist()["entries"]) == 1
    bl.clear()
    assert len(bl.filter_blacklist()["entries"]) == 0


@_handle_project
def test_filter_shape_contains_shorthand_maps():
    bl = BlackListManager()
    bl.clear()
    bl.create_blacklist_entry(
        medium=Medium.EMAIL,
        contact_detail="s@e.com",
        reason="test",
    )
    res = bl.filter_blacklist()
    assert "blacklist_keys_to_shorthand" in res
    assert "shorthand_to_blacklist_keys" in res
    fwd = res["blacklist_keys_to_shorthand"]
    inv = res["shorthand_to_blacklist_keys"]
    # Round-trip a couple of fields
    assert fwd["blacklist_id"] in inv and inv[fwd["blacklist_id"]] == "blacklist_id"
    assert (
        fwd["contact_detail"] in inv and inv[fwd["contact_detail"]] == "contact_detail"
    )
