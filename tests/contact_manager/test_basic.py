from unity.contact_manager.contact_manager import ContactManager
from tests.helpers import _handle_project


@_handle_project
def test_create():
    contact_manager = ContactManager()
    contact_manager._create_contact(
        first_name="Dan",
        bio="A bit of a loser",
    )

    # Exclude both the built-in assistant (id 0) *and* the default user (id 1)
    user_contacts = [
        c
        for c in contact_manager.filter_contacts()["contacts"]
        if c.contact_id not in {0, 1}
    ]

    assert len(user_contacts) == 1, "Exactly one user contact should have been created"
    contact = user_contacts[0]

    # ID should be **greater** than 1 because 0 = assistant, 1 = default user
    assert contact.contact_id > 1
    assert contact.first_name == "Dan"
    assert contact.bio == "A bit of a loser"
    assert contact.surname is None
    assert contact.email_address is None
    assert contact.phone_number is None
    assert contact.rolling_summary is None
    assert contact.timezone is None

    hydrated = contact_manager._hydrate(contact.contact_id)
    assert hydrated is not None and hydrated.membership is not None
    assert hydrated.membership.should_respond is True
    assert hydrated.membership.response_policy == ContactManager.DEFAULT_RESPONSE_POLICY


@_handle_project
def test_update():
    contact_manager = ContactManager()

    # create
    contact_manager._create_contact(
        first_name="Dan",
    )

    # check (exclude assistant)
    user_contacts = [
        c
        for c in contact_manager.filter_contacts()["contacts"]
        if c.contact_id not in (0, 1)
    ]
    assert len(user_contacts) == 1
    contact = user_contacts[0]
    assert contact.first_name == "Dan"

    # update
    contact_manager.update_contact(
        contact_id=contact.contact_id,
        first_name="Daniel",
        bio="He's alright",
    )

    user_contacts = [
        c
        for c in contact_manager.filter_contacts()["contacts"]
        if c.contact_id not in (0, 1)
    ]
    assert len(user_contacts) == 1
    contact = user_contacts[0]
    assert contact.first_name == "Daniel"
    assert contact.bio == "He's alright"

    hydrated = contact_manager._hydrate(contact.contact_id)
    assert hydrated is not None and hydrated.membership is not None
    assert hydrated.membership.should_respond is True
    assert hydrated.membership.response_policy == ContactManager.DEFAULT_RESPONSE_POLICY


@_handle_project
def test_create_multiple():
    contact_manager = ContactManager()

    # first
    contact_manager._create_contact(
        first_name="Dan",
    )
    user_contacts = [
        c
        for c in contact_manager.filter_contacts()["contacts"]
        if c.contact_id not in (0, 1)
    ]
    assert len(user_contacts) == 1
    contact = user_contacts[0]
    assert contact.first_name == "Dan"

    # second
    custom_policy = "Treat them courteously but share no secrets."
    contact_manager._create_contact(
        first_name="Tom",
        response_policy=custom_policy,
    )
    user_contacts = [
        c
        for c in contact_manager.filter_contacts()["contacts"]
        if c.contact_id not in (0, 1)
    ]
    assert len(user_contacts) == 2
    tom_contact = next(c for c in user_contacts if c.first_name == "Tom")
    dan_contact = next(c for c in user_contacts if c.first_name == "Dan")

    # ensure IDs are unique and not 0
    assert tom_contact.contact_id not in {0, 1} and dan_contact.contact_id not in {0, 1}
    assert tom_contact.surname is None
    assert tom_contact.bio is None
    assert tom_contact.email_address is None
    assert tom_contact.phone_number is None
    assert tom_contact.rolling_summary is None
    assert tom_contact.timezone is None

    tom_h = contact_manager._hydrate(tom_contact.contact_id)
    dan_h = contact_manager._hydrate(dan_contact.contact_id)
    assert tom_h is not None and tom_h.membership is not None
    assert dan_h is not None and dan_h.membership is not None
    assert tom_h.membership.should_respond is True
    assert tom_h.membership.response_policy == custom_policy
    assert dan_h.membership.response_policy == ContactManager.DEFAULT_RESPONSE_POLICY


@_handle_project
def test_search():
    contact_manager = ContactManager()
    contact_manager._create_contact(
        first_name="Dan",
    )


# ────────────────────────────────────────────────────────────────────────────
#  Timezone (timezone) basic read/update                             #
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_timezone():
    cm = ContactManager()

    # Create a regular (non-system) contact
    out = cm._create_contact(first_name="Zed")
    cid = out["details"]["contact_id"]
    assert cid > 1

    # Initially timezone should be None
    c = cm.filter_contacts(filter=f"contact_id == {cid}")["contacts"][0]
    assert c.timezone is None

    # Update timezone to "Asia/Kolkata"
    cm.update_contact(contact_id=cid, timezone="Asia/Kolkata")
    c = cm.filter_contacts(filter=f"contact_id == {cid}")["contacts"][0]
    assert c.timezone == "Asia/Kolkata"

    # Update timezone to "America/New_York"
    cm.update_contact(contact_id=cid, timezone="America/New_York")
    c = cm.filter_contacts(filter=f"contact_id == {cid}")["contacts"][0]
    assert c.timezone == "America/New_York"

    # Try invalid timezone
    try:
        cm.update_contact(contact_id=cid, timezone="Invalid/Timezone")
        assert False, "Should have raised ValueError for invalid timezone"
    except ValueError:
        pass


# ────────────────────────────────────────────────────────────────────────────
#  System contacts should_respond defaults                              #
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_system_should_respond():
    """The self and boss overlays default to ``should_respond=True``.

    ``should_respond`` / ``response_policy`` live on the per-body
    ``ContactMembership`` overlay. System-contact provisioning
    materializes exactly two overlays (``relationship="self"`` for this
    body and ``relationship="boss"`` for the Hive/solo user); this test
    asserts both.
    """
    cm = ContactManager()

    self_h = next(
        h
        for cid in [c.contact_id for c in cm.filter_contacts()["contacts"]]
        if (h := cm._hydrate(cid)) is not None
        and h.membership is not None
        and h.membership.relationship == "self"
    )
    assert self_h.membership.should_respond is True
    assert self_h.membership.response_policy == ""

    boss_h = next(
        h
        for cid in [c.contact_id for c in cm.filter_contacts()["contacts"]]
        if (h := cm._hydrate(cid)) is not None
        and h.membership is not None
        and h.membership.relationship == "boss"
    )
    assert boss_h.membership.should_respond is True
    assert (
        boss_h.membership.response_policy == ContactManager.USER_MANAGER_RESPONSE_POLICY
    )


@_handle_project
def test_clear():
    cm = ContactManager()

    # Seed a couple of user contacts (ids should be > 1)
    out1 = cm._create_contact(first_name="Alpha")
    out2 = cm._create_contact(first_name="Beta")
    id1 = out1["details"]["contact_id"]
    id2 = out2["details"]["contact_id"]
    assert id1 > 1 and id2 > 1

    # Sanity: system contacts present before clear
    a = cm.filter_contacts(filter="contact_id == 0")["contacts"]
    u = cm.filter_contacts(filter="contact_id == 1")["contacts"]
    assert a and u

    # Execute clear
    cm.clear()

    assistants = cm.filter_contacts(filter="contact_id == 0")["contacts"]
    users = cm.filter_contacts(filter="contact_id == 1")["contacts"]
    assert assistants and users

    self_h = cm._hydrate(assistants[0].contact_id)
    boss_h = cm._hydrate(users[0].contact_id)
    assert self_h is not None and self_h.membership is not None
    assert boss_h is not None and boss_h.membership is not None
    assert self_h.membership.should_respond is True
    assert boss_h.membership.should_respond is True

    # All prior user contacts should be gone
    remaining_1 = cm.filter_contacts(filter=f"contact_id == {id1}")["contacts"]
    remaining_2 = cm.filter_contacts(filter=f"contact_id == {id2}")["contacts"]
    assert len(remaining_1) == 0
    assert len(remaining_2) == 0


@_handle_project
def test_update_empty_string_unique_fields():
    """Updating two contacts with empty-string unique fields must not raise.

    Reproduces a production bug where update_contact sent phone_number="" to
    Orchestra, violating the unique constraint when multiple contacts had no
    phone number.  The fix normalizes "" → None via the Pydantic model before
    persisting, so the empty value is dropped from the update payload.
    """
    cm = ContactManager()

    out_a = cm._create_contact(first_name="Alpha", email_address="alpha@test.com")
    out_b = cm._create_contact(first_name="Beta", email_address="beta@test.com")
    id_a = out_a["details"]["contact_id"]
    id_b = out_b["details"]["contact_id"]

    # Both contacts have no phone_number. Updating each with phone_number=""
    # must succeed (the empty string should be normalized away, not sent as a
    # duplicate unique value).
    cm.update_contact(contact_id=id_a, first_name="Alpha-Updated", phone_number="")
    cm.update_contact(contact_id=id_b, first_name="Beta-Updated", phone_number="")

    a = cm.filter_contacts(filter=f"contact_id == {id_a}")["contacts"][0]
    b = cm.filter_contacts(filter=f"contact_id == {id_b}")["contacts"][0]
    assert a.first_name == "Alpha-Updated"
    assert b.first_name == "Beta-Updated"
    assert a.phone_number is None
    assert b.phone_number is None

    # Same for email_address — both contacts already have distinct emails, so
    # passing email_address="" should be a no-op for that field (normalized away),
    # NOT clear the existing email.
    cm.update_contact(contact_id=id_a, first_name="Alpha-Final", email_address="")
    cm.update_contact(contact_id=id_b, first_name="Beta-Final", email_address="")

    a = cm.filter_contacts(filter=f"contact_id == {id_a}")["contacts"][0]
    b = cm.filter_contacts(filter=f"contact_id == {id_b}")["contacts"][0]
    assert a.first_name == "Alpha-Final"
    assert b.first_name == "Beta-Final"
    assert a.email_address == "alpha@test.com"
    assert b.email_address == "beta@test.com"

    # Now test the actual duplicate scenario for email: remove both emails first,
    # then confirm that updating both with email_address="" doesn't violate uniqueness.
    cm.update_contact(contact_id=id_a, email_address="alpha-tmp@test.com")
    cm.update_contact(contact_id=id_b, email_address="beta-tmp@test.com")
    # Clear emails by overwriting with distinct values then removing via the backend
    # isn't possible through update_contact (empty string = no-op), so instead
    # create two fresh contacts with no email and verify the same pattern works.
    out_c = cm._create_contact(first_name="Gamma")
    out_d = cm._create_contact(first_name="Delta")
    id_c = out_c["details"]["contact_id"]
    id_d = out_d["details"]["contact_id"]

    cm.update_contact(contact_id=id_c, first_name="Gamma-Updated", email_address="")
    cm.update_contact(contact_id=id_d, first_name="Delta-Updated", email_address="")

    c = cm.filter_contacts(filter=f"contact_id == {id_c}")["contacts"][0]
    d = cm.filter_contacts(filter=f"contact_id == {id_d}")["contacts"][0]
    assert c.first_name == "Gamma-Updated"
    assert d.first_name == "Delta-Updated"
    assert c.email_address is None
    assert d.email_address is None
