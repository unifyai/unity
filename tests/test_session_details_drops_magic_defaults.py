"""
tests/test_session_details_drops_magic_defaults.py
==================================================

Pins the invariant that ``AssistantDetails.contact_id`` and
``UserDetails.contact_id`` no longer default to the retired ``0`` / ``1``
magic numbers. The runtime expects both ids to arrive via bootstrap; reading
them before ``populate()`` returns ``None``.
"""

import os

from unity.session_details import (
    AssistantDetails,
    SessionDetails,
    UserDetails,
)


class TestContactIdDefaults:
    def test_assistant_details_contact_id_defaults_to_none(self):
        assert AssistantDetails().contact_id is None

    def test_user_details_contact_id_defaults_to_none(self):
        assert UserDetails().contact_id is None

    def test_fresh_session_details_exposes_none_before_populate(self):
        sd = SessionDetails()
        assert sd.assistant.contact_id is None
        assert sd.user.contact_id is None

    def test_populate_sets_both_contact_ids(self):
        sd = SessionDetails()
        sd.populate(self_contact_id=42, boss_contact_id=43)
        assert sd.assistant.contact_id == 42
        assert sd.user.contact_id == 43

    def test_populate_without_contact_ids_leaves_them_none(self):
        sd = SessionDetails()
        sd.populate()
        assert sd.assistant.contact_id is None
        assert sd.user.contact_id is None

    def test_reset_restores_contact_ids_to_none(self):
        sd = SessionDetails()
        sd.populate(self_contact_id=7, boss_contact_id=9)
        sd.reset()
        assert sd.assistant.contact_id is None
        assert sd.user.contact_id is None

    def test_export_and_populate_from_env_round_trips(self):
        sd = SessionDetails()
        sd.populate(self_contact_id=101, boss_contact_id=202)
        sd.export_to_env()

        assert os.environ["SELF_CONTACT_ID"] == "101"
        assert os.environ["BOSS_CONTACT_ID"] == "202"

        sd2 = SessionDetails()
        sd2.populate_from_env()
        assert sd2.assistant.contact_id == 101
        assert sd2.user.contact_id == 202

        os.environ.pop("SELF_CONTACT_ID", None)
        os.environ.pop("BOSS_CONTACT_ID", None)

    def test_empty_env_values_leave_contact_ids_none(self):
        os.environ["SELF_CONTACT_ID"] = ""
        os.environ["BOSS_CONTACT_ID"] = ""
        try:
            sd = SessionDetails()
            sd.populate_from_env()
            assert sd.assistant.contact_id is None
            assert sd.user.contact_id is None
        finally:
            os.environ.pop("SELF_CONTACT_ID", None)
            os.environ.pop("BOSS_CONTACT_ID", None)
