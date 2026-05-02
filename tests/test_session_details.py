"""
tests/test_session_details.py
=============================

Unit tests for SessionDetails, focusing on the email_provider field
and its plumbing through populate / export_to_env / populate_from_env.
"""

import os
import json

import pytest

from unity.session_details import SessionDetails, SpaceSummary


class TestEmailProvider:
    def test_default_is_google_workspace(self):
        sd = SessionDetails()
        assert sd.assistant.email_provider == "google_workspace"

    def test_populate_sets_email_provider(self):
        sd = SessionDetails()
        sd.populate(assistant_email_provider="microsoft_365")
        assert sd.assistant.email_provider == "microsoft_365"

    def test_populate_defaults_to_google_workspace(self):
        sd = SessionDetails()
        sd.populate()
        assert sd.assistant.email_provider == "google_workspace"

    def test_export_and_populate_from_env_round_trips(self, monkeypatch):
        monkeypatch.delenv("ASSISTANT_EMAIL_PROVIDER", raising=False)
        sd = SessionDetails()
        sd.populate(assistant_email_provider="microsoft_365")
        sd.export_to_env()

        assert os.environ["ASSISTANT_EMAIL_PROVIDER"] == "microsoft_365"

        sd2 = SessionDetails()
        sd2.populate_from_env()
        assert sd2.assistant.email_provider == "microsoft_365"

    def test_reset_restores_default(self):
        sd = SessionDetails()
        sd.populate(assistant_email_provider="microsoft_365")
        assert sd.assistant.email_provider == "microsoft_365"

        sd.reset()
        assert sd.assistant.email_provider == "google_workspace"


class TestSpaceIds:
    def test_export_and_populate_from_env_round_trips(self, monkeypatch):
        monkeypatch.delenv("SPACE_IDS", raising=False)
        sd = SessionDetails()
        sd.populate(space_ids=[3, 7])
        sd.export_to_env()

        assert os.environ["SPACE_IDS"] == "3,7"

        sd2 = SessionDetails()
        sd2.populate_from_env()
        assert sd2.space_ids == [3, 7]
        assert sd2.assistant.space_ids == [3, 7]

        sd.populate(space_ids=[])
        sd.export_to_env()
        assert os.environ["SPACE_IDS"] == ""

        sd3 = SessionDetails()
        sd3.populate_from_env()
        assert sd3.space_ids == []

    def test_reset_restores_empty_memberships(self):
        sd = SessionDetails()
        sd.populate(space_ids=[3, 7])
        assert sd.space_ids == [3, 7]

        sd.reset()
        assert sd.space_ids == []
        assert sd.assistant.space_ids == []


class TestSpaceSummaries:
    def test_export_and_populate_from_env_round_trips(self, monkeypatch):
        monkeypatch.delenv("SPACE_SUMMARIES", raising=False)
        summaries = [
            {
                "space_id": 3,
                "name": "Repairs",
                "description": "South-East repairs patch daily operations.",
            },
        ]
        sd = SessionDetails()
        sd.populate(space_summaries=summaries)
        sd.export_to_env()

        assert json.loads(os.environ["SPACE_SUMMARIES"]) == summaries

        sd2 = SessionDetails()
        sd2.populate_from_env()
        assert sd2.space_summaries == [
            SpaceSummary(
                space_id=3,
                name="Repairs",
                description="South-East repairs patch daily operations.",
            ),
        ]

        sd.reset()
        assert sd.space_summaries == []

    @pytest.mark.parametrize(
        "summary",
        [
            {"space_id": True, "name": "Repairs", "description": "Valid text"},
            {"space_id": 3, "name": "", "description": "Valid text"},
            {"space_id": 3, "name": "Repairs", "description": None},
        ],
    )
    def test_rejects_malformed_space_summaries(self, summary):
        sd = SessionDetails()

        with pytest.raises(ValueError):
            sd.populate(space_summaries=[summary])


class TestContactIds:
    def test_defaults_and_shortcuts_use_subcontainer_storage(self):
        sd = SessionDetails()

        assert sd.self_contact_id == 0
        assert sd.boss_contact_id == 1

        sd.self_contact_id = 5
        sd.boss_contact_id = 6

        assert sd.assistant.self_contact_id == 5
        assert sd.assistant.contact_id == 5
        assert sd.user.boss_contact_id == 6
        assert sd.user.contact_id == 6

    def test_populate_sets_resolved_contact_ids(self):
        sd = SessionDetails()

        sd.populate(assistant_self_contact_id=42, user_boss_contact_id=43)

        assert sd.self_contact_id == 42
        assert sd.assistant.self_contact_id == 42
        assert sd.assistant.contact_id == 42
        assert sd.boss_contact_id == 43
        assert sd.user.boss_contact_id == 43
        assert sd.user.contact_id == 43

    def test_export_and_populate_from_env_round_trips(self, monkeypatch):
        monkeypatch.delenv("SELF_CONTACT_ID", raising=False)
        monkeypatch.delenv("BOSS_CONTACT_ID", raising=False)
        sd = SessionDetails()
        sd.populate(assistant_self_contact_id=42, user_boss_contact_id=43)
        sd.export_to_env()

        assert os.environ["SELF_CONTACT_ID"] == "42"
        assert os.environ["BOSS_CONTACT_ID"] == "43"

        sd2 = SessionDetails()
        sd2.populate_from_env()
        assert sd2.self_contact_id == 42
        assert sd2.boss_contact_id == 43

    def test_reset_restores_contact_id_defaults(self):
        sd = SessionDetails()
        sd.populate(assistant_self_contact_id=42, user_boss_contact_id=43)

        sd.reset()

        assert sd.self_contact_id == 0
        assert sd.boss_contact_id == 1
