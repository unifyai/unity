"""
tests/test_session_details.py
=============================

Unit tests for SessionDetails, focusing on the email_provider field
and its plumbing through populate / export_to_env / populate_from_env.
"""

import os

from unity.session_details import SessionDetails


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
