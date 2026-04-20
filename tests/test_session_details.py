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

    def test_export_and_populate_from_env_round_trips(self):
        sd = SessionDetails()
        sd.populate(assistant_email_provider="microsoft_365")
        sd.export_to_env()

        assert os.environ["ASSISTANT_EMAIL_PROVIDER"] == "microsoft_365"

        sd2 = SessionDetails()
        sd2.populate_from_env()
        assert sd2.assistant.email_provider == "microsoft_365"

        # Cleanup
        os.environ.pop("ASSISTANT_EMAIL_PROVIDER", None)

    def test_reset_restores_default(self):
        sd = SessionDetails()
        sd.populate(assistant_email_provider="microsoft_365")
        assert sd.assistant.email_provider == "microsoft_365"

        sd.reset()
        assert sd.assistant.email_provider == "google_workspace"
