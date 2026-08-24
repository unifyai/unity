"""Assistant identity env overrides for CM integration tests."""

from __future__ import annotations

import os

TEST_ASSISTANT_EMAIL = "assistant@test.example.com"
TEST_ASSISTANT_NUMBER = "+15550001000"
TEST_ASSISTANT_WHATSAPP_NUMBER = "+15550001000"


def ensure_test_assistant_identity_env() -> None:
    """Pin the assistant identity CM tests boot with, ignoring machine state.

    Comms identity is forced to fake values so medium-faithful send tools can
    be exposed in CM eval tests. ``setdefault`` is not enough: a blank
    ``ASSISTANT_NUMBER=`` entry loaded from ``.env`` is already set and leaves
    ``SESSION_DETAILS.assistant.number`` empty, hiding ``send_sms`` /
    ``send_email`` from the live tool schema.

    Platform identity is forced blank unconditionally: a developer's ``.env``
    may pin ``ASSISTANT_ID`` / ``OWNER_TEAM_ID`` to a hosted assistant, and a
    CM boot inheriting them derives ownership from that assistant's platform
    record — which the backend the tests run against has no reason to hold, so
    the boot's fail-closed ownership binding refuses to start. Blank rather
    than deleted: ``load_dotenv()`` (run when ``unify.conversation_manager``
    modules import) does not override existing variables, even blank ones,
    while a deleted variable would be re-loaded straight back from ``.env``.
    """
    if not (os.environ.get("ASSISTANT_EMAIL") or "").strip():
        os.environ["ASSISTANT_EMAIL"] = TEST_ASSISTANT_EMAIL
    if not (os.environ.get("ASSISTANT_NUMBER") or "").strip():
        os.environ["ASSISTANT_NUMBER"] = TEST_ASSISTANT_NUMBER
    if not (os.environ.get("ASSISTANT_WHATSAPP_NUMBER") or "").strip():
        os.environ["ASSISTANT_WHATSAPP_NUMBER"] = TEST_ASSISTANT_WHATSAPP_NUMBER
    os.environ["ASSISTANT_ID"] = ""
    os.environ["OWNER_TEAM_ID"] = ""
