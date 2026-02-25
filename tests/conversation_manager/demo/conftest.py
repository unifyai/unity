"""
tests/conversation_manager/demo/conftest.py
=============================================

Fixtures for demo mode conversation manager tests.

In demo mode:
- The boss contact (contact_id=1) starts empty — their details are unknown
  and will be learned during the demo conversation.
- The demo operator (contact_id=2) is a Unify colleague who introduces the
  prospect to the assistant.
- ``act`` is masked from the slow brain; ``set_boss_details`` is exposed.
- contact_id=1 is always present in active_conversations.

These fixtures override the parent conftest's ``conversation_manager`` and
``initialized_cm`` to set up the demo-specific environment.
"""

from __future__ import annotations

import os
import pytest
import pytest_asyncio

from tests.helpers import scenario_file_lock
from tests.conversation_manager.cm_test_driver import CMStepDriver

# ─────────────────────────────────────────────────────────────────────────────
# Contact definitions for demo tests
# ─────────────────────────────────────────────────────────────────────────────

# In demo mode, the boss (contact_id=1) starts with no details.
# Details are populated during the demo via set_boss_details and inline comms.
DEMO_BOSS_INITIAL = {
    "contact_id": 1,
    "first_name": None,
    "surname": None,
    "email_address": None,
    "phone_number": None,
    "should_respond": True,
    "is_system": True,
}

# The demo operator is a Unify team member (e.g., Daniel) who introduces the
# prospect to the assistant.
DEMO_OPERATOR = {
    "contact_id": 2,
    "first_name": "Daniel",
    "surname": "Lenton",
    "email_address": "daniel@unify.ai",
    "phone_number": "+15555559999",
    "should_respond": True,
    "is_system": True,
    "response_policy": (
        "This is a Unify team member who provisioned the demo. "
        "They may introduce you to your future boss."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Module-level setup
# ─────────────────────────────────────────────────────────────────────────────


def pytest_configure(config):
    """Enable demo mode before any tests in this folder run.

    Sets the env var for any future SETTINGS re-creation AND directly mutates
    the existing unity.settings.SETTINGS singleton (which was already created
    eagerly during root conftest imports, before this hook ran).

    Also populates SESSION_DETAILS.user with the demoer's details so that
    init_conv_manager creates contact_id=2 with correct name/phone/email
    (mirroring production where SESSION_DETAILS.user is populated from
    the Orchestra assistant record before initialization).
    """
    os.environ["DEMO_MODE"] = "true"
    from unity.settings import SETTINGS

    SETTINGS.DEMO_MODE = True

    from unity.session_details import SESSION_DETAILS

    SESSION_DETAILS.user.first_name = DEMO_OPERATOR["first_name"]
    SESSION_DETAILS.user.surname = DEMO_OPERATOR["surname"]
    SESSION_DETAILS.user.number = DEMO_OPERATOR["phone_number"]
    SESSION_DETAILS.user.email = DEMO_OPERATOR["email_address"]


def pytest_unconfigure(config):
    """Clean up demo mode env var and restore SETTINGS."""
    os.environ.pop("DEMO_MODE", None)
    from unity.settings import SETTINGS

    SETTINGS.DEMO_MODE = False

    from unity.session_details import SESSION_DETAILS

    SESSION_DETAILS.user.first_name = ""
    SESSION_DETAILS.user.surname = ""
    SESSION_DETAILS.user.number = ""
    SESSION_DETAILS.user.email = ""


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture(scope="module")
async def conversation_manager(request) -> CMStepDriver:
    """Start ConversationManager in demo mode for the test module.

    Similar to the parent conftest fixture, but:
    - DEMO_MODE is active (set via pytest_configure above)
    - Boss contact (contact_id=1) is left sparse (no name, no email, no phone)
    - Demo operator (contact_id=2) is seeded
    """
    from unity.actor.simulated import SimulatedActor
    from unity.conversation_manager.event_broker import reset_event_broker
    from unity.conversation_manager import start_async, stop_async
    from unity.conversation_manager.domains import managers_utils
    from unity.settings import SETTINGS

    assert (
        SETTINGS.DEMO_MODE
    ), "DEMO_MODE should be True — check pytest_configure in demo/conftest.py"

    reset_event_broker()

    print("\n✓ Starting ConversationManager in DEMO MODE...")
    cm = await start_async(
        project_name="TestProject",
        enable_comms_manager=False,
        apply_test_mocks=True,
    )

    actor = SimulatedActor(
        steps=None,
        duration=None,
        log_mode="log",
        emit_notifications=False,
    )

    with scenario_file_lock("cm_demo_mode"):
        await managers_utils.init_conv_manager(cm, actor=actor)
        if not cm.initialized:
            raise RuntimeError("ConversationManager failed to initialize in demo mode")
        print("✅ Managers initialized (demo mode)")

        if cm.contact_manager is not None:
            # Update assistant contact (contact_id=0) — same as parent
            cm.contact_manager.update_contact(
                contact_id=0,
                first_name="Lucy",
                surname="Demo",
                should_respond=True,
            )

            # Boss (contact_id=1) — verify it was created sparse by demo provisioning.
            # Do NOT set name/email/phone — that's the whole point of demo mode.
            # Only ensure should_respond=True so comms tools work.
            cm.contact_manager.update_contact(
                contact_id=1,
                should_respond=True,
            )
            print("✅ Boss contact left sparse (demo mode)")

            # Demo operator (contact_id=2) is created by init_conv_manager
            # from SESSION_DETAILS.user (populated in pytest_configure).
            # Just apply response_policy which isn't part of SESSION_DETAILS.
            if DEMO_OPERATOR.get("response_policy"):
                cm.contact_manager.update_contact(
                    contact_id=DEMO_OPERATOR["contact_id"],
                    response_policy=DEMO_OPERATOR["response_policy"],
                )
            print(f"✅ Demo operator seeded: {DEMO_OPERATOR['first_name']}")

    driver = CMStepDriver(cm)
    yield driver

    print("\n✓ Stopping ConversationManager (demo mode)...")
    await stop_async()
    reset_event_broker()


def _complete_in_flight_actions(cm: CMStepDriver) -> None:
    """Complete all in-flight actions to unblock watcher threads."""
    for handle_data in list(cm.cm.in_flight_actions.values()):
        handle = handle_data.get("handle")
        if handle and hasattr(handle, "trigger_completion"):
            handle.trigger_completion()
    cm.cm.in_flight_actions.clear()
    cm.cm.completed_actions.clear()


@pytest.fixture
def initialized_cm(conversation_manager: CMStepDriver):
    """Per-test fixture providing a clean demo-mode CM."""
    _complete_in_flight_actions(conversation_manager)
    conversation_manager.contact_index.clear_conversations()

    import unity.conversation_manager.domains.brain_action_tools as bat

    bat._next_handle_id = 0

    conversation_manager.cm.chat_history.clear()
    conversation_manager.all_tool_calls.clear()

    from unity.common.prompt_helpers import now as prompt_now

    conversation_manager.cm.last_snapshot = prompt_now(as_string=False)

    # Re-inject contact_id=1 into active_conversations (cleared above).
    # In production this happens during init; here we re-apply after clear.
    conversation_manager.contact_index.get_or_create_conversation(1)

    yield conversation_manager

    _complete_in_flight_actions(conversation_manager)
