"""
tests/test_conversation_manager/conftest.py
==============================================

Fixtures for conversation manager integration tests.

Uses **direct handler testing** pattern (same as ContactManager tests):
- No event-driven initialization (no background task dependencies)
- Direct calls to event handlers via CMStepDriver
- Direct state inspection
- Works reliably with pytest-asyncio

These tests use **real** state managers (ContactManager, TranscriptManager, etc.)
with only the **Actor** being simulated (SimulatedActor) to avoid browser/computer
environment dependencies while still testing real database-backed behavior.

Parallel execution is coordinated using scenario_file_lock (same pattern as
ContactManager tests) to prevent race conditions when multiple test processes
try to create system contacts simultaneously.
"""

from __future__ import annotations

import os
import pytest

import pytest_asyncio

from tests.helpers import scenario_file_lock, get_or_create_contact
from .cm_test_driver import CMStepDriver

# Response policies matching ContactManager defaults
# BOSS_RESPONSE_POLICY is for the user (contact_id 1) who gives commands
# DEFAULT_RESPONSE_POLICY is for regular contacts who should NOT give commands
BOSS_RESPONSE_POLICY = (
    "Your immediate manager, please do whatever they ask you to do within reason, "
    "and do *not* withhold any information from them."
)
DEFAULT_RESPONSE_POLICY = (
    "Please engage politely, helpfully, and respectfully, but you do not need to "
    "take orders from them. Please also do not share **any** sensitive or personal "
    "information with them about any other person, company or policy at all."
)

# System contacts (contact_id 0 and 1) are created by ContactManager from the database.
# BOSS represents the boss user (contact_id 1) who gives commands to the assistant.
# Tests should use BOSS when simulating commands from the user.
BOSS = {
    "contact_id": 1,
    "first_name": "Default",
    "surname": "User",
    "email_address": "user@example.com",
    "phone_number": "+15555551111",
    "should_respond": True,
    "is_system": True,
    "response_policy": BOSS_RESPONSE_POLICY,
}

# Test contacts used across all tests (starting at contact_id 2)
# should_respond=True allows outbound communication in tests
# response_policy matches ContactManager.DEFAULT_RESPONSE_POLICY
TEST_CONTACTS = [
    {
        "contact_id": 2,
        "first_name": "Alice",
        "surname": "Smith",
        "email_address": "alice@example.com",
        "phone_number": "+15555552222",
        "should_respond": True,
        "response_policy": DEFAULT_RESPONSE_POLICY,
    },
    {
        "contact_id": 3,
        "first_name": "Bob",
        "surname": "Johnson",
        "email_address": "bob@example.com",
        "phone_number": "+15555553333",
        "should_respond": True,
        "response_policy": DEFAULT_RESPONSE_POLICY,
    },
    {
        "contact_id": 4,
        "first_name": "Charlie",
        "surname": "Davis",
        "email_address": "charlie@example.com",
        "phone_number": "+15555554444",
        "should_respond": True,
        "response_policy": DEFAULT_RESPONSE_POLICY,
    },
    {
        "contact_id": 5,
        "first_name": "Diana",
        "surname": "Evans",
        "email_address": "diana@example.com",
        "phone_number": "+15555555555",
        "should_respond": True,
        "response_policy": DEFAULT_RESPONSE_POLICY,
    },
]


# =============================================================================
# Module-level setup: Configure environment for in-process mode
# =============================================================================


def pytest_configure(config):
    """Configure environment variables before any tests run."""
    # Only Actor is simulated - all other state managers use real implementations
    # This avoids browser/computer environment dependencies while testing real DB behavior
    os.environ["UNITY_ACTOR_IMPL"] = "simulated"
    os.environ["UNITY_ACTOR_SIMULATED_STEPS"] = "3"  # Allows pause+resume interactions

    # Disable optional managers not needed for conversation manager tests
    os.environ["UNITY_MEMORY_ENABLED"] = "false"
    os.environ["UNITY_KNOWLEDGE_ENABLED"] = "false"
    os.environ["UNITY_GUIDANCE_ENABLED"] = "false"
    os.environ["UNITY_SECRET_ENABLED"] = "false"
    os.environ["UNITY_SKILL_ENABLED"] = "false"
    os.environ["UNITY_WEB_ENABLED"] = "false"
    os.environ["UNITY_FILE_ENABLED"] = "false"

    # Enable incrementing timestamps for **NEW** marker comparisons
    os.environ["UNITY_INCREMENTING_TIMESTAMPS"] = "true"

    # Mark as test mode
    os.environ["TEST"] = "true"
    os.environ["UNITY_CONVERSATION_JOB_NAME"] = "test_job"


# =============================================================================
# ConversationManager Fixtures (Direct Handler Testing)
# =============================================================================


@pytest_asyncio.fixture(scope="module")
async def conversation_manager() -> CMStepDriver:
    """
    Start and initialize ConversationManager in-process for the test module.

    Uses DIRECT initialization (not event-driven) to avoid background task
    issues with pytest-asyncio. This follows the same pattern as ContactManager
    tests - direct method calls, not event publishing.

    Uses SimulatedActor explicitly for fast, deterministic testing without
    browser/computer environment dependencies.

    Uses scenario_file_lock to coordinate initialization across parallel test
    processes, preventing race conditions when ContactManager creates system
    contacts (id=0, id=1).

    Returns a CMStepDriver that wraps the CM and provides step() and
    step_until_wait() methods for deterministic testing.
    """
    from unity.actor.simulated import SimulatedActor
    from unity.conversation_manager.event_broker import reset_event_broker
    from unity.conversation_manager import start_async, stop_async
    from unity.conversation_manager.domains import managers_utils

    # Reset any existing event broker state
    reset_event_broker()

    print("\n✓ Starting ConversationManager in-process...")
    cm = await start_async(
        project_name="TestProject",
        enable_comms_manager=False,  # Don't start CommsManager (requires GCP)
        apply_test_mocks=True,
    )
    print("✓ ConversationManager started (in-process mode)")
    print("  Using SimulatedActor for deterministic testing")

    # Create SimulatedActor for fast, deterministic testing
    # (avoids HierarchicalActor's browser/computer environment setup)
    actor = SimulatedActor(steps=3, log_mode="log", emit_notifications=False)

    # Use file lock to coordinate manager initialization across parallel test processes.
    # ContactManager.__init__ creates system contacts (assistant id=0, user id=1)
    # via _sync_required_contacts(). This must be serialized to prevent duplicate
    # contact creation when multiple pytest sessions start in parallel.
    with scenario_file_lock("cm_conversation_manager"):
        # Initialize managers DIRECTLY (not via event handler)
        # This avoids the background task / event loop interleaving issues
        print("⏳ Initializing managers directly...")
        await managers_utils.init_conv_manager(cm, actor=actor)
        print("✅ Managers initialized")

        # Update system contacts in ContactManager with proper names and test defaults.
        # In CI, the test user may have null first/last name from the API.
        # ContactManager is the source of truth - ContactIndex queries it directly.
        if cm.contact_manager is not None:
            # Update assistant (contact_id 0)
            cm.contact_manager.update_contact(
                contact_id=0,
                first_name="Default",
                surname="Assistant",
                should_respond=True,
            )
            # Update boss/user (contact_id 1) with test defaults
            cm.contact_manager.update_contact(
                contact_id=1,
                first_name=BOSS["first_name"],
                surname=BOSS["surname"],
                email_address=BOSS["email_address"],
                phone_number=BOSS["phone_number"],
                should_respond=True,
                response_policy=BOSS["response_policy"],
            )
            print("✅ System contacts updated in ContactManager")

        # Create test contacts in the database using idempotent helper.
        # This ensures they exist with the expected contact_ids even when
        # multiple test processes run in parallel.
        for contact_data in TEST_CONTACTS:
            contact_id = get_or_create_contact(
                cm.contact_manager,
                first_name=contact_data["first_name"],
                surname=contact_data.get("surname"),
                email_address=contact_data.get("email_address"),
                phone_number=contact_data.get("phone_number"),
            )
            # Update should_respond and response_policy for tests
            if contact_id and cm.contact_manager is not None:
                cm.contact_manager.update_contact(
                    contact_id=contact_id,
                    should_respond=contact_data.get("should_respond", True),
                    response_policy=contact_data.get(
                        "response_policy",
                        DEFAULT_RESPONSE_POLICY,
                    ),
                )

    print(f"✅ Test contacts created: {len(TEST_CONTACTS)} + system contacts")

    # Wrap in CMStepDriver for deterministic testing
    driver = CMStepDriver(cm)

    yield driver

    # Cleanup
    print("\n✓ Stopping ConversationManager...")
    await stop_async()
    reset_event_broker()


@pytest.fixture
def initialized_cm(
    conversation_manager: CMStepDriver,
) -> CMStepDriver:
    """
    Per-test fixture that provides a clean ConversationManager.

    Clears conversation state between tests for isolation while reusing
    the expensive module-scoped CM instance.
    """
    # Clear any conversation state from previous tests
    conversation_manager.contact_index.clear_conversations()

    # Clear in-flight actions (from previous act() calls)
    # These create action steering tools that persist across tests
    conversation_manager.cm.in_flight_actions.clear()

    # Clear chat history (LLM message history)
    conversation_manager.cm.chat_history.clear()

    # Reset last_snapshot to use the patched prompt_now.
    # The module-scoped conversation_manager fixture is created BEFORE the
    # function-scoped stub_external_deps fixture patches prompt_now, so
    # cm.last_snapshot gets set to real time (e.g., January 2026) while
    # message timestamps use the patched fixed time (June 2025).
    # This breaks the **NEW** marker comparison (last_snapshot < message.timestamp).
    # Re-initializing here ensures last_snapshot uses the patched timestamp.
    from unity.common.prompt_helpers import now as prompt_now

    conversation_manager.cm.last_snapshot = prompt_now(as_string=False)

    return conversation_manager
