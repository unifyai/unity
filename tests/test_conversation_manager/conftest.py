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


# Test contacts used across all tests
# NOTE: contact_id 0 (assistant) and contact_id 1 (boss/user) are system contacts
# created automatically by ContactManager from the database. We do NOT define them
# here to avoid conflicts. Test contacts start at contact_id 2.
# should_respond=True allows outbound communication in tests
TEST_CONTACTS = [
    {
        "contact_id": 2,
        "first_name": "Alice",
        "surname": "Smith",
        "email_address": "alice@example.com",
        "phone_number": "+15555552222",
        "should_respond": True,
    },
    {
        "contact_id": 3,
        "first_name": "Bob",
        "surname": "Johnson",
        "email_address": "bob@example.com",
        "phone_number": "+15555553333",
        "should_respond": True,
    },
    {
        "contact_id": 4,
        "first_name": "Charlie",
        "surname": "Davis",
        "email_address": "charlie@example.com",
        "phone_number": "+15555554444",
        "should_respond": True,
    },
    {
        "contact_id": 5,
        "first_name": "Diana",
        "surname": "Evans",
        "email_address": "diana@example.com",
        "phone_number": "+15555555555",
        "should_respond": True,
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
    actor = SimulatedActor(steps=3, log_mode="log")

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

        # Fetch system contacts (contact_id 0 and 1) from ContactManager and add to local cache.
        # ContactManager creates these during initialization, but they're only in the database.
        # The brain.py code expects boss_contact (contact_id 1) to be in the local cache.
        if cm.contact_manager is not None:
            system_contact_ids = [0, 1]
            system_contacts_data = cm.contact_manager.get_contact_info(
                system_contact_ids,
            )
            for cid, contact_data in system_contacts_data.items():
                # Mark system contacts as should_respond=True for tests
                contact_data["should_respond"] = True
                cm.contact_index.set_contacts([contact_data])
            print(
                f"✅ System contacts synced to local cache: {list(system_contacts_data.keys())}",
            )

        # Create test contacts in the database using idempotent helper.
        # This ensures they exist with the expected contact_ids even when
        # multiple test processes run in parallel.
        for contact_data in TEST_CONTACTS:
            get_or_create_contact(
                cm.contact_manager,
                first_name=contact_data["first_name"],
                surname=contact_data.get("surname"),
                email_address=contact_data.get("email_address"),
                phone_number=contact_data.get("phone_number"),
            )

    # Set test contacts on contact_index local cache (includes should_respond=True)
    cm.contact_index.set_contacts(TEST_CONTACTS)
    print(f"✅ Test contacts set: {len(TEST_CONTACTS)}")

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

    # Clear active tasks (from previous act() calls)
    # These create task steering tools that persist across tests
    conversation_manager.cm.active_tasks.clear()

    # Clear chat history (LLM message history)
    conversation_manager.cm.chat_history.clear()

    return conversation_manager
