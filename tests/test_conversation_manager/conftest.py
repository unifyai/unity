"""
tests/test_conversation_manager/conftest.py
==============================================

Fixtures for conversation manager integration tests.

Uses **direct handler testing** pattern (same as ContactManager tests):
- No event-driven initialization (no background task dependencies)
- Direct calls to event handlers via CMStepDriver
- Direct state inspection
- Works reliably with pytest-asyncio

These tests use simulated implementations for managers to ensure:
- Fast, deterministic execution
- No database state conflicts between parallel test sessions
- Isolation from production data

Note: Tests requiring REAL ContactManager integration (e.g., contact data
freshness) are in tests/test_contact_manager/test_contact_index_freshness.py
"""

from __future__ import annotations

import os
import pytest

import pytest_asyncio

from .cm_test_driver import CMStepDriver


# Test contacts used across all tests
# contact_id 0 = assistant, contact_id 1 = boss (main user)
# should_respond=True allows outbound communication in tests
TEST_CONTACTS = [
    {
        "contact_id": 0,
        "first_name": "Test",
        "surname": "Assistant",
        "email_address": "assistant@test.com",
        "phone_number": "+15555551234",
        "should_respond": True,
    },
    {
        "contact_id": 1,
        "first_name": "Test",
        "surname": "Contact",
        "email_address": "test@contact.com",
        "phone_number": "+15555551111",
        "should_respond": True,
    },
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
]


# =============================================================================
# Module-level setup: Configure environment for in-process mode
# =============================================================================


def pytest_configure(config):
    """Configure environment variables before any tests run."""
    # Use simulated implementations for fast, isolated testing
    os.environ["UNITY_ACTOR_IMPL"] = "simulated"
    os.environ["UNITY_CONTACT_IMPL"] = "simulated"
    os.environ["UNITY_TRANSCRIPT_IMPL"] = "simulated"
    os.environ["UNITY_TASK_IMPL"] = "simulated"
    os.environ["UNITY_CONVERSATION_IMPL"] = "simulated"

    # Steps for SimulatedActor - 3 allows for pause+resume interactions
    os.environ["UNITY_ACTOR_SIMULATED_STEPS"] = "3"

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

    # Initialize managers DIRECTLY (not via event handler)
    # This avoids the background task / event loop interleaving issues
    print("⏳ Initializing managers directly...")
    await managers_utils.init_conv_manager(cm, actor=actor)
    print("✅ Managers initialized")

    # Set test contacts on contact_index (includes should_respond=True)
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
