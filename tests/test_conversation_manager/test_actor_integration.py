"""
tests/test_conversation_manager/test_actor_integration.py
=========================================================

Tests for Actor integration with ConversationManager.

These tests verify that the Actor is properly initialized and accessible
when the ConversationManager starts up. Unlike eval tests that depend on
LLM behavior, these are symbolic tests that check infrastructure correctness.
"""

import pytest


@pytest.mark.asyncio
async def test_actor_initialized(conversation_manager):
    """
    Test that the Actor is properly initialized after ConversationManager startup.

    The Actor should not be None after initialization completes. If it is None,
    it indicates a failure during Actor initialization (e.g., tool resolution
    errors in the HierarchicalActor constructor).
    """
    # Access the underlying ConversationManager's actor attribute
    actor = conversation_manager.cm.actor

    assert actor is not None, (
        "Actor was not initialized. This typically indicates a tool resolution "
        "error during HierarchicalActor construction. Check the test output for "
        "'Error initializing Actor' messages."
    )
