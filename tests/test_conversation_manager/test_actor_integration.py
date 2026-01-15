"""
tests/test_conversation_manager/test_actor_integration.py
=========================================================

Tests for Actor integration with ConversationManager.

These tests verify that the Actor is properly initialized and accessible
when the ConversationManager starts up. The fixture explicitly uses
SimulatedActor for fast, deterministic testing without browser/computer
environment dependencies.

Unlike eval tests that depend on LLM behavior, these are symbolic tests
that check infrastructure correctness.
"""

import pytest

from unity.actor.simulated import SimulatedActor


@pytest.mark.asyncio
async def test_actor_initialized(conversation_manager):
    """
    Test that the Actor is properly initialized after ConversationManager startup.

    The Actor should be a SimulatedActor instance (injected via the fixture).
    """
    actor = conversation_manager.cm.actor

    assert actor is not None, "Actor was not initialized"
    assert isinstance(actor, SimulatedActor), (
        f"Expected SimulatedActor but got {type(actor).__name__}. "
        "The fixture should inject SimulatedActor explicitly."
    )
