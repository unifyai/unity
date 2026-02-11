"""
tests/conversation_manager/demo/test_demo_mode_setup.py
========================================================

Symbolic tests verifying the demo mode infrastructure:
- Tool surface (act masked, set_boss_details exposed)
- Boss contact starts sparse
- set_boss_details updates contact_id=1 correctly
- contact_id=1 is always in active_conversations
- Communication tools work with inline details for contact_id=1
"""

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.demo.conftest import (
    DEMO_OPERATOR,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tool surface tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_act_is_masked_in_demo_mode(initialized_cm):
    """In demo mode, 'act' should not be in the tool surface."""
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    action_tools = ConversationManagerBrainActionTools(initialized_cm.cm)
    tools = action_tools.as_tools()

    assert "act" not in tools, "act should be masked in demo mode"
    assert (
        "ask_about_contacts" not in tools
    ), "ask_about_contacts should be masked in demo mode"
    assert (
        "update_contacts" not in tools
    ), "update_contacts should be masked in demo mode"
    assert (
        "set_boss_details" in tools
    ), "set_boss_details should be exposed in demo mode"


@pytest.mark.asyncio
@_handle_project
async def test_communication_tools_still_available(initialized_cm):
    """Communication tools (send_sms, send_email, make_call) should remain available."""
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    action_tools = ConversationManagerBrainActionTools(initialized_cm.cm)
    tools = action_tools.as_tools()

    for tool_name in (
        "send_sms",
        "send_email",
        "make_call",
        "send_unify_message",
        "wait",
    ):
        assert tool_name in tools, f"{tool_name} should be available in demo mode"


# ─────────────────────────────────────────────────────────────────────────────
# Boss contact provisioning tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_boss_contact_starts_sparse(initialized_cm):
    """Boss (contact_id=1) should start without name, email, or phone in demo mode."""
    boss = initialized_cm.contact_index.get_contact(1)
    assert boss is not None, "Boss contact should exist"

    # In demo mode, boss details should be empty/None
    # (they get populated during the demo conversation)
    assert not boss.get(
        "first_name",
    ), f"Boss first_name should be empty, got: {boss.get('first_name')!r}"
    assert not boss.get(
        "email_address",
    ), f"Boss email should be empty, got: {boss.get('email_address')!r}"


@pytest.mark.asyncio
@_handle_project
async def test_boss_always_in_active_conversations(initialized_cm):
    """contact_id=1 should be in active_conversations even before any messages."""
    assert (
        1 in initialized_cm.contact_index.active_conversations
    ), "Boss (contact_id=1) should be in active_conversations in demo mode"


# ─────────────────────────────────────────────────────────────────────────────
# set_boss_details tool tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_set_boss_details_updates_name(initialized_cm):
    """set_boss_details should update the boss contact's name."""
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    action_tools = ConversationManagerBrainActionTools(initialized_cm.cm)
    result = await action_tools.set_boss_details(
        first_name="Richard",
        surname="Branson",
    )

    assert result["status"] == "updated"
    assert result["updates"] == {"first_name": "Richard", "surname": "Branson"}

    # Verify the contact was actually updated
    boss = initialized_cm.contact_index.get_contact(1)
    assert boss["first_name"] == "Richard"
    assert boss["surname"] == "Branson"


@pytest.mark.asyncio
@_handle_project
async def test_set_boss_details_updates_email(initialized_cm):
    """set_boss_details should update the boss contact's email address."""
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    action_tools = ConversationManagerBrainActionTools(initialized_cm.cm)
    result = await action_tools.set_boss_details(
        email_address="richard@example.com",
    )

    assert result["status"] == "updated"

    boss = initialized_cm.contact_index.get_contact(1)
    assert boss["email_address"] == "richard@example.com"


@pytest.mark.asyncio
@_handle_project
async def test_set_boss_details_updates_phone(initialized_cm):
    """set_boss_details should update the boss contact's phone number."""
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    action_tools = ConversationManagerBrainActionTools(initialized_cm.cm)
    result = await action_tools.set_boss_details(
        phone_number="+447700900000",
    )

    assert result["status"] == "updated"

    boss = initialized_cm.contact_index.get_contact(1)
    assert boss["phone_number"] == "+447700900000"


@pytest.mark.asyncio
@_handle_project
async def test_set_boss_details_no_fields_returns_error(initialized_cm):
    """set_boss_details with no fields should return an error."""
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    action_tools = ConversationManagerBrainActionTools(initialized_cm.cm)
    result = await action_tools.set_boss_details()

    assert result["status"] == "error"


@pytest.mark.asyncio
@_handle_project
async def test_set_boss_details_partial_update(initialized_cm):
    """set_boss_details should only update provided fields, leaving others unchanged."""
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    action_tools = ConversationManagerBrainActionTools(initialized_cm.cm)

    # First update: set name
    await action_tools.set_boss_details(first_name="Richard")

    # Second update: set email (name should remain)
    await action_tools.set_boss_details(email_address="richard@example.com")

    boss = initialized_cm.contact_index.get_contact(1)
    assert boss["first_name"] == "Richard"
    assert boss["email_address"] == "richard@example.com"


# ─────────────────────────────────────────────────────────────────────────────
# Demo operator tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_demo_operator_exists(initialized_cm):
    """The demo operator (contact_id=2) should be seeded with full details."""
    operator = initialized_cm.contact_index.get_contact(
        DEMO_OPERATOR["contact_id"],
    )
    assert operator is not None, "Demo operator should exist"
    assert operator["first_name"] == DEMO_OPERATOR["first_name"]
    assert operator["phone_number"] == DEMO_OPERATOR["phone_number"]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt content tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_slow_brain_prompt_contains_demo_guidance(initialized_cm):
    """The slow brain system prompt should contain demo-specific guidance."""
    from unity.conversation_manager.domains.brain import build_brain_spec

    spec = build_brain_spec(initialized_cm.cm)
    prompt_text = spec.system_prompt.flatten()

    assert (
        "demo mode" in prompt_text.lower()
    ), "Slow brain prompt should mention demo mode"
    assert (
        "set_boss_details" in prompt_text
    ), "Slow brain prompt should mention set_boss_details tool"
    assert (
        "unify.ai" in prompt_text.lower()
    ), "Slow brain prompt should mention unify.ai for sign-up"
    # act should NOT be mentioned in the tool list
    assert (
        "act` freely" not in prompt_text
    ), "Slow brain prompt should not encourage using act in demo mode"


@pytest.mark.asyncio
@_handle_project
async def test_voice_prompt_contains_demo_guidance(initialized_cm):
    """The voice agent prompt should contain demo-specific guidance when in demo mode."""
    from unity.conversation_manager.prompt_builders import build_voice_agent_prompt

    prompt = build_voice_agent_prompt(
        bio="A helpful assistant",
        assistant_name="Lucy",
        boss_first_name="",
        boss_surname="",
        demo_mode=True,
    )
    text = prompt.flatten()

    assert "demo" in text.lower(), "Voice prompt should reference demo context"
    assert "unify.ai" in text.lower(), "Voice prompt should mention unify.ai"
    assert (
        "first time" in text.lower() or "not signed up" in text.lower()
    ), "Voice prompt should indicate boss hasn't signed up"
