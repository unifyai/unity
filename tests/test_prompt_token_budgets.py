"""Token-budget ratchet over the fixed per-LLM-call prompt payload.

Every LLM call pays its system prompt and serialized tool schemas before a
single trajectory token. These budgets pin the currently rendered sizes so
that growth is a deliberate decision — raise the constant in the same change
that grows the payload, and say why — rather than an accident, and so that
every landed cut is banked by tightening the constant to the new size.
"""

from __future__ import annotations

import json

import pytest

from unify.common.token_utils import count_tokens
from unify.manager_registry import ManagerRegistry

pytestmark = pytest.mark.no_unify_context

# Budgets sit just above the measured rendered size at the last tightening.
ACTOR_SYSTEM_PROMPT_BUDGET = 10_400
ACTOR_ACT_TOOL_SCHEMAS_BUDGET = 14_500
STORAGE_REVIEW_DOCTRINE_BUDGET = 4_700
CM_SYSTEM_PROMPT_BUDGET = 11_100
VOICE_AGENT_PROMPT_BUDGET = 2_800


@pytest.fixture(autouse=True)
def _clean_registry():
    ManagerRegistry.clear()
    yield
    ManagerRegistry.clear()


def _simulated_actor():
    from unify.actor.code_act_actor import CodeActActor
    from unify.function_manager.simulated import SimulatedFunctionManager
    from unify.guidance_manager.simulated import SimulatedGuidanceManager
    from unify.knowledge_manager.simulated import SimulatedKnowledgeManager

    return CodeActActor(
        function_manager=SimulatedFunctionManager(description="token budget"),
        guidance_manager=SimulatedGuidanceManager(description="token budget"),
        knowledge_manager=SimulatedKnowledgeManager(description="token budget"),
    )


def _actor_system_prompt() -> str:
    from unify.actor.environments.base import _CompositeEnvironment
    from unify.actor.environments.computer import ComputerEnvironment
    from unify.actor.environments.state_managers import StateManagerEnvironment
    from unify.actor.prompt_builders import build_code_act_prompt
    from unify.function_manager.primitives import ComputerPrimitives

    composite = _CompositeEnvironment(
        [
            ComputerEnvironment(ComputerPrimitives(computer_mode="mock")),
            StateManagerEnvironment(),
        ],
    )
    return build_code_act_prompt(
        environments={"primitives": composite},
        tools=dict(_simulated_actor().get_tools("act")),
        can_store=True,
        discovery_first_policy=True,
        # The OAuth section reads live workspace connections; the payload
        # under ratchet is the connection-independent render.
        include_oauth_helper=False,
    )


def _actor_act_tool_schemas() -> str:
    from unify.common.llm_helpers import method_to_schema

    return json.dumps(
        [
            method_to_schema(getattr(tool, "fn", tool), name)
            for name, tool in _simulated_actor().get_tools("act").items()
        ],
    )


def _storage_review_doctrine() -> str:
    # The static prefix every skill-librarian loop pays, in prompt order.
    from unify.actor.code_act_actor import (
        _STORAGE_BASE_INSTRUCTIONS,
        _STORAGE_RECURRING_DELIVERABLE,
        _STORAGE_SUB_AGENT_PATTERNS,
        _STORAGE_THREE_STORES,
        _STORAGE_WHAT_CAN_BE_STORED,
    )

    return "".join(
        [
            _STORAGE_WHAT_CAN_BE_STORED,
            _STORAGE_THREE_STORES,
            _STORAGE_SUB_AGENT_PATTERNS,
            _STORAGE_RECURRING_DELIVERABLE,
            _STORAGE_BASE_INSTRUCTIONS,
        ],
    )


def _cm_system_prompt() -> str:
    from unify.conversation_manager.prompt_builders import build_system_prompt

    return build_system_prompt(
        bio="A helpful assistant.",
        contact_id=1,
        first_name="Alice",
        surname="Smith",
        assistant_has_phone=True,
        assistant_has_email=True,
    ).flatten()


def _voice_agent_prompt() -> str:
    from unify.conversation_manager.prompt_builders import build_voice_agent_prompt

    return build_voice_agent_prompt(
        bio="I help Acme configure its Unify team.",
        assistant_name="Avery",
        boss_first_name="Dana",
        boss_surname="Owner",
    ).flatten()


_CASES = {
    "actor_system_prompt": (_actor_system_prompt, ACTOR_SYSTEM_PROMPT_BUDGET),
    "actor_act_tool_schemas": (
        _actor_act_tool_schemas,
        ACTOR_ACT_TOOL_SCHEMAS_BUDGET,
    ),
    "storage_review_doctrine": (
        _storage_review_doctrine,
        STORAGE_REVIEW_DOCTRINE_BUDGET,
    ),
    "cm_system_prompt": (_cm_system_prompt, CM_SYSTEM_PROMPT_BUDGET),
    "voice_agent_prompt": (_voice_agent_prompt, VOICE_AGENT_PROMPT_BUDGET),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_fixed_prompt_payload_stays_within_budget(case):
    render, budget = _CASES[case]
    tokens = count_tokens(render())
    assert tokens < budget, (
        f"{case} renders {tokens:,} tokens against a budget of {budget:,}. "
        "If the growth is deliberate, raise the budget constant in this file "
        "in the same change and say why; otherwise trim the payload."
    )
