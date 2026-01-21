"""
tests/test_conversation_manager/test_tool_docstrings.py
========================================================

Tests for brain tool docstring quality and schema stability.

Follows the gold standard pattern from test_contact_manager/test_tool_docstrings.py
and test_transcript_manager/test_tool_docstrings.py.

The ConversationManager uses two sets of brain tools:
- ConversationManagerBrainTools: Read-only state inspection tools
- ConversationManagerBrainActionTools: Communication and action management tools

Dynamic steering tools (generated based on in-flight actions) are not tested here
since they are created at runtime with generated docstrings.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from tests.assertion_helpers import first_diff_block

# Tests in this file are symbolic (deterministic, no LLM)
pytestmark = pytest.mark.symbolic


def _unwrap_callable(tool):
    """Return the underlying callable from either a ToolSpec or a function."""
    return getattr(tool, "fn", tool)


# =============================================================================
# Docstring Quality Tests
# =============================================================================


def test_brain_tools_docstrings(initialized_cm):
    """
    Test that ConversationManagerBrainTools have sufficient docstrings.

    These are read-only tools for state inspection:
    - cm_get_mode
    - cm_get_contact
    - cm_list_in_flight_actions
    - cm_list_notifications
    """
    from unity.conversation_manager.domains.brain_tools import (
        ConversationManagerBrainTools,
    )

    brain_tools = ConversationManagerBrainTools(initialized_cm.cm)
    tools = brain_tools.as_tools()

    assert tools, "ConversationManagerBrainTools should expose at least one tool"

    for name, fn in tools.items():
        fn = _unwrap_callable(fn)
        doc = (getattr(fn, "__doc__", None) or "").strip()
        assert doc, f"Brain tool '{name}' is missing a docstring"
        # Brain tools can have shorter docstrings since they are simpler
        # Use a lower threshold (50 chars) for these utility tools
        assert (
            len(doc) >= 50
        ), f"Docstring for brain tool '{name}' is too short (len={len(doc)})"


def test_brain_action_tools_docstrings(initialized_cm):
    """
    Test that ConversationManagerBrainActionTools have sufficient docstrings.

    These are side-effecting tools:
    - send_sms
    - send_unify_message
    - send_email
    - make_call
    - act
    - wait
    """
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    action_tools = ConversationManagerBrainActionTools(initialized_cm.cm)
    tools = action_tools.as_tools()

    assert tools, "ConversationManagerBrainActionTools should expose at least one tool"

    for name, fn in tools.items():
        fn = _unwrap_callable(fn)
        doc = (getattr(fn, "__doc__", None) or "").strip()
        assert doc, f"Action tool '{name}' is missing a docstring"
        # Action tools should have detailed docstrings (>= 100 chars)
        assert (
            len(doc) >= 100
        ), f"Docstring for action tool '{name}' is too short (len={len(doc)})"


def test_all_brain_tools_have_docstrings(initialized_cm):
    """
    Combined test ensuring all brain tools (read + action) have docstrings.
    """
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )
    from unity.conversation_manager.domains.brain_tools import (
        ConversationManagerBrainTools,
    )

    brain_tools = ConversationManagerBrainTools(initialized_cm.cm)
    action_tools = ConversationManagerBrainActionTools(initialized_cm.cm)

    all_tools = {**brain_tools.as_tools(), **action_tools.as_tools()}

    missing_docs = []
    for name, fn in all_tools.items():
        fn = _unwrap_callable(fn)
        doc = (getattr(fn, "__doc__", None) or "").strip()
        if not doc:
            missing_docs.append(name)

    assert not missing_docs, f"Tools missing docstrings: {missing_docs}"


# =============================================================================
# Schema Stability Tests
# =============================================================================


def _build_brain_tools_schema_in_subprocess(tool_class: str) -> str:
    """
    Build tools→schema JSON in a fresh Python process to catch cross-session drift.

    Args:
        tool_class: Either "brain_tools" or "brain_action_tools"
    """
    assert tool_class in {"brain_tools", "brain_action_tools"}

    if tool_class == "brain_tools":
        import_stmt = """from unity.conversation_manager.domains.brain_tools import ConversationManagerBrainTools"""
        instantiate = "ConversationManagerBrainTools(cm)"
    else:
        import_stmt = """from unity.conversation_manager.domains.brain_action_tools import ConversationManagerBrainActionTools"""
        instantiate = "ConversationManagerBrainActionTools(cm)"

    code = textwrap.dedent(
        f"""
        import os, sys, json
        sys.path.insert(0, os.getcwd())

        # Only Actor is simulated - avoids browser/computer environment dependencies
        os.environ["UNITY_ACTOR_IMPL"] = "simulated"

        from unity.common.llm_helpers import method_to_schema

        def _unwrap_callable(tool):
            return getattr(tool, "fn", tool)

        # Create a minimal mock CM for tool instantiation
        class MockCM:
            mode = "text"
            contact_index = None
            notifications_bar = None
            in_flight_actions = {{}}
            contact_manager = None
            assistant_number = None
            assistant_email = None
            actor = None
            chat_history = []

        class MockNotificationsBar:
            notifications = []

        cm = MockCM()
        cm.notifications_bar = MockNotificationsBar()

        {import_stmt}
        tools_instance = {instantiate}
        tools = tools_instance.as_tools()

        if not tools:
            raise AssertionError("Should expose at least one tool")

        mapping = {{
            name: method_to_schema(_unwrap_callable(fn), name)
            for name, fn in tools.items()
        }}
        sys.stdout.write(json.dumps(mapping, sort_keys=True, indent=2))
        """,
    )

    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
        env=env,
    )
    return proc.stdout


def test_brain_tools_schemas_stable():
    """
    Test that brain tool schemas are stable across Python sessions.

    Instability could indicate non-deterministic schema generation.
    """
    p1 = _build_brain_tools_schema_in_subprocess("brain_tools")
    p2 = _build_brain_tools_schema_in_subprocess("brain_tools")

    if p1 != p2:
        snippet = first_diff_block(
            p1,
            p2,
            context=3,
            label_a="First JSON",
            label_b="Second JSON",
        )
        raise AssertionError(
            "Tool schemas for brain_tools changed between Python sessions.\n\n"
            + snippet,
        )


def test_brain_action_tools_schemas_stable():
    """
    Test that brain action tool schemas are stable across Python sessions.

    Instability could indicate non-deterministic schema generation.
    """
    p1 = _build_brain_tools_schema_in_subprocess("brain_action_tools")
    p2 = _build_brain_tools_schema_in_subprocess("brain_action_tools")

    if p1 != p2:
        snippet = first_diff_block(
            p1,
            p2,
            context=3,
            label_a="First JSON",
            label_b="Second JSON",
        )
        raise AssertionError(
            "Tool schemas for brain_action_tools changed between Python sessions.\n\n"
            + snippet,
        )


# =============================================================================
# Docstring Content Tests
# =============================================================================


def test_act_tool_has_comprehensive_docstring(initialized_cm):
    """
    Test that the 'act' tool has a comprehensive docstring.

    The 'act' tool is the central delegation mechanism and should have
    detailed documentation about its capabilities.
    """
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    action_tools = ConversationManagerBrainActionTools(initialized_cm.cm)
    tools = action_tools.as_tools()

    act_fn = _unwrap_callable(tools["act"])
    doc = (getattr(act_fn, "__doc__", None) or "").strip()

    # 'act' should have comprehensive documentation
    assert len(doc) >= 300, f"'act' docstring should be >= 300 chars, got {len(doc)}"

    # Should mention key capabilities
    doc_lower = doc.lower()
    assert "retrieval" in doc_lower, "'act' docstring should mention retrieval"
    assert "action" in doc_lower, "'act' docstring should mention action"
    assert (
        "contact" in doc_lower or "knowledge" in doc_lower
    ), "'act' docstring should mention contacts or knowledge"


def test_wait_tool_has_usage_guidance(initialized_cm):
    """
    Test that the 'wait' tool has clear usage guidance.

    The 'wait' tool is important for preventing over-communication.
    """
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    action_tools = ConversationManagerBrainActionTools(initialized_cm.cm)
    tools = action_tools.as_tools()

    wait_fn = _unwrap_callable(tools["wait"])
    doc = (getattr(wait_fn, "__doc__", None) or "").strip()

    # 'wait' should explain when to use it
    doc_lower = doc.lower()
    assert (
        "prefer" in doc_lower or "when" in doc_lower
    ), "'wait' docstring should explain when to use it"
