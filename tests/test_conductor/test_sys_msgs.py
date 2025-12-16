"""System message tests for Conductor prompt builders.

Conductor orchestrates all state managers and has complex tool routing.
These tests validate the prompt structure and stability using mock tools.
"""

import re
import sys
import subprocess
import textwrap

from tests.assertion_helpers import (
    extract_tools_dict,
    assert_in_order,
    assert_time_footer,
    first_diff_block,
)


from unity.conductor.prompt_builders import build_request_prompt


def _make_mock_conductor_tools() -> dict:
    """Create mock tool functions matching Conductor's expected tool set."""
    # ContactManager
    def ContactManager_ask(text: str) -> str:
        """Ask about contacts."""
        return ""

    def ContactManager_update(text: str) -> str:
        """Update contacts."""
        return ""

    # TranscriptManager
    def TranscriptManager_ask(text: str) -> str:
        """Ask about transcripts."""
        return ""

    # KnowledgeManager
    def KnowledgeManager_ask(text: str) -> str:
        """Ask about knowledge."""
        return ""

    def KnowledgeManager_update(text: str) -> str:
        """Update knowledge."""
        return ""

    # GuidanceManager
    def GuidanceManager_ask(text: str) -> str:
        """Ask about guidance."""
        return ""

    def GuidanceManager_update(text: str) -> str:
        """Update guidance."""
        return ""

    # TaskScheduler
    def TaskScheduler_ask(text: str) -> str:
        """Ask about tasks."""
        return ""

    def TaskScheduler_update(text: str) -> str:
        """Update tasks."""
        return ""

    def TaskScheduler_execute(text: str) -> str:
        """Execute tasks."""
        return ""

    # WebSearcher
    def WebSearcher_ask(text: str) -> str:
        """Web search."""
        return ""

    def WebSearcher_update(text: str) -> str:
        """Update web searcher config."""
        return ""

    # SecretManager
    def SecretManager_ask(text: str) -> str:
        """Ask about secrets."""
        return ""

    # Actor
    def Actor_act(description: str) -> str:
        """Execute an action."""
        return ""

    # GlobalFileManager
    def GlobalFileManager_ask(text: str) -> str:
        """Ask about files."""
        return ""

    def GlobalFileManager_organize(text: str) -> str:
        """Organize files."""
        return ""

    # ConversationManagerHandle
    def ConversationManagerHandle_ask(question: str) -> str:
        """Ask via conversation."""
        return ""

    def ConversationManagerHandle_interject(message: str) -> str:
        """Interject in conversation."""
        return ""

    def ConversationManagerHandle_get_full_transcript() -> str:
        """Get full transcript."""
        return ""

    return {
        "ContactManager_ask": ContactManager_ask,
        "ContactManager_update": ContactManager_update,
        "TranscriptManager_ask": TranscriptManager_ask,
        "KnowledgeManager_ask": KnowledgeManager_ask,
        "KnowledgeManager_update": KnowledgeManager_update,
        "GuidanceManager_ask": GuidanceManager_ask,
        "GuidanceManager_update": GuidanceManager_update,
        "TaskScheduler_ask": TaskScheduler_ask,
        "TaskScheduler_update": TaskScheduler_update,
        "TaskScheduler_execute": TaskScheduler_execute,
        "WebSearcher_ask": WebSearcher_ask,
        "WebSearcher_update": WebSearcher_update,
        "SecretManager_ask": SecretManager_ask,
        "Actor_act": Actor_act,
        "GlobalFileManager_ask": GlobalFileManager_ask,
        "GlobalFileManager_organize": GlobalFileManager_organize,
        "ConversationManagerHandle_ask": ConversationManagerHandle_ask,
        "ConversationManagerHandle_interject": ConversationManagerHandle_interject,
        "ConversationManagerHandle_get_full_transcript": ConversationManagerHandle_get_full_transcript,
    }


def test_request_system_prompt_formatting():
    tools = _make_mock_conductor_tools()
    prompt = build_request_prompt(tools=tools)

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Tools (name" in prompt

    # Conductor-specific sections - core orchestration guidance
    assert "read-write control" in prompt
    assert "task" in prompt.lower()
    assert "contact" in prompt.lower()
    assert "knowledge" in prompt.lower()

    # Task execution policy
    assert "execute" in prompt.lower()

    # Decomposition and concurrency
    assert "Decompose and parallelize" in prompt

    # Update philosophy
    assert "Update tools" in prompt

    # Examples
    assert "Examples" in prompt

    # Time footer
    assert_time_footer(prompt, "Current UTC time is ")

    # Ordering checks
    assert_in_order(
        prompt,
        [
            "read-write control",
            "Decompose and parallelize",
            "Update tools",
            "Tools (name",
            "Examples",
            "Current UTC time",
        ],
    )

    print(
        "Conductor request system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n\n\n"
        + prompt[:3000],
    )


def test_request_prompt_has_key_tool_references():
    """Verify that the prompt references expected manager tools."""
    tools = _make_mock_conductor_tools()
    prompt = build_request_prompt(tools=tools)

    # Core managers should be referenced
    expected_managers = [
        "ContactManager",
        "TranscriptManager",
        "TaskScheduler",
        "KnowledgeManager",
    ]

    for manager in expected_managers:
        assert manager.lower() in prompt.lower(), f"Expected reference to {manager}"


def test_request_prompt_clarification_handling():
    """Verify clarification guidance is present."""
    tools = _make_mock_conductor_tools()
    prompt = build_request_prompt(tools=tools)

    # Should have clarification guidance (either tool-based or default policy)
    has_clarification = (
        "clarification" in prompt.lower()
        or "ambiguous" in prompt.lower()
        or "Do not ask the user questions" in prompt
    )
    assert has_clarification, "Expected clarification guidance in prompt"


def test_actor_act_guidance():
    """Verify Actor.act routing guidance when available."""
    tools = _make_mock_conductor_tools()
    prompt = build_request_prompt(tools=tools)

    # Should have Actor execution guidance
    assert "Actor_act" in prompt or "actor" in prompt.lower()


def test_conversation_steering_guidance():
    """Verify conversation steering guidance is present."""
    tools = _make_mock_conductor_tools()
    prompt = build_request_prompt(tools=tools)

    # Should have conversation steering guidance
    assert "Steering the Live Conversation" in prompt or "transcript" in prompt.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Stability: prompts should be identical across serial builder calls
# ─────────────────────────────────────────────────────────────────────────────


def _build_prompt_in_subprocess() -> str:
    """
    Build the Conductor system prompt in a fresh Python process and return it.
    This ensures we catch differences that only manifest across Python sessions.
    """
    code = textwrap.dedent(
        '''
        import os, sys
        sys.path.insert(0, os.getcwd())
        # Install the same static timestamp override used by pytest's autouse fixture,
        # but inside this fresh process so the time footer is deterministic.
        import unity.common.prompt_helpers as _ph
        from datetime import datetime, timezone
        def _static_now(time_only: bool = False):
            dt = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
            label = "UTC"
            return (
                dt.strftime("%H:%M:%S ") + label
                if time_only
                else dt.strftime("%Y-%m-%d %H:%M:%S ") + label
            )
        _ph.now = _static_now
        from unity.conductor.prompt_builders import build_request_prompt

        # Create mock tools
        def ContactManager_ask(text: str) -> str: return ""
        def ContactManager_update(text: str) -> str: return ""
        def TranscriptManager_ask(text: str) -> str: return ""
        def KnowledgeManager_ask(text: str) -> str: return ""
        def KnowledgeManager_update(text: str) -> str: return ""
        def GuidanceManager_ask(text: str) -> str: return ""
        def GuidanceManager_update(text: str) -> str: return ""
        def TaskScheduler_ask(text: str) -> str: return ""
        def TaskScheduler_update(text: str) -> str: return ""
        def TaskScheduler_execute(text: str) -> str: return ""
        def WebSearcher_ask(text: str) -> str: return ""
        def WebSearcher_update(text: str) -> str: return ""
        def SecretManager_ask(text: str) -> str: return ""
        def Actor_act(description: str) -> str: return ""
        def GlobalFileManager_ask(text: str) -> str: return ""
        def GlobalFileManager_organize(text: str) -> str: return ""
        def ConversationManagerHandle_ask(question: str) -> str: return ""
        def ConversationManagerHandle_interject(message: str) -> str: return ""
        def ConversationManagerHandle_get_full_transcript() -> str: return ""

        tools = {
            "ContactManager_ask": ContactManager_ask,
            "ContactManager_update": ContactManager_update,
            "TranscriptManager_ask": TranscriptManager_ask,
            "KnowledgeManager_ask": KnowledgeManager_ask,
            "KnowledgeManager_update": KnowledgeManager_update,
            "GuidanceManager_ask": GuidanceManager_ask,
            "GuidanceManager_update": GuidanceManager_update,
            "TaskScheduler_ask": TaskScheduler_ask,
            "TaskScheduler_update": TaskScheduler_update,
            "TaskScheduler_execute": TaskScheduler_execute,
            "WebSearcher_ask": WebSearcher_ask,
            "WebSearcher_update": WebSearcher_update,
            "SecretManager_ask": SecretManager_ask,
            "Actor_act": Actor_act,
            "GlobalFileManager_ask": GlobalFileManager_ask,
            "GlobalFileManager_organize": GlobalFileManager_organize,
            "ConversationManagerHandle_ask": ConversationManagerHandle_ask,
            "ConversationManagerHandle_interject": ConversationManagerHandle_interject,
            "ConversationManagerHandle_get_full_transcript": ConversationManagerHandle_get_full_transcript,
        }

        prompt = build_request_prompt(tools=tools)
        sys.stdout.write(prompt)
        ''',
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return proc.stdout


def test_request_prompt_stable():
    # Build prompts in two separate Python processes to catch cross-session drift
    p1 = _build_prompt_in_subprocess()
    p2 = _build_prompt_in_subprocess()
    if p1 != p2:
        snippet = first_diff_block(p1, p2, context=3, label_a="First", label_b="Second")
        raise AssertionError(
            "Request system prompt changed between separate Python sessions.\n\n"
            + snippet,
        )
