"""
Shared assertion helpers for tests
==================================

Contains common functions for formatting assertion error messages
with detailed context including reasoning steps from LLM tool usage.
"""

import json
from typing import Any, List, Dict, Optional, Union


def format_reasoning_steps(reasoning: List[Dict[str, Any]]) -> str:
    """Format reasoning steps from LLM tool use loops for better readability."""
    if not reasoning:
        return "No reasoning steps available"

    # Pretty print the reasoning steps, handling nested content fields
    def format_json_content(msg):
        if "content" in msg and msg["content"]:
            try:
                msg["content"] = json.loads(msg["content"])
            except (json.JSONDecodeError, TypeError):
                pass
        return msg

    formatted_reasoning = [format_json_content(msg) for msg in reasoning]
    formatted_reasoning = json.dumps(formatted_reasoning, indent=4)
    formatted_reasoning = formatted_reasoning.replace("\\n", "\n")

    return formatted_reasoning


def assertion_failed(
    expected: Any,
    actual: Any,
    reasoning: List[Dict[str, Any]],
    description: str = "",
    context_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a detailed error message for assertion failures with LLM reasoning.

    Args:
        expected: The expected value
        actual: The actual value received
        reasoning: List of reasoning steps from LLM tool use
        description: Optional description of the assertion
        context_data: Optional additional context data to include (e.g., tasks, messages)

    Returns:
        Formatted error message string
    """
    context_str = ""
    if context_data:
        for label, data in context_data.items():
            context_str += f"\n{label}:\n{json.dumps(data, indent=4)}\n"

    formatted_reasoning = format_reasoning_steps(reasoning)

    return (
        f"\n{description}\n"
        f"Expected:\n{expected}\n"
        f"Got:\n{actual}\n"
        f"{context_str}"
        f"Reasoning:\n{formatted_reasoning}\n"
    )
