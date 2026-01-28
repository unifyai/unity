"""Enum for chat context propagation control."""

from enum import Enum


class ChatContextPropagation(Enum):
    """Controls how parent chat context is propagated to inner tools.

    ALWAYS
        Always pass parent chat context to tools that accept ``_parent_chat_context``.
        The LLM does not get a choice; context is injected automatically.

    NEVER
        Never pass parent chat context, even to tools that accept it.

    LLM_DECIDES
        Expose an ``include_parent_chat_context: bool = True`` parameter in the
        tool schema. The LLM can set this to ``false`` to skip context propagation
        for a specific tool call. If omitted, defaults to ``True`` (context included).
    """

    ALWAYS = "always"
    NEVER = "never"
    LLM_DECIDES = "llm"
