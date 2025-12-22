from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel


class ToolMetadata(BaseModel):
    """Metadata describing a tool's behavior and safety characteristics.

    Attributes:
        name: Fully-qualified tool name as used in the Actor execution sandbox.
        is_impure: True if the tool can cause side effects.
        is_steerable: True if calling the tool may return a steerable handle.
        docstring: Tool documentation string (if available).
        signature: Human-readable signature string (if available).
    """

    name: str
    is_impure: bool
    is_steerable: bool = False
    docstring: Optional[str] = None
    signature: Optional[str] = None


class BaseEnvironment(ABC):
    """Abstract interface for execution environments.

    An environment encapsulates a domain of tools (browser control, state managers,
    custom adapters) and provides:
    - a namespace to inject into the plan execution sandbox
    - metadata for tools (purity/steerability)
    - a prompt context section describing usage patterns for those tools

    NOTE: proxying/caching/logging is owned by the Actor, not the environment.
    """

    @property
    @abstractmethod
    def namespace(self) -> str:
        """Global variable name injected into the sandbox (e.g. "computer_primitives")."""

    @abstractmethod
    def get_instance(self) -> Any:
        """Return the object injected into the sandbox under `namespace`."""

    @abstractmethod
    def get_tools(self) -> Dict[str, ToolMetadata]:
        """Return metadata for tools exposed by this environment.

        The returned keys MUST be fully-qualified tool names as used in execution,
        so callers can look up metadata by the same string that appears in logs.
        """

    @abstractmethod
    def get_prompt_context(self) -> str:
        """Return Markdown-formatted rules/examples for using this environment."""

    @abstractmethod
    async def capture_state(self) -> Dict[str, Any]:
        """Capture environment-specific evidence for verification.

        This is used by the Actor's verification system to gather a structured
        snapshot of the environment's observable state before/after executing a
        plan function.

        Implementations should be best-effort and never raise; if state capture
        fails, return a structured error payload (e.g. `{"type": "...", "error": "..."}`).
        """
