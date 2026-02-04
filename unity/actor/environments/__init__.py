"""Pluggable execution environments for Actors.

Environments expose domain-specific tool providers (e.g. computer/web control,
state managers) in a consistent way so an Actor can inject multiple namespaces
into its execution sandbox.

This package intentionally contains only lightweight adapters and metadata.
The Actor owns proxying/caching/logging behavior.
"""

import inspect
from typing import Any, Dict

from unity.actor.environments.base import BaseEnvironment, ToolMetadata
from unity.actor.environments.computer import ComputerEnvironment
from unity.actor.environments.state_managers import StateManagerEnvironment


def create_env(namespace: str, instance: Any) -> BaseEnvironment:
    """Create an environment from a service instance.

    The instance's public methods become callable under the namespace
    in the sandbox (e.g., `agents.run_subagent(...)`).

    Args:
        namespace: Name for the environment in the sandbox (e.g., "agents")
        instance: Service instance with methods to expose

    Returns:
        A BaseEnvironment that injects the instance into the sandbox

    Example:
        class AgentService:
            async def run_subagent(self, task: str):
                '''Spawn a subagent to handle a task.'''
                ...

        actor = CodeActActor(
            environments=[create_env("agents", AgentService())],
        )
        # In sandbox: await agents.run_subagent("do something")
    """

    class _ServiceEnv(BaseEnvironment):
        def get_tools(self) -> Dict[str, ToolMetadata]:
            # Return empty - ToolMetadata is unused by CodeActActor
            return {}

        def get_prompt_context(self) -> str:
            # Auto-generate method documentation
            lines = [f"### `{namespace}` Service\n"]
            for name in dir(instance):
                if name.startswith("_"):
                    continue
                attr = getattr(instance, name, None)
                if callable(attr):
                    try:
                        sig = str(inspect.signature(attr))
                    except (ValueError, TypeError):
                        sig = "(...)"
                    doc = inspect.getdoc(attr) or ""
                    first_line = doc.split("\n")[0] if doc else ""
                    lines.append(f"- `{namespace}.{name}{sig}`: {first_line}")
            return "\n".join(lines)

        async def capture_state(self) -> Dict[str, Any]:
            return {"type": "service", "namespace": namespace}

    return _ServiceEnv(instance=instance, namespace=namespace)


__all__ = [
    "BaseEnvironment",
    "ToolMetadata",
    "ComputerEnvironment",
    "StateManagerEnvironment",
    "create_env",
]
