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
from unity.function_manager.primitives.registry import get_registry


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
    _registry = get_registry()

    class _ServiceEnv(BaseEnvironment):
        def get_tools(self) -> Dict[str, ToolMetadata]:
            tools: Dict[str, ToolMetadata] = {}
            for name in dir(instance):
                if name.startswith("_"):
                    continue
                attr = getattr(instance, name, None)
                if not callable(attr):
                    continue
                fq_name = f"{namespace}.{name}"
                tools[fq_name] = ToolMetadata(
                    name=fq_name,
                    is_impure=True,
                    is_steerable=False,
                )
            return tools

        def get_prompt_context(self) -> str:
            lines = [f"### `{namespace}` Service\n"]
            for name in dir(instance):
                if name.startswith("_"):
                    continue
                attr = getattr(instance, name, None)
                if not callable(attr):
                    continue
                sig_str = _registry._format_method_signature(
                    type(instance),
                    name,
                )
                full_doc = inspect.getdoc(attr) or ""
                filtered_doc = _registry._filter_internal_params_from_docstring(
                    full_doc,
                )
                lines.append(f"\n**`{namespace}.{name}{sig_str}`**")
                if filtered_doc:
                    for doc_line in filtered_doc.splitlines():
                        lines.append(f"  {doc_line}")
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
