from __future__ import annotations

from typing import Any, Dict

from unity.actor.environments.base import BaseEnvironment, ToolMetadata
from unity.function_manager.primitives import PRIMITIVE_SOURCES, Primitives


class StateManagerEnvironment(BaseEnvironment):
    """State manager environment backed by `unity.function_manager.primitives.Primitives`.

    Exposes state manager methods like `primitives.contacts.ask(...)` for use inside
    generated plan code.
    """

    def __init__(self, primitives: Primitives):
        self._primitives = primitives

    @property
    def namespace(self) -> str:
        return "primitives"

    def get_instance(self) -> Primitives:
        return self._primitives

    def get_prompt_context(self) -> str:
        """Return Markdown-formatted rules/examples for using state managers."""
        return ""

    def get_tools(self) -> Dict[str, ToolMetadata]:
        # The public surface for state managers is driven by the shared primitives registry
        # (`PRIMITIVE_SOURCES`) to avoid hardcoding manager/method lists in multiple places.
        #
        # IMPORTANT: We are intentionally conservative with purity:
        # - Only clearly read-only methods are treated as pure (cacheable).
        # - Unknown methods default to impure to avoid incorrectly caching side effects.
        pure_methods = {
            "ask",
            "get",
            "list",
            "search",
            "exists",
            "parse",
            "preview",
        }

        def _infer_primitives_attr_name(class_path: str) -> str | None:
            class_name = class_path.rsplit(".", 1)[-1]
            # Strip common suffixes used by managers.
            for suffix in ("Manager", "Scheduler", "Searcher"):
                if class_name.endswith(suffix):
                    class_name = class_name[: -len(suffix)]
                    break

            base = class_name[:1].lower() + class_name[1:]

            # Prefer plural if present on Primitives (common convention: contacts, tasks, secrets).
            plural = f"{base}s"
            if hasattr(Primitives, plural):
                return plural
            if hasattr(Primitives, base):
                return base

            # Fallback for irregular cases.
            special = {
                "Task": "tasks",
                "Tasks": "tasks",
                "Contact": "contacts",
                "Transcript": "transcripts",
                "Secret": "secrets",
                "Web": "web",
            }
            for k, v in special.items():
                if class_name == k and hasattr(Primitives, v):
                    return v
            return None

        tools: Dict[str, ToolMetadata] = {}
        for class_path, method_names in PRIMITIVE_SOURCES:
            # Skip ComputerPrimitives; those belong to the `computer_primitives` environment.
            if class_path.endswith(".ComputerPrimitives"):
                continue

            manager_attr = _infer_primitives_attr_name(class_path)
            if not manager_attr:
                # If the runtime `Primitives` interface doesn't expose this manager, skip it.
                continue

            for method_name in method_names:
                fq_name = f"{self.namespace}.{manager_attr}.{method_name}"
                tools[fq_name] = ToolMetadata(
                    name=fq_name,
                    is_impure=(method_name not in pure_methods),
                    is_steerable=True,
                    docstring=None,
                    signature=None,
                )

        return tools

    def get_prompt_context(self) -> str:
        """Markdown-formatted guidance for using state-manager primitives in plans."""

        return (
            "### State manager primitives (`primitives.*`)\n"
            "- Use these tools to query or mutate durable assistant state (contacts, tasks, transcripts, etc.).\n"
            "- Calls like `await primitives.contacts.ask(...)` may return a steerable handle; await `.result()` for the final answer.\n"
            "- If a tool asks for clarification, wait for the user response and then answer via the handle's clarification API.\n"
            "- Prefer `ask(...)` for read-only queries and `update(...)`/`execute(...)` only when mutations or tracked execution are intended.\n"
        )

    async def capture_state(self) -> Dict[str, Any]:
        """State manager \"state\" is primarily evidenced via return values."""
        return {
            "type": "return_value",
            "note": "State manager evidence is captured via function return values.",
        }
