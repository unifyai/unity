from __future__ import annotations

from typing import Any, Dict, Optional, Set

from unity.actor.environments.base import BaseEnvironment, ToolMetadata
from unity.function_manager.primitives import (
    MANAGER_METADATA,
    PRIMITIVE_SOURCES,
    Primitives,
)


class _FilteredPrimitivesProxy:
    """Proxy that restricts access to only allowed managers.

    When `exposed_managers` is set on StateManagerEnvironment, this proxy
    is returned by `get_instance()` instead of the raw Primitives object.
    Accessing non-allowed managers raises AttributeError with a helpful message.

    This is transparent to code that only uses allowed managers - all attributes
    and methods work identically to the underlying Primitives instance.
    """

    # Managers that exist on Primitives (used for error messages)
    _ALL_MANAGERS = frozenset(
        {
            "contacts",
            "tasks",
            "transcripts",
            "knowledge",
            "web",
            "guidance",
            "files",
            "secrets",
            "computer",
        },
    )

    def __init__(self, primitives: Primitives, allowed_managers: Set[str]):
        # Use object.__setattr__ to avoid triggering our __setattr__
        object.__setattr__(self, "_primitives", primitives)
        object.__setattr__(self, "_allowed", frozenset(allowed_managers))

    def __getattr__(self, name: str) -> Any:
        # Allow private/dunder attributes to pass through
        if name.startswith("_"):
            return getattr(self._primitives, name)

        # Check if this is a manager access
        if name in self._ALL_MANAGERS:
            if name not in self._allowed:
                raise AttributeError(
                    f"Manager 'primitives.{name}' is not available in this sandbox. "
                    f"Available managers: {sorted(self._allowed)}",
                )

        # Pass through to underlying Primitives
        return getattr(self._primitives, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._primitives, name, value)

    def __repr__(self) -> str:
        return f"<FilteredPrimitives allowed={sorted(self._allowed)}>"


class StateManagerEnvironment(BaseEnvironment):
    """State manager environment backed by `unity.function_manager.primitives.Primitives`.

    Exposes state manager methods like `primitives.contacts.ask(...)` for use inside
    generated plan code.

    Parameters
    ----------
    primitives : Primitives
        The primitives instance to wrap.
    exposed_managers : set[str] | None
        If provided, only these managers will be exposed in tools and prompt context.
        Example: {"files"} to only expose primitives.files.*
        If None (default), all managers are exposed.
    """

    def __init__(
        self,
        primitives: Primitives,
        exposed_managers: Optional[Set[str]] = None,
    ):
        self._primitives = primitives
        self._exposed_managers = exposed_managers

    @property
    def namespace(self) -> str:
        return "primitives"

    def get_instance(self) -> Primitives:
        """Return the primitives instance (always the full Primitives object)."""
        return self._primitives

    def get_sandbox_instance(self) -> Any:
        """Return the instance for sandbox injection, filtered if exposed_managers is set.

        This is used by CodeExecutionSandbox in CodeActActor to inject a filtered proxy when
        exposed_managers is set. For normal use cases (HierarchicalActor, etc.),
        use get_instance() which always returns the full Primitives. This is a temporary
        solution to avoid hardcoding the list of exposed managers in the CodeExecutionSandbox.
        """
        if self._exposed_managers is None:
            return self._primitives
        return _FilteredPrimitivesProxy(self._primitives, self._exposed_managers)

    def _infer_primitives_attr_name(self, class_path: str) -> str | None:
        """Infer the attribute name on Primitives from a class path."""
        class_name = class_path.rsplit(".", 1)[-1]
        for suffix in ("Manager", "Scheduler", "Searcher"):
            if class_name.endswith(suffix):
                class_name = class_name[: -len(suffix)]
                break

        base = class_name[:1].lower() + class_name[1:]
        plural = f"{base}s"
        if hasattr(Primitives, plural):
            return plural
        if hasattr(Primitives, base):
            return base

        special = {
            "Task": "tasks",
            "Contact": "contacts",
            "Transcript": "transcripts",
            "Secret": "secrets",
            "Web": "web",
            "File": "files",
        }
        for k, v in special.items():
            if class_name == k and hasattr(Primitives, v):
                return v
        return None

    def _is_manager_exposed(self, manager_attr: str) -> bool:
        """Check if a manager should be exposed based on filtering."""
        if self._exposed_managers is None:
            return True
        return manager_attr in self._exposed_managers

    def get_tools(self) -> Dict[str, ToolMetadata]:
        """Get tool metadata, filtered by exposed_managers if set."""
        # The public surface for state managers is driven by the shared primitives registry
        # (`PRIMITIVE_SOURCES`) to avoid hardcoding manager/method lists in multiple places.
        #
        # IMPORTANT: We are intentionally conservative with purity:
        # - Only clearly read-only methods are treated as pure (cacheable).
        # - Unknown methods default to impure to avoid incorrectly caching side effects.
        pure_methods = {
            "ask",
            "ask_about_file",  # FileManager read-only
            "get",
            "list",
            "search",
            "exists",
            "parse",
            "preview",
            "reduce",  # FileManager read-only
            "filter_files",  # FileManager read-only
            "search_files",  # FileManager read-only
            "visualize",  # FileManager read-only (generates plots, no mutation)
            "tables_overview",  # FileManager read-only
            "list_columns",  # FileManager read-only
            "schema_explain",  # FileManager read-only
        }

        tools: Dict[str, ToolMetadata] = {}
        for class_path, method_names in PRIMITIVE_SOURCES:
            # Skip ComputerPrimitives; those belong to the `computer_primitives` environment.
            if class_path.endswith(".ComputerPrimitives"):
                continue

            manager_attr = self._infer_primitives_attr_name(class_path)
            if not manager_attr:
                # If the runtime `Primitives` interface doesn't expose this manager, skip it.
                continue

            # Apply exposed_managers filter
            if not self._is_manager_exposed(manager_attr):
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
        """Dynamically generate prompt context from PRIMITIVE_SOURCES + MANAGER_METADATA.

        If exposed_managers is set, only those managers are included.
        """
        parts = ["### State manager primitives (`primitives.*`)\n"]
        parts.append(
            "Each manager owns a specific domain of the assistant's durable state. "
            "Choose the right manager for your task:\n",
        )

        # Collect exposed managers with their metadata
        exposed: list[tuple[str, list[str], dict]] = []
        for class_path, method_names in PRIMITIVE_SOURCES:
            if class_path.endswith(".ComputerPrimitives"):
                continue

            manager_attr = self._infer_primitives_attr_name(class_path)
            if not manager_attr:
                continue

            if not self._is_manager_exposed(manager_attr):
                continue

            meta = MANAGER_METADATA.get(manager_attr, {})
            exposed.append((manager_attr, method_names, meta))

        # Sort by priority (lower = higher priority)
        exposed.sort(key=lambda x: x[2].get("priority", 99))

        for manager_attr, method_names, meta in exposed:
            if not meta:
                # Fallback for managers without metadata
                parts.append(f"\n**`primitives.{manager_attr}`**")
                parts.append(
                    f"- Methods: {', '.join(f'`.{m}(...)`' for m in method_names)}",
                )
                continue

            # Format: **Domain** → `primitives.manager`
            parts.append(f"\n**{meta['domain']}** → `primitives.{manager_attr}`")
            parts.append(f"- **Domain**: {meta['description']}")

            # Show methods with their descriptions
            methods_meta = meta.get("methods", {})
            for method_name in method_names:
                if method_name in methods_meta:
                    parts.append(
                        f"- `.{method_name}(...)`: {methods_meta[method_name]}",
                    )
                else:
                    # Method exists but no description in metadata
                    parts.append(f"- `.{method_name}(...)`")

            # Add get_tools for files (special case - not in PRIMITIVE_SOURCES)
            if manager_attr == "files" and "get_tools" in methods_meta:
                parts.append(f"- `.get_tools()`: {methods_meta['get_tools']}")

            if meta.get("use_when"):
                parts.append(f"- **Use when**: {meta['use_when']}")

            if meta.get("examples"):
                parts.append(f"- **Examples**: {meta['examples']}")

            if meta.get("special_note"):
                parts.append(f"- **Note**: {meta['special_note']}")

        # Add general rules only if multiple managers exposed
        if len(exposed) > 1:
            parts.append("\n**Manager Selection Priorities**:")
            parts.append(
                "1. **knowledge** takes priority for organizational policies, procedures, company facts, internal documentation",
            )
            parts.append(
                "2. **transcripts** for historical communications (what was said/written)",
            )
            parts.append("3. **contacts** for people/relationship information")
            parts.append("4. **tasks** for work items, deadlines, assignments")
            parts.append(
                "5. **web** for current external information (weather, news, real-time data)",
            )
            parts.append("6. **guidance** for execution instructions and runbooks")
            parts.append(
                "7. **files** when dealing with specific documents or data operations",
            )

            parts.append("\n**General Rules**:")
            parts.append(
                "- All manager calls return a steerable handle; await `.result()` to get the final answer",
            )
            parts.append(
                "- If a manager asks for clarification, wait for the user response and answer via the handle's API",
            )
            parts.append(
                "- Prefer `ask(...)` for read-only queries; only use `update(...)`/`execute(...)` when mutations are needed",
            )
            parts.append(
                "- When in doubt between managers, prefer the most specific domain match",
            )

        return "\n".join(parts)

    async def capture_state(self) -> Dict[str, Any]:
        """State manager \"state\" is primarily evidenced via return values."""
        return {
            "type": "return_value",
            "note": "State manager evidence is captured via function return values.",
        }
