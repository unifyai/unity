"""
Integration tests for PrimitiveScope across all system layers.

These tests verify that the scoping mechanism works consistently at ALL levels:
- Prompts (prompt_context, prompt_examples)
- Tool list (tool_names, get_tools)
- Semantic search results (search_functions filtering)
- Primitives syncing (per-manager hash tracking, batched sync)
- Sandbox runtime vars (StateManagerEnvironment)

This ensures the single source of truth (PrimitiveScope) controls what the model sees.
"""

import pytest

from unity.function_manager.primitives import (
    PrimitiveScope,
    get_registry,
    VALID_MANAGER_ALIASES,
)
from unity.function_manager.primitives.registry import get_primitive_sources
from unity.function_manager.function_manager import FunctionManager
from unity.actor.environments.state_managers import StateManagerEnvironment
from unity.function_manager.primitives import Primitives
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def scoped_function_manager_factory():
    """Factory that creates FunctionManager with specific scope."""
    managers = []

    def _create(scope: PrimitiveScope):
        ContextRegistry.forget(FunctionManager, "Functions/VirtualEnvs")
        ContextRegistry.forget(FunctionManager, "Functions/Compositional")
        ContextRegistry.forget(FunctionManager, "Functions/Primitives")
        ContextRegistry.forget(FunctionManager, "Functions/Meta")
        fm = FunctionManager(primitive_scope=scope)
        managers.append(fm)
        return fm

    yield _create

    for fm in managers:
        try:
            fm.clear()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# 1. Prompt Context Scoping
# ────────────────────────────────────────────────────────────────────────────


def test_prompt_context_only_includes_scoped_managers():
    """prompt_context() must NOT mention unscoped managers."""
    registry = get_registry()
    scope = PrimitiveScope(scoped_managers=frozenset({"files"}))
    context = registry.prompt_context(scope)

    # Should include files
    assert "primitives.files" in context

    # Should NOT include any other managers
    for alias in VALID_MANAGER_ALIASES - {"files"}:
        assert (
            f"primitives.{alias}" not in context
        ), f"Unscoped manager '{alias}' should not appear in prompt context"


def test_prompt_examples_only_includes_scoped_managers():
    """prompt_examples() must NOT include examples for unscoped managers."""
    registry = get_registry()

    # Scope to only files
    scope_files = PrimitiveScope.single("files")
    examples_files = registry.prompt_examples(scope_files)

    # Should contain files examples
    assert "files" in examples_files.lower()

    # Should NOT contain contacts examples
    assert "contacts.ask" not in examples_files
    assert "contacts.update" not in examples_files


# ────────────────────────────────────────────────────────────────────────────
# 2. Tool List Scoping
# ────────────────────────────────────────────────────────────────────────────


def test_tool_names_strictly_scoped():
    """tool_names() must return ONLY tools for scoped managers."""
    registry = get_registry()
    scope = PrimitiveScope(scoped_managers=frozenset({"files", "data"}))
    names = registry.tool_names(scope)

    for name in names:
        # Must start with primitives.files. or primitives.data.
        valid_prefixes = ["primitives.files.", "primitives.data."]
        assert any(
            name.startswith(p) for p in valid_prefixes
        ), f"Tool '{name}' should not be exposed for scope {scope.scoped_managers}"


def test_state_manager_env_get_tools_respects_scope():
    """StateManagerEnvironment.get_tools() must respect primitive_scope."""
    scope = PrimitiveScope.single("files")
    env = StateManagerEnvironment(Primitives(primitive_scope=scope))
    tools = env.get_tools()

    for tool_name in tools:
        assert tool_name.startswith(
            "primitives.files.",
        ), f"Tool '{tool_name}' should not be exposed for files-only scope"


def test_state_manager_env_exposes_comms_namespace_when_scoped():
    """Comms-only scope should expose assistant-owned comms under `primitives.comms`."""
    scope = PrimitiveScope.single("comms")
    env = StateManagerEnvironment(Primitives(primitive_scope=scope))
    tools = env.get_tools()

    expected = {
        "primitives.comms.send_sms",
        "primitives.comms.send_whatsapp",
        "primitives.comms.send_discord_message",
        "primitives.comms.send_discord_channel_message",
        "primitives.comms.send_unify_message",
        "primitives.comms.send_api_response",
        "primitives.comms.send_email",
        "primitives.comms.make_call",
        "primitives.comms.make_whatsapp_call",
    }

    assert expected.issubset(set(tools))
    assert all(name.startswith("primitives.comms.") for name in tools)


def test_state_manager_env_excludes_computer_primitives():
    """StateManagerEnvironment respects the scope it is given.

    When the scope excludes computer, no computer primitives appear.
    When the scope includes computer, they flow through like any other manager.
    """
    # Scope without computer → no computer primitives.
    sm_only = frozenset(VALID_MANAGER_ALIASES - {"computer", "actor"})
    scope = PrimitiveScope(scoped_managers=sm_only)
    env = StateManagerEnvironment(Primitives(primitive_scope=scope))
    tools = env.get_tools()

    for tool_name in tools:
        assert not tool_name.startswith(
            "primitives.computer",
        ), "ComputerPrimitives should not appear when excluded from scope"
        assert not tool_name.startswith(
            "primitives.actor",
        ), "ActorPrimitives should not appear when excluded from scope"

    # Scope with computer → computer primitives included.
    full_scope = PrimitiveScope.all_managers()
    full_env = StateManagerEnvironment(Primitives(primitive_scope=full_scope))
    full_tools = full_env.get_tools()
    computer_tools = [t for t in full_tools if t.startswith("primitives.computer")]
    assert (
        len(computer_tools) > 0
    ), "Computer primitives should appear when included in scope"


# ────────────────────────────────────────────────────────────────────────────
# 3. Semantic Search Scoping
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_search_functions_respects_scope(scoped_function_manager_factory):
    """search_functions() should only return primitives for scoped managers."""
    scope = PrimitiveScope.single("files")
    fm = scoped_function_manager_factory(scope)

    # Sync primitives (scoped)
    fm.sync_primitives()

    # Search for something generic
    results = fm.search_functions(query="data operations", n=20)

    # All primitive results should be from files manager
    for r in results:
        if r.get("is_primitive"):
            assert "FileManager" in r.get(
                "primitive_class",
                "",
            ), f"Primitive {r.get('name')} should be from FileManager for files-only scope"


@_handle_project
def test_list_primitives_respects_scope(scoped_function_manager_factory):
    """list_primitives() should only return primitives for scoped managers."""
    scope = PrimitiveScope(scoped_managers=frozenset({"files", "contacts"}))
    fm = scoped_function_manager_factory(scope)

    fm.sync_primitives()
    primitives = fm.list_primitives()

    # All primitives should be from files or contacts
    for name, data in primitives.items():
        primitive_class = data.get("primitive_class", "")
        assert (
            "FileManager" in primitive_class or "ContactManager" in primitive_class
        ), f"Primitive {name} should be from FileManager or ContactManager"


# ────────────────────────────────────────────────────────────────────────────
# 4. Primitives Syncing Scoping
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_sync_primitives_only_syncs_scoped_managers(scoped_function_manager_factory):
    """sync_primitives() should only sync primitives for scoped managers."""
    # Create FM with files-only scope
    scope = PrimitiveScope.single("files")
    fm = scoped_function_manager_factory(scope)

    fm.sync_primitives()
    primitives = fm.list_primitives()

    # Should only have FileManager primitives
    assert len(primitives) > 0, "Should have synced some primitives"
    for name, data in primitives.items():
        assert "FileManager" in data.get(
            "primitive_class",
            "",
        ), f"Primitive {name} should be from FileManager"


@_handle_project
def test_per_manager_hash_tracking(scoped_function_manager_factory):
    """FunctionManager should track hashes per-manager for efficient syncing."""
    scope = PrimitiveScope.all_managers()
    fm = scoped_function_manager_factory(scope)

    # Sync primitives
    fm.sync_primitives()

    # Check that per-manager hashes are stored
    stored_hashes = fm._get_stored_primitives_hash_by_manager()

    # Should have a hash for each scoped manager
    for alias in scope.scoped_managers:
        assert alias in stored_hashes, f"Should have hash for {alias}"
        assert (
            len(stored_hashes[alias]) == 16
        ), f"Hash for {alias} should be 16-char hex"


# ────────────────────────────────────────────────────────────────────────────
# 5. Sandbox Runtime Scoping
# ────────────────────────────────────────────────────────────────────────────


def test_primitives_instance_respects_scope():
    """Primitives instance should only expose scoped managers."""
    from unity.function_manager.primitives import Primitives

    scope = PrimitiveScope.single("files")
    primitives = Primitives(primitive_scope=scope)

    # files should be accessible
    assert hasattr(primitives, "files")

    # Other managers should still be accessible (Primitives doesn't block access)
    # but the scoping is enforced at the environment/FunctionManager level
    # The runtime Primitives instance gives full access for flexibility


def test_state_manager_env_get_prompt_context_respects_scope():
    """StateManagerEnvironment.get_prompt_context() must respect scope."""
    scope = PrimitiveScope(scoped_managers=frozenset({"files", "contacts"}))
    env = StateManagerEnvironment(Primitives(primitive_scope=scope))
    context = env.get_prompt_context()

    # Should include scoped managers in the method reference sections
    assert "#### `primitives.files`" in context
    assert "#### `primitives.contacts`" in context

    # Should NOT include unscoped managers in method reference sections
    # (general rules/examples may reference them generically)
    assert "#### `primitives.tasks`" not in context
    assert "#### `primitives.knowledge`" not in context


# ────────────────────────────────────────────────────────────────────────────
# 6. Cross-Layer Consistency
# ────────────────────────────────────────────────────────────────────────────


def test_scope_consistency_across_layers():
    """All layers should expose the same set of managers for a given scope."""
    registry = get_registry()
    scope = PrimitiveScope(scoped_managers=frozenset({"files", "contacts"}))

    # Get managers from each layer
    tool_names_managers = {name.split(".")[1] for name in registry.tool_names(scope)}

    env = StateManagerEnvironment(Primitives(primitive_scope=scope))
    env_tools_managers = {name.split(".")[1] for name in env.get_tools().keys()}

    prompt_context = registry.prompt_context(scope)
    # Extract managers mentioned in prompt
    context_managers = {
        alias
        for alias in VALID_MANAGER_ALIASES
        if f"primitives.{alias}" in prompt_context
    }

    # All layers should expose exactly the scoped managers
    expected = {"files", "contacts"}
    assert tool_names_managers == expected, f"tool_names exposed {tool_names_managers}"
    assert env_tools_managers == expected, f"env.get_tools exposed {env_tools_managers}"
    assert context_managers == expected, f"prompt_context mentioned {context_managers}"


def test_primitive_discovery_complete():
    """All auto-discovered primitives should be indexed correctly."""
    registry = get_registry()
    scope = PrimitiveScope.all_managers()
    collected = registry.collect_primitives(scope)

    # Verify against get_primitive_sources.
    # Build a lookup by (class_name_suffix, method) since names are now
    # in ``primitives.{alias}.{method}`` format.
    method_to_name = {
        (row["primitive_class"].rsplit(".", 1)[-1], row["primitive_method"]): name
        for name, row in collected.items()
    }
    for cls, method_names in get_primitive_sources():
        class_name = cls.__name__
        for method_name in method_names:
            assert (
                class_name,
                method_name,
            ) in method_to_name, (
                f"Primitive for {class_name}.{method_name} should be collected"
            )
            # Verify it has all required fields
            data = collected[method_to_name[(class_name, method_name)]]
            assert data["is_primitive"] is True
            assert data["primitive_class"] is not None
            assert data["primitive_method"] == method_name
