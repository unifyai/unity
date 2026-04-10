"""Tests for ToolSurfaceRegistry."""

import pytest

from unity.function_manager.primitives.scope import PrimitiveScope
from unity.function_manager.primitives.registry import (
    get_registry,
    get_primitive_sources,
    ManagerSpec,
    _COMMON_EXCLUDED_METHODS,
    _MANAGER_SPECS,
)

# ────────────────────────────────────────────────────────────────────────────
# Singleton and basic registry tests
# ────────────────────────────────────────────────────────────────────────────


def test_singleton_registry():
    """get_registry() returns singleton."""
    r1 = get_registry()
    r2 = get_registry()
    assert r1 is r2


def test_manager_spec_frozen():
    """ManagerSpec is frozen (immutable)."""
    spec = ManagerSpec(
        manager_alias="test",
        manager_registry_key="test",
        primitive_class_path="test.TestManager",
    )
    with pytest.raises(AttributeError):
        spec.manager_alias = "changed"  # type: ignore


# ────────────────────────────────────────────────────────────────────────────
# Scoping tests - manager_specs filtering
# ────────────────────────────────────────────────────────────────────────────


def test_manager_specs_filtered_by_scope():
    """manager_specs() returns only scoped managers."""
    registry = get_registry()
    scope = PrimitiveScope(scoped_managers=frozenset({"files", "contacts"}))
    specs = registry.manager_specs(scope)
    aliases = {s.manager_alias for s in specs}
    assert aliases == {"files", "contacts"}
    # Should NOT include unscoped managers
    assert "tasks" not in aliases
    assert "knowledge" not in aliases


def test_manager_specs_sorted_by_priority():
    """manager_specs() returns specs sorted by priority."""
    registry = get_registry()
    scope = PrimitiveScope.all_managers()
    specs = registry.manager_specs(scope)
    priorities = [s.priority for s in specs]
    assert priorities == sorted(priorities)


def test_manager_specs_includes_computer_primitives():
    """manager_specs() includes ComputerPrimitives like any other manager."""
    registry = get_registry()
    scope = PrimitiveScope.all_managers()
    specs = registry.manager_specs(scope)
    aliases = {s.manager_alias for s in specs}
    assert "computer" in aliases


def test_get_manager_spec_valid():
    """get_manager_spec() returns spec for valid alias."""
    registry = get_registry()
    spec = registry.get_manager_spec("files")
    assert spec is not None
    assert spec.manager_alias == "files"
    assert spec.domain == "File Operations & Document Parsing"


def test_get_manager_spec_invalid():
    """get_manager_spec() returns None for invalid alias."""
    registry = get_registry()
    spec = registry.get_manager_spec("invalid")
    assert spec is None


def test_get_manager_spec_includes_computer():
    """get_manager_spec() returns the ComputerPrimitives spec."""
    registry = get_registry()
    spec = registry.get_manager_spec("computer")
    assert spec is not None
    assert spec.manager_alias == "computer"


# ────────────────────────────────────────────────────────────────────────────
# Primitive methods discovery tests
# ────────────────────────────────────────────────────────────────────────────


def test_primitive_methods_for_files():
    """primitive_methods() returns expected methods for files manager."""
    registry = get_registry()
    methods = registry.primitive_methods(manager_alias="files")
    # Should include real FileManager primitives
    assert "describe" in methods
    assert "reduce" in methods
    assert "filter_files" in methods
    assert "search_files" in methods
    # Should exclude internal methods from EXCLUDED_METHODS
    assert "ingest_files" not in methods
    assert "delete_file" not in methods
    assert "exists" not in methods


def test_primitive_methods_for_contacts():
    """primitive_methods() returns expected methods for contacts manager."""
    registry = get_registry()
    methods = registry.primitive_methods(manager_alias="contacts")
    assert "ask" in methods
    assert "update" in methods
    # Should exclude internal methods
    assert "filter_contacts" not in methods
    assert "update_contact" not in methods


def test_primitive_methods_respects_common_exclusions():
    """primitive_methods() excludes common excluded methods."""
    registry = get_registry()
    # Check for all managers
    for alias in ["contacts", "files", "tasks", "knowledge"]:
        methods = registry.primitive_methods(manager_alias=alias)
        for excluded in _COMMON_EXCLUDED_METHODS:
            assert (
                excluded not in methods
            ), f"{excluded} should be excluded from {alias}"


# ────────────────────────────────────────────────────────────────────────────
# Tool names scoping tests
# ────────────────────────────────────────────────────────────────────────────


def test_tool_names_scoped():
    """tool_names() returns fully-qualified names for scoped managers only."""
    registry = get_registry()
    scope = PrimitiveScope.single("files")
    names = registry.tool_names(scope)
    assert all(name.startswith("primitives.files.") for name in names)
    assert "primitives.files.describe" in names
    # Should NOT include other managers
    assert not any(name.startswith("primitives.contacts.") for name in names)


def test_tool_names_multiple_managers():
    """tool_names() includes all scoped managers."""
    registry = get_registry()
    scope = PrimitiveScope(scoped_managers=frozenset({"files", "contacts"}))
    names = registry.tool_names(scope)
    has_files = any(name.startswith("primitives.files.") for name in names)
    has_contacts = any(name.startswith("primitives.contacts.") for name in names)
    assert has_files and has_contacts


# ────────────────────────────────────────────────────────────────────────────
# Prompt context scoping tests
# ────────────────────────────────────────────────────────────────────────────


def test_prompt_context_includes_scoped_managers():
    """prompt_context() includes only scoped managers."""
    registry = get_registry()
    scope = PrimitiveScope(scoped_managers=frozenset({"files", "contacts"}))
    context = registry.prompt_context(scope)
    assert "primitives.files" in context
    assert "primitives.contacts" in context
    # Should NOT include unscoped managers
    assert "primitives.tasks" not in context
    assert "primitives.knowledge" not in context


def test_prompt_context_single_manager_no_general_rules():
    """prompt_context() omits general rules for single manager."""
    registry = get_registry()
    scope = PrimitiveScope.single("files")
    context = registry.prompt_context(scope)
    # Should NOT include multi-manager rules
    assert "Manager Selection Priorities" not in context
    assert "General Rules" not in context


def test_prompt_context_multiple_managers_has_general_rules():
    """prompt_context() includes general rules for multiple managers."""
    registry = get_registry()
    scope = PrimitiveScope(scoped_managers=frozenset({"files", "contacts", "tasks"}))
    context = registry.prompt_context(scope)
    assert "Manager Selection Priorities" in context
    assert "General Rules" in context


def test_prompt_context_routing_guidance_only_when_both_present():
    """prompt_context() only includes routing guidance when both managers are scoped."""
    registry = get_registry()
    # data+files has routing guidance
    scope_both = PrimitiveScope(scoped_managers=frozenset({"data", "files"}))
    context_both = registry.prompt_context(scope_both)
    assert "primitives.data.*` vs `primitives.files.*" in context_both

    # files only should NOT have data/files routing guidance
    scope_files = PrimitiveScope.single("files")
    context_files = registry.prompt_context(scope_files)
    assert "primitives.data.*` vs `primitives.files.*" not in context_files

    # dashboards+data are orthogonal -- no routing guidance between them
    scope_dash_data = PrimitiveScope(
        scoped_managers=frozenset({"dashboards", "data"}),
    )
    context_dash_data = registry.prompt_context(scope_dash_data)
    assert "primitives.dashboards.*` vs `primitives.data.*" not in context_dash_data


# ────────────────────────────────────────────────────────────────────────────
# Primitive row filter tests (for FunctionManager queries)
# ────────────────────────────────────────────────────────────────────────────


def test_primitive_row_filter():
    """primitive_row_filter() builds valid filter expression using primitive_class."""
    registry = get_registry()
    scope = PrimitiveScope(scoped_managers=frozenset({"files", "contacts"}))
    filter_expr = registry.primitive_row_filter(scope)
    # Filter uses OR clauses with primitive_class
    assert "primitive_class ==" in filter_expr
    assert " or " in filter_expr
    assert "unity.contact_manager.contact_manager.ContactManager" in filter_expr
    assert "unity.file_manager.managers.file_manager.FileManager" in filter_expr
    # Should NOT include unscoped managers
    assert "TaskScheduler" not in filter_expr


def test_primitive_row_filter_single_manager():
    """primitive_row_filter() works for single manager."""
    registry = get_registry()
    scope = PrimitiveScope.single("contacts")
    filter_expr = registry.primitive_row_filter(scope)
    # Single manager should have a single clause (no "or")
    assert "primitive_class ==" in filter_expr
    assert "ContactManager" in filter_expr
    assert "FileManager" not in filter_expr
    # No OR needed for single manager
    assert " or " not in filter_expr


# ────────────────────────────────────────────────────────────────────────────
# Collect primitives tests
# ────────────────────────────────────────────────────────────────────────────


def test_collect_primitives_returns_expected_fields():
    """collect_primitives() returns rows with required fields."""
    registry = get_registry()
    scope = PrimitiveScope.single("files")
    primitives = registry.collect_primitives(scope)
    assert len(primitives) > 0
    for name, row in primitives.items():
        assert "name" in row
        assert "function_id" in row
        assert "primitive_class" in row
        assert "primitive_method" in row
        assert "argspec" in row
        assert "docstring" in row
        assert "embedding_text" in row
        assert row["is_primitive"] is True


def test_collect_primitives_matches_get_primitive_sources():
    """collect_primitives() discovers a superset of get_primitive_sources().

    get_primitive_sources() only returns state managers, while
    collect_primitives() also includes ComputerPrimitives.  Every state
    manager primitive from get_primitive_sources must appear in the
    collected set.
    """
    registry = get_registry()
    scope = PrimitiveScope.all_managers()
    primitives = registry.collect_primitives(scope)

    # Build a reverse lookup: (primitive_class_suffix, method) -> name
    method_to_name = {
        (row["primitive_class"].rsplit(".", 1)[-1], row["primitive_method"]): name
        for name, row in primitives.items()
    }
    for cls, method_names in get_primitive_sources():
        class_name = cls.__name__
        for method_name in method_names:
            assert (
                class_name,
                method_name,
            ) in method_to_name, f"Expected auto-discovered primitive for {class_name}.{method_name} not found"

    # Additionally, computer primitives should be present.
    computer_entries = {
        name for name in primitives if name.startswith("primitives.computer.")
    }
    assert (
        len(computer_entries) > 0
    ), "Expected ComputerPrimitives in collected primitives"


def test_collect_primitives_respects_scope():
    """collect_primitives() only collects primitives for scoped managers."""
    registry = get_registry()
    scope = PrimitiveScope.single("files")
    primitives = registry.collect_primitives(scope)

    # All primitives should be from FileManager
    for name, row in primitives.items():
        assert (
            "FileManager" in row["primitive_class"]
        ), f"Primitive {name} should be from FileManager"


def test_collect_primitives_stable_ids():
    """collect_primitives() generates stable IDs across calls."""
    registry = get_registry()
    scope = PrimitiveScope.all_managers()

    primitives1 = registry.collect_primitives(scope)
    primitives2 = registry.collect_primitives(scope)

    for name in primitives1:
        assert (
            primitives1[name]["function_id"] == primitives2[name]["function_id"]
        ), f"ID for '{name}' should be stable across calls"


def test_collect_primitives_unique_ids():
    """collect_primitives() generates unique IDs for each primitive."""
    registry = get_registry()
    scope = PrimitiveScope.all_managers()
    primitives = registry.collect_primitives(scope)

    ids = [p["function_id"] for p in primitives.values()]
    assert len(ids) == len(set(ids)), "Primitive IDs should be unique"


# ────────────────────────────────────────────────────────────────────────────
# Hash computation tests
# ────────────────────────────────────────────────────────────────────────────


def test_compute_hash_for_manager():
    """compute_hash_for_manager() returns consistent hash."""
    registry = get_registry()
    hash1 = registry.compute_hash_for_manager("files")
    hash2 = registry.compute_hash_for_manager("files")
    assert hash1 == hash2
    assert len(hash1) == 16  # 16-char hex


def test_compute_hash_different_for_different_managers():
    """compute_hash_for_manager() returns different hashes for different managers."""
    registry = get_registry()
    hash_files = registry.compute_hash_for_manager("files")
    hash_contacts = registry.compute_hash_for_manager("contacts")
    assert hash_files != hash_contacts


def test_compute_primitives_hash_stable():
    """compute_primitives_hash() is stable for same scope."""
    registry = get_registry()
    scope = PrimitiveScope.all_managers()
    hash1 = registry.compute_primitives_hash(primitive_scope=scope)
    hash2 = registry.compute_primitives_hash(primitive_scope=scope)
    assert hash1 == hash2


def test_compute_primitives_hash_different_for_different_scopes():
    """compute_primitives_hash() returns different hashes for different scopes."""
    registry = get_registry()
    scope_all = PrimitiveScope.all_managers()
    scope_files = PrimitiveScope.single("files")

    hash_all = registry.compute_primitives_hash(primitive_scope=scope_all)
    hash_files = registry.compute_primitives_hash(primitive_scope=scope_files)

    assert hash_all != hash_files


def test_compute_primitives_hash_accepts_precomputed():
    """compute_primitives_hash() can use pre-collected primitives."""
    registry = get_registry()
    scope = PrimitiveScope.single("files")
    primitives = registry.collect_primitives(scope)

    # Hash with pre-collected should match hash computed internally
    hash_precomputed = registry.compute_primitives_hash(primitives=primitives)
    hash_computed = registry.compute_primitives_hash(primitive_scope=scope)

    assert hash_precomputed == hash_computed


# ────────────────────────────────────────────────────────────────────────────
# Manager specs parity tests
# ────────────────────────────────────────────────────────────────────────────


def test_manager_specs_includes_all_aliases():
    """_MANAGER_SPECS covers every alias in VALID_MANAGER_ALIASES."""
    from unity.function_manager.primitives.scope import VALID_MANAGER_ALIASES

    spec_aliases = {s.manager_alias for s in _MANAGER_SPECS}
    assert spec_aliases == VALID_MANAGER_ALIASES


# ────────────────────────────────────────────────────────────────────────────
# Docstring quality tests (prompt context derives from these)
# ────────────────────────────────────────────────────────────────────────────

MIN_SUMMARY_CHARS = 20
MIN_PARAMS_CHARS = 10


def _first_paragraph(docstring: str) -> str:
    """Extract text up to the first blank line."""
    lines = []
    for line in docstring.splitlines():
        if not line.strip():
            if lines:
                break
        else:
            lines.append(line)
    return "\n".join(lines)


def _parameters_block(docstring: str) -> str:
    """Extract the NumPy-style Parameters section."""
    import re

    lines = docstring.splitlines()
    params_lines = []
    in_params = False
    for j, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped == "Parameters":
            in_params = True
            continue
        if in_params and stripped.startswith("---"):
            continue
        if in_params:
            if stripped and not stripped[0].isspace() and not stripped.startswith("-"):
                if j + 1 < len(lines) and lines[j + 1].strip().startswith("---"):
                    break
                if re.match(r"^[A-Z][a-zA-Z\s]+$", stripped) and len(stripped) < 30:
                    break
            params_lines.append(raw)
    return "\n".join(params_lines).rstrip()


def test_all_primitive_methods_have_summary_and_parameters():
    """Every public primitive method must have a non-empty first-paragraph
    summary and a non-empty NumPy-style Parameters block in its docstring.

    The prompt context rendered for the CodeActActor is derived directly
    from these docstrings.  Missing or empty sections mean the LLM gets
    no guidance on what a method does or how to call it.
    """
    registry = get_registry()

    missing_summary = []
    missing_params = []

    for spec in _MANAGER_SPECS:
        cls = registry._load_manager_class(spec.primitive_class_path)
        if cls is None:
            continue
        methods = registry.primitive_methods(manager_alias=spec.manager_alias)

        # For ComputerPrimitives, dynamic methods live on ComputerBackend.
        fallback_cls = None
        if spec.manager_alias == "computer":
            try:
                from unity.function_manager.computer_backends import ComputerBackend

                fallback_cls = ComputerBackend
            except ImportError:
                pass

        for method_name in methods:
            fq = f"primitives.{spec.manager_alias}.{method_name}"

            # Resolve the source class for docstring extraction.
            source_cls = cls
            doc = registry._extract_method_docstring(cls, method_name)
            if not doc and fallback_cls is not None:
                doc = registry._extract_method_docstring(fallback_cls, method_name)
                source_cls = fallback_cls

            summary = _first_paragraph(doc)
            if len(summary) < MIN_SUMMARY_CHARS:
                missing_summary.append(f"{fq} (got {len(summary)} chars)")

            # Only require a Parameters block if the method actually has parameters
            # beyond `self`.
            import inspect as _inspect

            try:
                sig = _inspect.signature(getattr(source_cls, method_name))
                has_params = any(
                    p.name != "self"
                    for p in sig.parameters.values()
                    if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                )
            except (ValueError, TypeError):
                has_params = True  # err on the side of requiring docs

            if has_params:
                params = _parameters_block(doc)
                if len(params) < MIN_PARAMS_CHARS:
                    missing_params.append(f"{fq} (got {len(params)} chars)")

    assert not missing_summary, (
        f"Methods with missing/short first-paragraph summary "
        f"(min {MIN_SUMMARY_CHARS} chars):\n  " + "\n  ".join(missing_summary)
    )
    assert not missing_params, (
        f"Methods with missing/short Parameters block "
        f"(min {MIN_PARAMS_CHARS} chars):\n  " + "\n  ".join(missing_params)
    )
