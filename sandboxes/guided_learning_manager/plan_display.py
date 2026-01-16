"""
Plan display layer for the guided learning sandbox.

This module provides formatting utilities to display Actor plans in a human-readable
format. It extracts metadata from the Actor's `clean_function_source_map` using
minimal AST parsing and formats it as a scannable tree view with change indicators.

Usage:
    from sandboxes.guided_learning_manager.plan_display import (
        PlanDisplayFormatter,
        PlanDisplayState,
        FunctionDisplayInfo,
    )

    formatter = PlanDisplayFormatter()
    plan_data = formatter.parse_plan_for_display(actor_handle)
    tree_view = formatter.format_tree_view(plan_data, mode="learning")
    full_plan = formatter.format_full_plan(actor_handle)
"""

import ast
import difflib
from dataclasses import dataclass
from typing import Dict, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from unity.actor.hierarchical_actor import HierarchicalActorHandle


# ────────────────────────────────────────────────────────────────────────────
# Data Classes
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class FunctionDisplayInfo:
    """Display metadata for a single function."""

    name: str
    signature: str  # e.g., "search_recipe(ingredient)"
    docstring: str  # Truncated to 60 chars
    status: str  # "NEW", "MODIFIED", "UNCHANGED", "REMOVED"
    is_main: bool = False  # True if this is main_plan()


@dataclass
class PlanDisplayState:
    """Current state of the plan for display purposes."""

    step_number: int
    functions: List[FunctionDisplayInfo]
    summary: str  # e.g., "3 functions | +1 new, 1 modified"
    mode: str = "learning"  # "learning" or "execution"
    new_count: int = 0
    modified_count: int = 0
    removed_count: int = 0
    git_diff: str = ""  # Optional unified diff for changed functions (git-style)


# ────────────────────────────────────────────────────────────────────────────
# PlanDisplayFormatter
# ────────────────────────────────────────────────────────────────────────────


class PlanDisplayFormatter:
    """
    Formats Actor plans for terminal display.

    Uses minimal AST parsing to extract function signatures and docstrings
    from the Actor's `clean_function_source_map`. Tracks changes between
    updates to show [NEW], [MODIFIED], and [REMOVED] indicators.

    Usage:
        formatter = PlanDisplayFormatter()

        # After each Actor interjection:
        plan_data = formatter.parse_plan_for_display(actor_handle)
        tree_view = formatter.format_tree_view(plan_data, mode="learning")
        print(tree_view)

        # For debug mode:
        full_plan = formatter.format_full_plan(actor_handle)
        print(full_plan)
    """

    def __init__(self):
        self.previous_functions: Set[str] = set()
        self.previous_source_map: Dict[str, str] = {}
        self.step_count: int = 0

    def parse_plan_for_display(
        self,
        handle: "HierarchicalActorHandle",
    ) -> PlanDisplayState:
        """
        Extract display-friendly metadata from Actor's data structures.

        Uses `handle.clean_function_source_map` to get function source code,
        then applies minimal AST parsing to extract signatures and docstrings.
        Compares with previous state to detect changes.

        Args:
            handle: The HierarchicalActorHandle containing plan data

        Returns:
            PlanDisplayState with function metadata and change summary
        """
        self.step_count += 1

        # Get current function source map
        current_source_map = getattr(handle, "clean_function_source_map", {})
        current_functions = set(current_source_map.keys())

        # Detect changes
        changes = self._detect_changes(
            current_functions,
            self.previous_functions,
            current_source_map,
            self.previous_source_map,
        )

        # Parse function metadata
        functions: List[FunctionDisplayInfo] = []
        for name in sorted(current_functions):
            source = current_source_map.get(name, "")
            metadata = self._parse_function_metadata(source)

            functions.append(
                FunctionDisplayInfo(
                    name=name,
                    signature=metadata["signature"],
                    docstring=metadata["docstring"],
                    status=changes.get(name, "UNCHANGED"),
                    is_main=(name == "main_plan"),
                ),
            )

        # Add removed functions for display
        for name in self.previous_functions - current_functions:
            # Get metadata from previous source if available
            prev_source = self.previous_source_map.get(name, f"def {name}(): pass")
            metadata = self._parse_function_metadata(prev_source)
            functions.append(
                FunctionDisplayInfo(
                    name=name,
                    signature=metadata["signature"],
                    docstring=metadata["docstring"],
                    status="REMOVED",
                    is_main=(name == "main_plan"),
                ),
            )

        # Count changes
        new_count = sum(1 for f in functions if f.status == "NEW")
        modified_count = sum(1 for f in functions if f.status == "MODIFIED")
        removed_count = sum(1 for f in functions if f.status == "REMOVED")

        # Generate summary
        summary = self._generate_summary(
            len(current_functions),
            new_count,
            modified_count,
            removed_count,
        )

        # Build unified diff BEFORE updating previous state.
        git_diff = self._format_unified_diff(
            changes=changes,
            current_source_map=current_source_map,
        )

        # Update previous state for next comparison
        self.previous_functions = current_functions.copy()
        self.previous_source_map = dict(current_source_map)

        return PlanDisplayState(
            step_number=self.step_count,
            functions=functions,
            summary=summary,
            mode="learning",
            new_count=new_count,
            modified_count=modified_count,
            removed_count=removed_count,
            git_diff=git_diff,
        )

    def format_tree_view(
        self,
        plan_state: PlanDisplayState,
        mode: str = "learning",
    ) -> str:
        """
        Format tree view with change indicators.

        Example output:
        ```
        🤖 PLAN UPDATED (Step 2) [learning]:
        ────────────────────────────────────────────────────────
           ├─ ✨ search_recipe(ingredient) [NEW]
           │     "Search for a recipe on allrecipes.com"
           ├─ 📝 main_plan() [MODIFIED]
           │     "Entry point - search for chicken soup"
           └─ 📊 2 functions | +1 new, 1 modified
        ────────────────────────────────────────────────────────
        ```

        Args:
            plan_state: PlanDisplayState from parse_plan_for_display()
            mode: "learning" or "execution"

        Returns:
            Formatted string for terminal display
        """
        lines: List[str] = []

        # Header
        lines.append(f"\n🤖 PLAN UPDATED (Step {plan_state.step_number}) [{mode}]:")
        lines.append("─" * 56)

        # Sort functions: main_plan first, then alphabetically
        sorted_functions = sorted(
            plan_state.functions,
            key=lambda f: (not f.is_main, f.name),
        )

        # Filter out REMOVED for main display (show at end if any)
        active_functions = [f for f in sorted_functions if f.status != "REMOVED"]
        removed_functions = [f for f in sorted_functions if f.status == "REMOVED"]

        # Format each function
        for i, func in enumerate(active_functions):
            is_last = i == len(active_functions) - 1 and not removed_functions
            prefix = "└─" if is_last else "├─"
            connector = " " if is_last else "│"

            # Status icon
            icon = self._get_status_icon(func.status)

            # Status tag
            tag = f" [{func.status}]" if func.status != "UNCHANGED" else ""

            lines.append(f"   {prefix} {icon} {func.signature}{tag}")

            # Docstring (indented)
            if func.docstring:
                lines.append(f'   {connector}     "{func.docstring}"')

        # Show removed functions
        for i, func in enumerate(removed_functions):
            is_last = i == len(removed_functions) - 1
            prefix = "└─" if is_last else "├─"

            lines.append(f"   {prefix} ❌ {func.signature} [REMOVED]")

        # Summary line
        lines.append(f"   └─ 📊 {plan_state.summary}")
        lines.append("─" * 56)

        return "\n".join(lines)

    def format_full_plan(
        self,
        handle: "HierarchicalActorHandle",
    ) -> str:
        """
        Compose full plan source from clean_function_source_map.

        This produces clean, executable Python code without instrumentation.
        Functions are ordered with main_plan last (standard convention).

        Args:
            handle: The HierarchicalActorHandle containing plan data

        Returns:
            Complete Python source code as a string
        """
        source_map = getattr(handle, "clean_function_source_map", {})

        if not source_map:
            return "# No plan generated yet"

        # Order functions: helpers first, main_plan last
        function_names = sorted(source_map.keys())
        if "main_plan" in function_names:
            function_names.remove("main_plan")
            function_names.append("main_plan")

        # Compose source
        parts: List[str] = []
        for name in function_names:
            if name in source_map:
                parts.append(source_map[name])

        return "\n\n".join(parts)

    def reset(self) -> None:
        """Reset the formatter state (for new sessions)."""
        self.previous_functions = set()
        self.previous_source_map = {}
        self.step_count = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_function_metadata(self, source: str) -> Dict[str, str]:
        """
        Extract signature and docstring via minimal AST parsing.

        Args:
            source: Python function source code

        Returns:
            Dict with "signature" and "docstring" keys
        """
        default = {"signature": "unknown()", "docstring": ""}

        if not source or not source.strip():
            return default

        try:
            tree = ast.parse(source)

            # Find the first function definition
            func_node = None
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_node = node
                    break

            if func_node is None:
                return default

            # Extract signature
            args = [arg.arg for arg in func_node.args.args]
            signature = f"{func_node.name}({', '.join(args)})"

            # Extract docstring
            docstring = ast.get_docstring(func_node) or ""

            # Truncate docstring if too long
            if len(docstring) > 60:
                docstring = docstring[:57] + "..."

            # Remove newlines for display
            docstring = docstring.replace("\n", " ").strip()

            return {"signature": signature, "docstring": docstring}

        except SyntaxError:
            # If parsing fails, try to extract function name from source
            for line in source.split("\n"):
                line = line.strip()
                if line.startswith("def ") or line.startswith("async def "):
                    # Extract name and args from definition line
                    if "(" in line and ")" in line:
                        start = line.index(" ") + 1
                        end = line.index(")")
                        signature = line[start : end + 1].replace("async ", "")
                        return {"signature": signature, "docstring": ""}

            return default

    def _format_unified_diff(
        self,
        *,
        changes: Dict[str, str],
        current_source_map: Dict[str, str],
        context_lines: int = 3,
    ) -> str:
        """Return a git-style unified diff for changed functions."""

        hunks: list[str] = []
        # Stable ordering: main_plan first if changed, then alphabetical.
        changed_names = [n for n, st in changes.items() if st in ("NEW", "MODIFIED")]
        removed_names = [n for n, st in changes.items() if st == "REMOVED"]
        ordered = sorted(changed_names) + sorted(removed_names)
        if "main_plan" in ordered:
            ordered.remove("main_plan")
            ordered.insert(0, "main_plan")

        for name in ordered:
            status = changes.get(name)
            if status not in ("NEW", "MODIFIED", "REMOVED"):
                continue

            old_src = self.previous_source_map.get(name, "")
            new_src = current_source_map.get(name, "")
            if status == "NEW":
                old_src = ""
            if status == "REMOVED":
                new_src = ""

            old_lines = old_src.splitlines(keepends=True)
            new_lines = new_src.splitlines(keepends=True)

            diff_lines = list(
                difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=f"a/{name}.py",
                    tofile=f"b/{name}.py",
                    n=context_lines,
                ),
            )
            if diff_lines:
                hunks.append("".join(diff_lines).rstrip())

        return ("\n\n".join(hunks)).strip()

    def _detect_changes(
        self,
        current_functions: Set[str],
        previous_functions: Set[str],
        current_source_map: Dict[str, str],
        previous_source_map: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Detect NEW/MODIFIED/REMOVED/UNCHANGED functions.

        Args:
            current_functions: Set of current function names
            previous_functions: Set of previous function names
            current_source_map: Current function name -> source mapping
            previous_source_map: Previous function name -> source mapping

        Returns:
            Dict mapping function names to status strings
        """
        changes: Dict[str, str] = {}

        for name in current_functions:
            if name not in previous_functions:
                changes[name] = "NEW"
            elif current_source_map.get(name) != previous_source_map.get(name):
                changes[name] = "MODIFIED"
            else:
                changes[name] = "UNCHANGED"

        for name in previous_functions - current_functions:
            changes[name] = "REMOVED"

        return changes

    def _generate_summary(
        self,
        total: int,
        new_count: int,
        modified_count: int,
        removed_count: int,
    ) -> str:
        """Generate a summary string for the plan state."""
        parts: List[str] = []
        parts.append(f"{total} function{'s' if total != 1 else ''}")

        change_parts: List[str] = []
        if new_count > 0:
            change_parts.append(f"+{new_count} new")
        if modified_count > 0:
            change_parts.append(f"{modified_count} modified")
        if removed_count > 0:
            change_parts.append(f"-{removed_count} removed")

        if change_parts:
            parts.append(" | ")
            parts.append(", ".join(change_parts))

        return "".join(parts)

    def _get_status_icon(self, status: str) -> str:
        """Get the emoji icon for a status."""
        icons = {
            "NEW": "✨",
            "MODIFIED": "📝",
            "UNCHANGED": "📄",
            "REMOVED": "❌",
        }
        return icons.get(status, "📄")
