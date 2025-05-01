"""
Model module for the planner.

This module defines the core data structures for representing and executing plans:
- Primitive: An atomic action that can be executed
- FunctionNode: A function call with a body of primitives or other function nodes
- Plan: The overall execution plan with methods to traverse and execute the plan
"""

from typing import List, Union, Optional, Callable, Any, Dict
import asyncio
from asyncio import Queue
from .context import context


class Primitive:
    """
    Represents an atomic action that can be executed.

    Attributes:
        name: The name of the primitive action
        args: Arguments for the primitive action
        call_literal: String representation of the function call
    """

    def __init__(self, name: str, args: Dict[str, Any], call_literal: str):
        """
        Initialize a Primitive action.

        Args:
            name: The name of the primitive action
            args: Arguments for the primitive action
            call_literal: String representation of the function call
        """
        self.name = name
        self.args = args
        self.call_literal = call_literal

    def __repr__(self) -> str:
        """String representation of the Primitive."""
        return f"Primitive({self.name}, {self.args}, {self.call_literal})"


class FunctionNode:
    """
    Represents a function call with a body of primitives or other function nodes.

    Attributes:
        name: The name of the function
        body: List of Primitive or FunctionNode objects that make up the function body
        parent: Optional parent FunctionNode
        cursor: Current position in the body during execution
    """

    def __init__(
        self,
        name: str,
        body: List[Union["FunctionNode", Primitive]],
        parent: Optional["FunctionNode"] = None,
        cursor: int = 0,
    ):
        """
        Initialize a FunctionNode.

        Args:
            name: The name of the function
            body: List of Primitive or FunctionNode objects that make up the function body
            parent: Optional parent FunctionNode
            cursor: Current position in the body during execution
        """
        self.name = name
        self.body = body
        self.parent = parent
        self.cursor = cursor

    def __repr__(self) -> str:
        """String representation of the FunctionNode."""
        return (
            f"FunctionNode({self.name}, body[{len(self.body)}], cursor={self.cursor})"
        )

    def is_complete(self) -> bool:
        """Check if all items in the body have been executed."""
        return self.cursor >= len(self.body)


class Plan:
    """
    Represents the overall execution plan with methods to traverse and execute the plan.

    Attributes:
        root: The root FunctionNode of the plan
        current_node: The FunctionNode currently being executed
        completion_callback: Optional callback to be called when the plan is complete
        task_completion_q: Queue to signal task completion
    """

    def __init__(self, root: FunctionNode, task_completion_q: Optional[Queue] = None):
        """
        Initialize a Plan.

        Args:
            root: The root FunctionNode of the plan
            task_completion_q: Queue to signal task completion
        """
        self.root = root
        self.current_node = root
        self.task_completion_q = task_completion_q
        self._expected_primitive = None  # Store the expected primitive for validation

    def ready(self) -> bool:
        """
        Check if the plan has more actions to execute.

        Returns:
            bool: True if there are more actions to execute, False otherwise
        """
        # Plan is ready if the root node is not complete and there's no outstanding expected primitive
        return not self.root.is_complete() and self._expected_primitive is None

    def next_action(self) -> Optional[str]:
        """
        Get the next primitive action to execute.

        Returns:
            str: The command string of the next primitive action to execute, or None if the plan is complete
        """
        if not self.ready():
            return None

        # Find the next primitive action by traversing the tree
        while True:
            if self.current_node.cursor >= len(self.current_node.body):
                # Current node is complete, go back to parent
                if self.current_node.parent is None:
                    # We're at the root and it's complete
                    return None
                self.current_node = self.current_node.parent
                continue

            current_item = self.current_node.body[self.current_node.cursor]

            if isinstance(current_item, Primitive):
                # Capture state *before* issuing the command
                self._prev_snapshot = context.last_state_snapshot()
                # Found a primitive to execute, store it as the expected primitive
                self._expected_primitive = current_item
                return current_item.call_literal
            elif isinstance(current_item, FunctionNode):
                # Descend into the function node
                current_item.parent = self.current_node
                self.current_node = current_item
            else:
                raise TypeError(f"Unexpected item type in plan: {type(current_item)}")

    def mark_action_done(self, primitive_literal: str) -> None:
        from .verifier import Verifier

        """
        Mark the current action as completed and advance to the next action.
        
        Args:
            primitive_literal: The literal string of the primitive action that was completed
        """
        if not hasattr(self, "_expected_primitive") or self._expected_primitive is None:
            print(f"Warning: No expected action to mark as done")
            return

        # Normalize literals by removing quotes and parentheses for fuzzy matching
        def normalize(s: str) -> str:
            return s.strip().strip("()").strip("\"'")

        normalized_completed = normalize(primitive_literal)
        normalized_expected = normalize(self._expected_primitive.call_literal)

        # Fuzzy match - check if normalized strings are similar enough
        if (
            normalized_completed != normalized_expected
            and normalized_completed not in normalized_expected
        ):
            # The completed action doesn't match what we expected
            print(
                f"Warning: Completed action '{primitive_literal}' doesn't match expected action '{self._expected_primitive.call_literal}'"
            )
            return

        # Advance the cursor in the current node
        self.current_node.cursor += 1

        # Clear the expected primitive
        self._expected_primitive = None

        # ---- Verification -------------------------------------------------
        after_snapshot = context.last_state_snapshot()
        # Determine current function context name (node)
        fn_name = self.current_node.name if self.current_node else "root"
        try:
            # FIXME: 'src' is a function name until the AST-based path is removed
            verdict = Verifier.check(
                src=fn_name,
                before=getattr(self, "_prev_snapshot", None),
                after=after_snapshot,
                args=None,
                kwargs=None,
            )
            if verdict == "reimplement":
                Verifier.get_reimplement_queue().put(fn_name)
        except Exception:
            pass

        # Advance pointers and check for completion
        self._advance_pointers()

    def _advance_pointers(self) -> None:
        """
        Update cursors after action completion and handle completed nodes.
        """
        # Check if current node is complete
        while self.current_node.is_complete():
            if self.current_node.parent is None:
                # Root node is complete, plan is done
                return

            # Move up to parent node
            parent = self.current_node.parent
            parent.cursor += 1  # Move parent's cursor past the completed child node
            self.current_node = parent

            # Check if parent is now complete
            if self.current_node.is_complete() and self.current_node.parent is None:
                # Root node is complete, plan is done
                return

    def get_function_by_name(self, name: str) -> Optional[FunctionNode]:
        """Return the first FunctionNode in the plan that matches *name*.

        The search is depth-first over the *root* tree. Returns *None* if no
        node with that name exists.
        """

        def _dfs(node: FunctionNode) -> Optional[FunctionNode]:
            if node.name == name:
                return node
            for item in node.body:
                if isinstance(item, FunctionNode):
                    found = _dfs(item)
                    if found is not None:
                        return found
            return None

        return _dfs(self.root)
