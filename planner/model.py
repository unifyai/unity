"""
Model module for the planner.

This module defines the core data structure for representing atomic actions:
- Primitive: An atomic action that can be executed
"""

from typing import Any, Dict
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
