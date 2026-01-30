"""Pluggable execution environments for Actors.

Environments expose domain-specific tool providers (e.g. computer/web control,
state managers) in a consistent way so an Actor can inject multiple namespaces
into its execution sandbox.

This package intentionally contains only lightweight adapters and metadata.
The Actor owns proxying/caching/logging behavior.
"""

from unity.actor.environments.base import BaseEnvironment, ToolMetadata
from unity.actor.environments.computer import ComputerEnvironment
from unity.actor.environments.state_managers import StateManagerEnvironment

__all__ = [
    "BaseEnvironment",
    "ToolMetadata",
    "ComputerEnvironment",
    "StateManagerEnvironment",
]
