"""Assistant-owned outbound communication domain.

This package contains the shared comms implementation used by both
`primitives.comms.*` and the live ConversationManager action tools.
"""

from .primitives import CommsPrimitives

__all__ = ["CommsPrimitives"]
