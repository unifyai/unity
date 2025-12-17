"""
JSON-friendly type aliases.

We use these aliases to avoid `Any` in the strict parse boundary models while
still allowing flexible, JSON-serializable metadata and error details.
"""

from __future__ import annotations

from typing import Dict, TypeAlias

from pydantic import JsonValue as _PydanticJsonValue

# JSON primitives
JsonScalar: TypeAlias = str | int | float | bool | None

# Recursive JSON value (Pydantic-provided, schema-aware).
#
# NOTE:
# We intentionally use Pydantic's built-in JsonValue instead of a self-referential
# TypeAlias to avoid `model_rebuild()` issues in Pydantic v2.
JsonValue: TypeAlias = _PydanticJsonValue

# Convenience alias for objects
JsonObject: TypeAlias = Dict[str, JsonValue]
