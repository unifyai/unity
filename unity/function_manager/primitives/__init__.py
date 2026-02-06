"""Primitives scoping and registry for state manager primitives.

This subpackage provides:
- `PrimitiveScope` - The single knob for controlling which managers are exposed
- `VALID_MANAGER_ALIASES` - Canonical set of valid manager aliases
- `ToolSurfaceRegistry` / `get_registry` - Central registry for manager configuration
- `ManagerSpec` - Per-manager configuration dataclass
- `Primitives` - Scoped runtime interface for accessing state manager primitives
- `ComputerPrimitives` - Computer use (web/desktop) control and reasoning
- `get_primitive_callable` - Resolve primitive metadata to callables
- `collect_primitives` / `compute_primitives_hash` - Module-level convenience functions
"""

from unity.function_manager.primitives.scope import (
    PrimitiveScope,
    VALID_MANAGER_ALIASES,
)
from unity.function_manager.primitives.registry import (
    ManagerSpec,
    ToolSurfaceRegistry,
    get_registry,
    # Module-level convenience functions (matching old API)
    collect_primitives,
    compute_primitives_hash,
    get_primitive_sources,
    _COMMON_EXCLUDED_METHODS,
    # Reverse mapping for deriving alias from primitive_class
    _CLASS_PATH_TO_ALIAS,
)
from unity.function_manager.primitives.runtime import (
    Primitives,
    ComputerPrimitives,
    DEFAULT_AGENT_SERVER_URL,
    get_primitive_callable,
    _AsyncPrimitiveWrapper,
    _create_async_wrapper,
)

__all__ = [
    # Scope
    "PrimitiveScope",
    "VALID_MANAGER_ALIASES",
    # Registry
    "ManagerSpec",
    "ToolSurfaceRegistry",
    "get_registry",
    # Module-level functions (matching old API)
    "collect_primitives",
    "compute_primitives_hash",
    "get_primitive_sources",
    "_COMMON_EXCLUDED_METHODS",
    "_CLASS_PATH_TO_ALIAS",
    # Runtime
    "Primitives",
    "ComputerPrimitives",
    "DEFAULT_AGENT_SERVER_URL",
    "get_primitive_callable",
    "_AsyncPrimitiveWrapper",
    "_create_async_wrapper",
]
