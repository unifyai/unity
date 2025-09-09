from typing import TYPE_CHECKING
from importlib import import_module

__all__ = ["BrowserUseActor", "ToolLoopActor", "HierarchicalActor", "CodeActActor"]

_lazy_map = {
    "BrowserUseActor": "unity.actor.browser_use_actor",
    "ToolLoopActor": "unity.actor.tool_loop_actor",
    "HierarchicalActor": "unity.actor.hierarchical_actor",
    "CodeActActor": "unity.actor.code_act_actor",
}


def __getattr__(name: str):
    if name in _lazy_map:
        module = import_module(_lazy_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:
    # Imports for type checkers / IDEs only; avoided at runtime to prevent side effects
    from .browser_use_actor import BrowserUseActor
    from .tool_loop_actor import ToolLoopActor
    from .hierarchical_actor import HierarchicalActor
    from .code_act_actor import CodeActActor
