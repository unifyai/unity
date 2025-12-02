from typing import TYPE_CHECKING
from importlib import import_module

__all__ = ["HierarchicalActor", "CodeActActor", "Plan"]

_lazy_map = {
    "HierarchicalActor": "unity.actor.hierarchical_actor",
    "CodeActActor": "unity.actor.code_act_actor",
    "Plan": "unity.actor.plan",
}


def __getattr__(name: str):
    if name in _lazy_map:
        module = import_module(_lazy_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:
    from .hierarchical_actor import HierarchicalActor
    from .code_act_actor import CodeActActor
    from .plan import Plan
