from __future__ import annotations

from abc import abstractmethod

from ..manager_registry import SingletonABCMeta
from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING
from ..common.state_managers import BaseStateManager


class BaseGuidanceManager(BaseStateManager, metaclass=SingletonABCMeta):
    """
    Public contract that every concrete guidance-manager must satisfy.

    Exposes CRUD operations on guidance entries (search, filter,
    add_guidance, update_guidance, delete_guidance) but no high-level
    tool-loop methods.  Read and mutation orchestration is handled
    externally by the CodeActActor via top-level JSON tool calls.
    """

    _as_caller_description: str = (
        "the GuidanceManager, managing assistant guidance and policies"
    )

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


# Attach centralised docstring
BaseGuidanceManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
