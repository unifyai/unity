from __future__ import annotations

from abc import abstractmethod

from ..manager_registry import SingletonABCMeta
from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING
from ..common.state_managers import BaseStateManager


class BaseGuidanceManager(BaseStateManager, metaclass=SingletonABCMeta):
    """
    Public contract that every concrete guidance-manager must satisfy.

    Stores procedural how-to information: step-by-step instructions,
    standard operating procedures, software usage walkthroughs, and
    strategies for composing functions together.

    Exposes CRUD operations (search, filter, add_guidance,
    update_guidance, delete_guidance) as first-class JSON tool calls
    on the CodeActActor — both in the main doing loop and in the
    post-completion storage review loop.
    """

    _as_caller_description: str = (
        "the GuidanceManager, managing procedural instructions and operating procedures"
    )

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


# Attach centralised docstring
BaseGuidanceManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
