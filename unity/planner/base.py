# base.py
from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from typing import Callable, Dict, Generic, Optional, TypeVar

from unity.common.llm_helpers import SteerableToolHandle

__all__ = ["BasePlan", "BasePlanner"]

# --------------------------------------------------------------------------- #
# BasePlan
# --------------------------------------------------------------------------- #


class BasePlan(SteerableToolHandle, ABC):
    """
    Abstract contract that every concrete *plan* must satisfy.

    A plan represents a long-running task that can be steered at runtime
    (pause / resume / interject / ask / stop) and that ultimately resolves
    to a single result string.

    Sub-classes **must** provide concrete implementations of all abstract
    members below and expose them via ``valid_tools`` so that higher-level
    agents (or the UI) can discover the currently available controls.
    """

    # ───────────────────────────── Public API ───────────────────────────── #

    @abstractmethod
    def ask(self, question: str) -> str:
        """
        Ask any question about the live (ongoing and active) task being worked on.
        """

    @property
    @abstractmethod
    def valid_tools(self) -> Dict[str, Callable]:
        """
        Map of *public-name* ➜ *callable* for the user-accessible controls
        that are *currently* valid in the plan’s lifecycle state.
        """


# --------------------------------------------------------------------------- #
# BasePlanner
# --------------------------------------------------------------------------- #

PlanT = TypeVar("PlanT", bound=BasePlan)


class BasePlanner(Generic[PlanT], ABC):
    """
    Abstract contract that every concrete *planner* must satisfy.

    A planner is a *factory* that spawns exactly one *active* plan at a time
    (for now).  It keeps a reference to that plan so that external callers
    can query its status or steer it later.
    """

    def __init__(self) -> None:
        self._active_plan: Optional[PlanT] = None

    # ─────────────────────────── Plan management ────────────────────────── #

    def plan(
        self,
        task_description: str,
        *,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> PlanT:
        """
        Create (and start) a new plan.

        Sub-classes implement the actual creation logic in
        :meth:`_make_plan`.  This thin wrapper only enforces the
        *single-active-plan* rule and stores the reference.
        """
        if self._active_plan is not None:
            raise RuntimeError(
                "Another plan is still active. Stop it or wait for "
                "completion before starting a new one.",
            )

        plan = self._make_plan(
            task_description,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
        self._active_plan = plan
        return plan

    @abstractmethod
    def _make_plan(
        self,
        task_description: str,
        *,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> PlanT:
        """
        Concrete planner must build **and start** a plan implementation
        (e.g. ``SimulatedPlan``) and return it.
        """

    # ────────────────────────── Convenience API ─────────────────────────── #

    @property
    def active_plan(self) -> Optional[PlanT]:
        """Return the currently running plan (or *None* if idle)."""
        return self._active_plan

    def clear_active_plan(self) -> None:
        """Forget the active plan (useful once it has completed)."""
        self._active_plan = None
