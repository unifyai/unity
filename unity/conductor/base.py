"""
*Public* contract for every **Conductor** implementation.

The top-level manager unifies four sub-domains

• tasks  • contacts  • transcripts  • knowledge

and it exposes exactly **two** conversational entry-points:

1. `request`    – unified read-write orchestrator across all domains
2. `start_task` – immediately start execution of a queued task (returns a live handle)
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..common.async_tool_loop import SteerableToolHandle
from ..manager_registry import SingletonABCMeta
from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING
from ..common.state_managers import BaseStateManager
from .types import StateManager

if TYPE_CHECKING:  # type hints only
    from ..image_manager.types.image_refs import ImageRefs


class BaseConductor(BaseStateManager, metaclass=SingletonABCMeta):
    _as_caller_description: str = (
        "the Conductor, orchestrating work across managers on behalf of the end user"
    )

    # ------------------------------------------------------------------ #
    #  request – unified read/write orchestrator                          #
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def request(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _log_tool_steps: bool = True,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Unified orchestrator for all Conductor interactions.

        Use this for any question, mutation, or execution. Describe the desired
        outcome in natural language; the Conductor determines which managers
        and tools to apply.

        Parameters
        ----------
        text : str
            The exact user question or request (natural language).
        _return_reasoning_steps : bool, default ``False``
            When *True*, the handle's ``.result()`` yields
            ``(assistant_answer, hidden_messages)`` instead of just the answer.
        _log_tool_steps : bool, default ``True``
            Emit server-side logs for each internal tool call (debugging aid).
        _parent_chat_context : list[dict] | None
            Optional **read-only** context inherited from a parent conversation
            and made visible to the inner tool loop.
        _clarification_up_q / _clarification_down_q : asyncio.Queue[str] | None
            Two-way channels enabling interactive clarification questions:
            the LLM places a question on *up* and blocks waiting for the human
            answer on *down*.

        Returns
        -------
        SteerableToolHandle
            Await ``handle.result()`` for the final answer or steer execution
            mid-flight via ``pause()``, ``resume()``, ``interject()`` or
            ``stop()``.
        """

    # ------------------------------------------------------------------ #
    #  start_task – immediately start execution of a queued task         #
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def start_task(
        self,
        task_id: int,
        trigger_reason: str,
    ) -> SteerableToolHandle:
        """
        Start execution of an existing queued task identified by ``task_id`` and
        return a live, steerable handle to the running session.

        This method is intended for scheduler- or event-driven starts where a task
        should begin immediately without an initial conversational turn. The returned
        handle exposes the standard steering surface:
        ``pause()``, ``resume()``, ``interject(message)``, ``stop()``, ``done()``,
        as well as ``result()`` and history accessors.

        Parameters
        ----------
        task_id : int
            Identifier of the existing task to start.
        trigger_reason : str
            Short human-readable description of why the task is starting now
            (e.g., "scheduled time reached", "external trigger").

        Returns
        -------
        SteerableToolHandle
            A live handle representing the running execution, suitable for
            awaiting completion or steering mid-flight.
        """

    # ------------------------------------------------------------------ #
    #  pause_actor / resume_actor – steer any in‑flight interactive run   #
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def pause_actor(
        self,
        reason: str,
        images: "ImageRefs | list | None" = None,
    ) -> dict:
        """
        Pause any in‑flight interactive execution started via this Conductor and
        announce the pause.

        Behaviour
        ---------
        - If a live interactive session is active, applies a pause to the
          current child execution and emits a concise interjection explaining
          the reason for the pause.
        - If no interactive session is active, returns a benign no‑op summary.
        - When images are provided, they are forwarded alongside the interjection(s).

        Parameters
        ----------
        reason : str
            Human‑readable reason attached to the interjection(s).
        images : optional
            Optional image references to forward with interjections.

        Returns
        -------
        dict
            Summary object describing which operations were applied or skipped.
        """

    @abstractmethod
    async def resume_actor(
        self,
        reason: str,
        images: "ImageRefs | list | None" = None,
    ) -> dict:
        """
        Resume any in‑flight interactive execution started via this Conductor and
        announce the resume.

        Behaviour
        ---------
        - If a live interactive session is active, emits a concise interjection
          explaining the reason and then resumes the current child execution.
        - If no interactive session is active, returns a benign no‑op summary.
        - When images are provided, they are forwarded alongside the interjection(s).

        Parameters
        ----------
        reason : str
            Human‑readable reason attached to the interjection(s).
        images : optional
            Optional image references to forward with interjections.

        Returns
        -------
        dict
            Summary object describing which operations were applied or skipped.
        """

    # ------------------------------------------------------------------ #
    #  clear – irreversible state wipe for a selected manager            #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def clear(self, target: StateManager) -> None:
        """
        {base}

        Parameters
        ----------
        target : StateManager
            Which manager to clear. Options include: CONTACTS, TRANSCRIPTS, KNOWLEDGE,
            TASKS, WEB_SEARCH, FILES and forward-compat entries FUNCTIONS, GUIDANCE, IMAGES, SECRETS.
        """


BaseConductor.clear.__doc__ = (BaseConductor.clear.__doc__ or "").format(
    base=CLEAR_METHOD_DOCSTRING,
)
