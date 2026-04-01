# contact_manager/base.py
from __future__ import annotations

from abc import abstractmethod
import asyncio
from typing import Dict, List, Optional, Any, Type, TYPE_CHECKING
from pydantic import BaseModel

from ..common.async_tool_loop import SteerableToolHandle
from ..manager_registry import SingletonABCMeta
from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING
from ..common.state_managers import BaseStateManager


class BaseContactManager(BaseStateManager, metaclass=SingletonABCMeta):
    """
    *Public* contract that every concrete **contact-manager** must satisfy.

    A contact-manager answers questions (`ask`) about stored contacts and
    handles English instructions (`update`) that create or change those
    contacts.  Implementations may talk to a real database, a remote CRM,
    an in-memory mock, or even a purely simulated LLM – but they **all**
    expose exactly the two public methods documented below.
    """

    _as_caller_description: str = (
        "the ContactManager, managing contact records on behalf of the end user"
    )

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def ask(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Interrogate the **existing contact list** (read‑only) and obtain a live
        :class:`SteerableToolHandle`.

        Purpose
        -------
        Use this method to locate and inspect contacts that already exist in the
        table: find ``contact_id`` values, check emails/phone numbers, look up
        attributes or summarise/compare existing entries. This call must never
        create, modify or delete contacts.

        Clarifications
        --------------
        Do not use this method to ask the human questions. If the caller needs
        clarification about a prospective/new contact (e.g., correct spelling,
        missing fields, preferred channel), route the question via a dedicated
        ``request_clarification`` tool when available. If no clarification
        channel exists, proceed with sensible defaults/best‑guess values and
        state those assumptions in the outer loop's final reply.

        Do not request how the question should be answered; ask in natural
        language and allow this method to determine the best approach.

        Examples
        --------
        • Good: "Who is the contact living in Berlin working as a product designer?"
          → identify the ``contact_id`` so an update tool can be applied next.
        • Bad:  "What surname should I use for the new contact I'm about to create?"
          → this is a human clarification; use ``request_clarification`` instead.

        Parameters
        ----------
        text : str
            The user's plain-English question (e.g. "Show me Alice's phone number.").
        response_format : Type[BaseModel] | None, default ``None``
            Optional Pydantic model to request a structured answer. When provided,
            the final result should conform to this schema; otherwise a plain
            string answer is returned.
        _return_reasoning_steps : bool, default ``False``
            When True, :pyfunc:`SteerableToolHandle.result` returns a
            tuple ``(answer, messages)`` where *messages* is the invisible
            chain-of-thought exchanged with the LLM.
        _parent_chat_context : list[dict] | None
            Read-only conversation context to prepend to the tool loop.
        _clarification_up_q / _clarification_down_q : asyncio.Queue[str] | None
            Optional duplex channels. When supplied the LLM can ask the human
            follow-up questions via *up_q* and must read answers from *down_q*.

        Returns
        -------
        SteerableToolHandle
            A live handle that ultimately yields the assistant's answer and
            exposes steering operations (``pause``, ``resume``, ``interject``,
            ``stop``).
        """

    @abstractmethod
    async def update(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Apply a **mutation** request – create, edit, delete or merge contacts –
        expressed in plain English and receive a steerable LLM handle.

        Do not request how the change should be implemented; describe the
        desired end‑state in natural language and allow this method to
        determine the best approach.

        Ask vs Clarification
        --------------------
        • ``ask`` is ONLY for inspecting/locating contacts that already exist (e.g.,
          to find ``contact_id`` or verify stored fields).
        • Do NOT use ``ask`` to ask the human for details about new contacts being
          created/changed in this update request; call ``request_clarification``
          when a clarification channel is available.
        • When no clarification tool exists, proceed with sensible defaults or
          best‑guess values and state those assumptions in the final reply.

        Nameless Contacts
        -----------------
        Contacts may legitimately have ``first_name`` and ``surname`` set to
        ``None``.  Whether to populate the name depends on what the contact
        *represents*, which is conveyed by the ``bio`` and surrounding context:

        • **Organisation / service contacts** – a support hotline, a help-desk
          email, a company switchboard, etc.  The contact detail belongs to the
          *entity*, not to any individual.  ``first_name`` and ``surname`` should
          stay ``None``; the ``bio`` should describe the organisation or service
          (e.g. "Acme Corp billing support line").  Individual names encountered
          during a specific interaction (e.g. "Hi, this is Sarah from Billing")
          are transient representatives and must **not** be written into the
          contact's name fields.
        • **Unknown-name personal contacts** – a real person whose name simply
          hasn't been discovered yet (e.g. "call my friend at this number").
          Here, ``first_name`` / ``surname`` **should** be populated as soon as
          the name becomes known.

        Use the ``bio``, ``response_policy``, and any available context to
        determine which case applies before deciding whether to set or leave
        the name fields.

        Parameters
        ----------
        text : str
            The user's request (e.g. "Add Sarah Connor's phone number …").
        response_format : Type[BaseModel] | None, default ``None``
            Optional Pydantic model to request a structured outcome. When provided,
            the final result should conform to this schema; otherwise a plain
            string summary is returned.
        _return_reasoning_steps, _parent_chat_context,
        _clarification_up_q, _clarification_down_q
            Same semantics as in :py:meth:`ask`.

        Returns
        -------
        SteerableToolHandle
            Handle whose :pyfunc:`result` yields confirmation of the mutation
            and (optionally) reasoning steps.
        """

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def filter_contacts(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List["Contact"]:
        """
        Retrieve contact records that satisfy *filter*.

        This private method is intentionally *part* of the public-facing contract
        because other components may rely on its existence for tool‑chaining.
        Concrete subclasses **must** implement it – even simulated ones – so that
        the LLM can access a deterministic search primitive.

        Parameters
        ----------
        filter : str | None, default ``None``
            Python expression evaluated against every contact (``None`` selects all).
        offset : int, default ``0``
            Zero-based index of the first result to return.
        limit : int, default ``100``
            Maximum number of contacts to return.

        Returns
        -------
        list[Contact]
            Matching contacts in creation order.
        """
        raise NotImplementedError

    @abstractmethod
    def update_contact(
        self,
        *,
        contact_id: int,
        first_name: Optional[str] = None,
        surname: Optional[str] = None,
        email_address: Optional[str] = None,
        phone_number: Optional[str] = None,
        whatsapp_number: Optional[str] = None,
        description: Optional[str] = None,
        bio: Optional[str] = None,
        rolling_summary: Optional[str] = None,
        custom_fields: Optional[Dict[str, Any]] = None,
    ) -> "ToolOutcome":
        """
        Modify **one** existing contact identified by *contact_id*.

        Although private, this helper is *part* of the public-facing
        contract (alongside the filtering helper). Downstream components may rely
        on its presence for fast, deterministic updates without a full natural-
        language round-trip through :pyfunc:`update`.

        Concrete subclasses **must** supply a *synchronous* implementation so
        that it can safely be invoked inside an ``asyncio.to_thread`` call.

        Parameters
        ----------
        contact_id : int
            The unique ``contact_id`` of the record to update.
        first_name, surname, email_address, phone_number, whatsapp_number,
        description, bio, rolling_summary, custom_fields
            Same semantics as the public :pyfunc:`update` method.

        Returns
        -------
        ToolOutcome
            A standard outcome payload summarising what changed.  Must be
            non-empty so that simulated managers can fabricate realistic
            confirmations.
        """
        raise NotImplementedError


if TYPE_CHECKING:
    # Avoid a runtime import to prevent circular dependencies
    from .types.contact import Contact
    from ..common.tool_outcome import ToolOutcome


# Attach centralised docstring
BaseContactManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
