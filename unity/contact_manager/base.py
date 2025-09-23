# contact_manager/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from ..common.llm_helpers import SteerableToolHandle
from ..singleton_registry import SingletonABCMeta


class BaseContactManager(ABC, metaclass=SingletonABCMeta):
    """
    *Public* contract that every concrete **contact-manager** must satisfy.

    A contact-manager answers questions (`ask`) about stored contacts and
    handles English instructions (`update`) that create or change those
    contacts.  Implementations may talk to a real database, a remote CRM,
    an in-memory mock, or even a purely simulated LLM – but they **all**
    expose exactly the two public methods documented below.
    """

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
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

        Do *not* request *how* the question should be answered; just ask the
        question in natural language and allow this `ask` method to determine
        the best method to answer it.

        Examples
        --------
        • Good: "Who is the contact living in Berlin working as a product designer?"
          → identify the ``contact_id`` so an update tool can be applied next.
        • Bad:  "What surname should I use for the new contact I'm about to create?"
          → this is a human clarification; use ``request_clarification`` instead.

        Parameters
        ----------
        text : str
            The user's plain-English question (e.g. *"Show me Alice's phone
            number."*).
        _return_reasoning_steps : bool, default ``False``
            When *True*, :pyfunc:`SteerableToolHandle.result` returns a
            tuple ``(answer, messages)`` where *messages* is the invisible
            chain-of-thought exchanged with the LLM.
        parent_chat_context : list[dict] | None
            **Read-only** conversation context to prepend to the tool loop.
        clarification_up_q / clarification_down_q : asyncio.Queue[str] | None
            Optional duplex channels.  When supplied the LLM can ask the human
            follow-up questions via *up_q* and must read answers from
            *down_q*.

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
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> SteerableToolHandle:
        """
        Apply a **mutation** request – create, edit, delete or merge contacts –
        expressed in plain English and receive a steerable LLM handle.

        Do *not* request *how* the change should be implemented; describe the
        desired end‑state in natural language and allow the `update` method to
        determine the best method to apply it.

        Ask vs Clarification
        --------------------
        • `ask` is ONLY for inspecting/locating contacts that ALREADY EXIST (e.g.,
          to find ``contact_id`` or verify stored fields).
        • Do NOT use `ask` to ask the human for details about NEW contacts being
          created/changed in this update request; call ``request_clarification``
          when a clarification channel is available.
        • When no clarification tool exists, proceed with sensible defaults or
          best‑guess values and state those assumptions in the final reply.

        Parameters
        ----------
        text : str
            The user's request (e.g. *"Add Sarah Connor's phone number …"*).
        _return_reasoning_steps, parent_chat_context,
        clarification_up_q, clarification_down_q
            Same semantics as in :py:meth:`ask`.

        Returns
        -------
        SteerableToolHandle
            Handle whose :pyfunc:`result` yields confirmation of the mutation
            and (optionally) reasoning steps.
        """

    @abstractmethod
    def _filter_contacts(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List["Contact"]:
        """
        Retrieve contact records that satisfy *filter*.

        This private method is intentionally *part* of the public-facing contract
        because other managers (e.g. :class:`~unity.transcript_manager.TranscriptManager`)
        rely on its existence for tool-chaining.  Concrete subclasses **must**
        implement it – even simulated ones – so that the LLM can access a
        deterministic search primitive.

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
    def _update_contact(
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
        contract just like :pyfunc:`_search_contacts`.  Other managers –
        notably :class:`~unity.memory_manager.MemoryManager` – rely on its
        presence for fast, deterministic updates without a full natural-
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

    @abstractmethod
    def _delete_contact(
        self,
        *,
        contact_id: int,
    ) -> "ToolOutcome":
        """
        Permanently **remove** a contact from storage.

        Parameters
        ----------
        contact_id : int
            Identifier of the contact to delete.

        Returns
        -------
        ToolOutcome
            A standard outcome payload summarising what was deleted (at minimum the contact_id).
        """
        raise NotImplementedError


if TYPE_CHECKING:
    # Avoid a runtime import to prevent circular dependencies
    from .types.contact import Contact
    from ..common.tool_outcome import ToolOutcome
