from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional

from ..conversation_manager.cm_types.medium import Medium
from ..manager_registry import SingletonABCMeta
from ..common.state_managers import BaseStateManager


class BaseBlackListManager(BaseStateManager, metaclass=SingletonABCMeta):
    """
    Public contract for a blacklist catalogue that stores and retrieves
    blocked contact details (per communication medium) with concise, primitive APIs.

    Overview
    --------
    Implementations may talk to a real database, a remote service, or an
    in-memory mock – but they all expose the same public methods documented below.

    Data Model
    ----------
    All records conform to the Pydantic model
    ``unity.blacklist_manager.types.blacklist.BlackList`` (referred to as
    "BlackList" in the method docs). Implementations may return either instances
    of this model or JSON‑serialisable dictionaries whose keys and value types
    match the model schema. The schema serves as the single source of truth for
    field names and types.

    Shared-Space Semantics
    ----------------------
    Reads include personal memory plus every accessible shared-space blacklist.
    This is a strictest-rule-wins model: if any visible root blocks a
    medium/detail pair, the contact detail should be treated as blocked. Writes
    default to personal memory. A space write blocks the contact for every
    other member of that space, so destination choice has wider consequences
    than ordinary private preferences.
    """

    _as_caller_description: str = "the BlacklistManager, managing blocked contacts"

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def filter_blacklist(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Return a paginated list of blacklist entries matching an optional filter.

        Parameters
        ----------
        filter : Optional[str]
            A Unify‑style filter expression (e.g., "medium == 'email' and contact_detail == 'spam@example.com'").
            When omitted, returns the first page of entries.
        offset : int
            Number of matching rows to skip (for pagination).
        limit : int
            Max number of rows to return.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
            - "entries": list[BlackList]
            - "blacklist_keys_to_shorthand": dict[str, str]
            - "shorthand_to_blacklist_keys": dict[str, str]
        """
        raise NotImplementedError

    @abstractmethod
    def create_blacklist_entry(
        self,
        *,
        medium: "Medium",  # Forward reference; actual type lives under transcript_manager.types.message
        contact_detail: str,
        reason: str,
        destination: str | None = None,
    ) -> Dict[str, Any]:
        """
        Create a new blacklist entry.

        Parameters
        ----------
        medium : Medium
            Communication channel (e.g., email, sms_message, phone_call).
        contact_detail : str
            The concrete contact detail to be blocked (email address, phone number, etc.).
        reason : str
            Human‑readable reason/context for the block.
        destination : str | None, default None
            Where this blacklist entry lives. Pass ``"personal"`` (the
            default) for contacts you personally want to block, such as spam
            callers or contacts you do not want to engage with individually.
            Pass ``"space:<id>"`` for an operational team-level block: a
            contact every member of the space should refuse. Strictest-rule-wins:
            an entry in any accessible root blocks the contact from your
            sessions, so a personal block does not need a space write to take
            effect for you, but a space write blocks the contact for every
            other member of that space. Read the *Accessible shared spaces*
            block before choosing. Default to personal when in doubt; call
            ``request_clarification`` for ambiguity-going-wider.

        Returns
        -------
        Dict[str, Any]
            {"outcome": "blacklist entry created", "details": {"blacklist_id": int}}
        """
        raise NotImplementedError

    @abstractmethod
    def update_blacklist_entry(
        self,
        *,
        blacklist_id: int,
        medium: Optional["Medium"] = None,
        contact_detail: Optional[str] = None,
        reason: Optional[str] = None,
        destination: str | None = None,
    ) -> Dict[str, Any]:
        """
        Update fields for an existing blacklist entry.

        At least one of ``medium``, ``contact_detail`` or ``reason`` must be provided.
        ``destination`` selects the exact BlackList root to update. Pass
        ``"personal"`` (the default) for contacts you personally want to block
        and ``"space:<id>"`` for an operational team-level block every member
        of the space should refuse. Strictest-rule-wins means an entry in any
        accessible root blocks the contact from your sessions, but updates only
        change the selected root. Read the *Accessible shared spaces* block
        before choosing; call ``request_clarification`` for ambiguity-going-wider.

        Returns
        -------
        Dict[str, Any]
            {"outcome": "blacklist entry updated", "details": {"blacklist_id": int}}
        """
        raise NotImplementedError

    @abstractmethod
    def delete_blacklist_entry(
        self,
        *,
        blacklist_id: int,
        destination: str | None = None,
    ) -> Dict[str, Any]:
        """
        Delete a blacklist entry by its identifier.
        ``destination`` selects the exact BlackList root to delete from. Pass
        ``"personal"`` (the default) for contacts you personally block and
        ``"space:<id>"`` for an operational team-level block every member of
        the space should refuse. Strictest-rule-wins means an entry in any
        accessible root blocks the contact from your sessions, but deletion
        only removes the selected root's row. Read the *Accessible shared
        spaces* block before choosing; call ``request_clarification`` for
        ambiguity-going-wider.

        Returns
        -------
        Dict[str, Any]
            {"outcome": "blacklist entry deleted", "details": {"blacklist_id": int}}
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self, *, destination: str | None = None) -> None:
        """
        Clear every blacklist entry in one destination.

        This is a destructive operation: it removes the selected root's
        blacklist rows and cannot be undone. Ask for explicit confirmation
        before clearing a personal or shared blacklist.

        destination : str | None, default None
            Which BlackList root to clear. Pass ``"personal"`` (the default)
            for contacts you personally block and ``"space:<id>"`` for an
            operational team-level block list. Strictest-rule-wins means rows
            in any accessible root block contacts from your sessions, but clear
            only removes rows from the selected root. Read the *Accessible
            shared spaces* block before choosing; call ``request_clarification``
            for ambiguity-going-wider because clearing or writing a team block
            surprises other members.
        """
        raise NotImplementedError
