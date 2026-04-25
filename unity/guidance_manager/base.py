from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Optional

from ..manager_registry import SingletonABCMeta
from ..common.global_docstrings import CLEAR_METHOD_DOCSTRING
from ..common.state_managers import BaseStateManager
from ..common.tool_outcome import ToolOutcome
from ..image_manager.types import AnnotatedImageRefs
from .types.guidance import Guidance


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

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def search(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List["Guidance"]:
        """Search for guidance entries by semantic similarity to reference content.

        Guidance entries contain procedural how-to information: step-by-step
        instructions, operating procedures, software walkthroughs, and
        strategies for composing functions together.

        Parameters
        ----------
        references : Dict[str, str] | None, default None
            Mapping of source expressions to reference text for semantic
            search.  Keys are column names or descriptive labels; values
            are the reference text to compare against.
        k : int, default 10
            Maximum number of results to return. Must be <= 1000.

        Returns
        -------
        List[Guidance]
            Up to *k* rows ranked by similarity, backfilled to *k* when
            similarity yields fewer rows.
        """
        raise NotImplementedError

    @abstractmethod
    def filter(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List["Guidance"]:
        """Filter guidance entries using a Python filter expression.

        Guidance entries contain procedural how-to information: step-by-step
        instructions, operating procedures, software walkthroughs, and
        strategies for composing functions together.

        Parameters
        ----------
        filter : str | None, default None
            A Python boolean expression evaluated with column names in
            scope (e.g. ``"guidance_id == 42"``).  When ``None``, returns
            all guidance records subject to pagination.
        offset : int, default 0
            Zero-based index of the first result to include.
        limit : int, default 100
            Maximum number of records to return. Must be <= 1000.

        Returns
        -------
        List[Guidance]
            Matching guidance records as Guidance models.
        """
        raise NotImplementedError

    @abstractmethod
    def add_guidance(
        self,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        images: Optional[AnnotatedImageRefs] = None,
        function_ids: Optional[List[int]] = None,
    ) -> "ToolOutcome":
        """Create a new guidance entry for procedural or operational how-to
        information: step-by-step instructions, standard operating procedures,
        software usage walkthroughs, composition strategies for combining
        functions, or any other actionable "how to do X" content.

        Before authoring new guidance, search existing Guidance for a rule that
        covers the same situation. If one exists, update that entry instead of
        adding a parallel rule. If the new guidance contradicts an existing
        rule, reconcile them while writing by specializing the conditions or
        replacing the outdated rule.

        At least one of ``title``, ``content``, or ``images`` must be provided.

        Parameters
        ----------
        title : str | None
            Short human-readable title for the guidance entry.
        content : str | None
            Longer freeform guidance text describing the procedure.
        images : AnnotatedImageRefs | None
            Annotated image references to attach.  Each entry pairs a
            ``RawImageRef`` (identified by ``image_id`` and/or ``filepath``)
            with a freeform ``annotation`` explaining relevance.  When a ref
            carries only a ``filepath``, the implementation resolves it to an
            ``image_id`` via ``ImageManager`` before persisting.
        function_ids : list[int] | None
            Optional ids of related functions to cross-reference.

        Returns
        -------
        ToolOutcome
            Outcome string and details containing the newly assigned
            ``guidance_id``.
        """
        raise NotImplementedError

    @abstractmethod
    def update_guidance(
        self,
        *,
        guidance_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        images: Optional[AnnotatedImageRefs] = None,
        function_ids: Optional[List[int]] = None,
    ) -> "ToolOutcome":
        """Update fields of an existing guidance entry by id.

        Use this to revise procedural instructions, operating procedures,
        or compositional strategies that are already stored. When resolving a
        contradiction or near-duplicate, prefer editing the existing guidance
        so the shared rule set stays coherent.

        Parameters
        ----------
        guidance_id : int
            Identifier of the row to update.
        title : str | None
            New title (omit to keep existing value).
        content : str | None
            New content (omit to keep existing value).
        images : AnnotatedImageRefs | None
            Replacement image references.  Each entry pairs a
            ``RawImageRef`` (identified by ``image_id`` and/or ``filepath``)
            with a freeform ``annotation``.  Filepath-only refs are resolved
            to ``image_id`` values via ``ImageManager`` before persisting.
            Omit to keep existing images.
        function_ids : list[int] | None
            Replacement list of related function ids.

        Returns
        -------
        ToolOutcome
            Outcome string and details with the ``guidance_id``.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_guidance(
        self,
        *,
        guidance_id: int,
    ) -> "ToolOutcome":
        """Delete a guidance entry by id.

        Parameters
        ----------
        guidance_id : int
            Identifier of the row to delete.

        Returns
        -------
        ToolOutcome
            Outcome string and details with the removed ``guidance_id``.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


# Attach centralised docstring
BaseGuidanceManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
