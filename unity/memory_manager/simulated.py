# memory_manager/simulated.py
"""
A lightweight, *offline-only* stand-in for the real `MemoryManager`.

It keeps an **internal, in-memory dictionary** of "contacts" so that calls to
`update_contact_bio` and `update_contact_rolling_summary` appear to mutate
state across invocations – but nothing ever touches an external store.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Callable, Any

import unify

# ── new helpers & simulated back-ends ────────────────────────────────────────
from ..contact_manager.simulated import SimulatedContactManager
from ..transcript_manager.simulated import SimulatedTranscriptManager
from ..knowledge_manager.simulated import SimulatedKnowledgeManager
from ..common.llm_helpers import (
    methods_to_tool_dict,
    start_async_tool_use_loop,
)
from . import prompt_builders as pb
from .base import BaseMemoryManager


class SimulatedMemoryManager(BaseMemoryManager):
    """
    Test-double that **really uses** the simulated contact & transcript
    managers instead of hallucinating everything from scratch.  Still
    returns *plain strings* (no steerable handles).
    """

    def __init__(self, description: str = "imaginary scenario") -> None:
        # ── plug into the *other* simulated services so state is shared ─────
        self._contact_manager = SimulatedContactManager(description=description)
        self._transcript_manager = SimulatedTranscriptManager(description=description)
        self._knowledge_manager = SimulatedKnowledgeManager(description=description)

        # Light-weight overlay that remembers the *latest* bio / rolling / knowledge writes
        # without touching an external store – key = contact_id
        self._overlays: Dict[int, Dict[str, str]] = {}

        # One shared, stateful LLM that orchestrates tool-use loops
        self._llm = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    async def update_contacts(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str:  # noqa: D401
        """
        Pretend to parse the transcript and add / update contacts.
        Simply returns a short, human-readable summary.
        """
        # Build a *dynamic* tool-set: contact ask / update  +  transcript ask
        tools: Dict[str, Callable[..., Any]] = methods_to_tool_dict(
            self._contact_manager.ask,
            self._contact_manager.update,
            self._transcript_manager.ask,
            include_class_name=True,
        )

        self._llm.set_system_message(
            pb.build_contact_update_prompt(tools, guidance=guidance),
        )

        handle = start_async_tool_use_loop(
            self._llm,
            transcript,
            tools,
            loop_id="SimulatedMemoryManager.update_contacts",
            tool_policy=lambda i, _: ("required", _) if i < 2 else ("auto", _),
        )

        # returns a **string** – the tool-loop terminates internally
        return await handle.result()

    async def update_contact_bio(
        self,
        transcript: str,
        latest_bio: Optional[str] = None,
        guidance: Optional[str] = None,
    ) -> str:
        """
        Fabricates a new bio (or keeps the old one) and stores it in RAM.
        """

        # --- scoped mutator --------------------------------------------------
        async def set_bio(contact_id: int, bio: str) -> str:
            # overlay for the test's benefit
            self._overlays.setdefault(contact_id, {})["bio"] = bio
            return f"Bio for contact with id {contact_id} successfully updated"

        tools: Dict[str, Callable[..., Any]] = {
            "transcript_ask": self._transcript_manager.ask,
            "contact_ask": self._contact_manager.ask,
            "set_bio": set_bio,
        }

        self._llm.set_system_message(pb.build_bio_prompt(tools, guidance=guidance))

        payload = json.dumps(
            {
                "latest_bio": latest_bio,
                "transcript": transcript,
            },
            indent=2,
        )

        handle = start_async_tool_use_loop(
            self._llm,
            payload,
            tools,
            loop_id="SimulatedMemoryManager.update_contact_bio",
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
        )
        return await handle.result()

    async def update_contact_rolling_summary(
        self,
        transcript: str,
        latest_rolling_summary: Optional[str] = None,
        guidance: Optional[str] = None,
    ) -> str:
        """
        Generates a fresh ≤120-word rolling summary and stores it in RAM.
        """

        async def set_rolling_summary(contact_id: int, rolling_summary: str) -> str:
            self._overlays.setdefault(contact_id, {})[
                "rolling_summary"
            ] = rolling_summary
            await self._contact_manager._update_contact(
                contact_id=contact_id,
                custom_fields={"rolling_summary": rolling_summary},
            )
            return "rolling_summary updated"

        tools: Dict[str, Callable[..., Any]] = {
            "transcript_ask": self._transcript_manager.ask,
            "contact_ask": self._contact_manager.ask,
            "set_rolling_summary": set_rolling_summary,
        }

        self._llm.set_system_message(pb.build_rolling_prompt(tools, guidance=guidance))

        payload = json.dumps(
            {
                "latest_rolling_summary": latest_rolling_summary,
                "transcript": transcript,
            },
            indent=2,
        )

        handle = start_async_tool_use_loop(
            self._llm,
            payload,
            tools,
            loop_id="SimulatedMemoryManager.update_contact_rolling_summary",
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
        )
        return await handle.result()

    async def update_knowledge(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str:
        """
        Pass transcript through a tool-loop wired to the simulated
        KnowledgeManager; store any harvested facts in `_overlays['kb']`
        for rudimentary statefulness.
        """

        async def _kb_update(card_id: int | None, content: str) -> str:
            """
            Tiny wrapper that calls the simulated `.update` and records the
            content locally so tests can assert changes between calls.
            """
            self._overlays.setdefault("kb", []).append(content)
            return await self._knowledge_manager.update(
                f"Update knowledge card {card_id}: {content}",
            )

        tools: Dict[str, Callable[..., Any]] = {
            "contact_ask": self._contact_manager.ask,
            "transcript_ask": self._transcript_manager.ask,
            "kb_ask": self._knowledge_manager.ask,
            "kb_refactor": self._knowledge_manager.refactor,
            "kb_update": _kb_update,
        }

        self._llm.set_system_message(
            pb.build_knowledge_prompt(tools, guidance=guidance),
        )

        handle = start_async_tool_use_loop(
            self._llm,
            transcript,
            tools,
            loop_id="SimulatedMemoryManager.update_knowledge",
            tool_policy=lambda i, _: ("required", _) if i < 2 else ("auto", _),
        )
        return await handle.result()
