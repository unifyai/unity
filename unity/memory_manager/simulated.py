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
import asyncio

# ── new helpers & simulated back-ends ────────────────────────────────────────
from ..contact_manager.simulated import SimulatedContactManager
from ..transcript_manager.simulated import SimulatedTranscriptManager
from ..knowledge_manager.simulated import SimulatedKnowledgeManager
from ..task_scheduler.simulated import SimulatedTaskScheduler
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
        self._task_scheduler = SimulatedTaskScheduler(description=description)

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
        *,
        contact_id: int,
        guidance: Optional[str] = None,
    ) -> str:
        """
        Fabricates a new bio (or keeps the old one) and stores it in RAM.

        The method now **fetches** the latest bio for the given contact id so
        callers are not required to provide it explicitly.
        """

        # --- scoped mutator --------------------------------------------------
        target_id = contact_id

        async def set_bio(contact_id: int, bio: str) -> str:
            final_id = contact_id or target_id
            if final_id is None:
                raise ValueError(
                    "contact_id must be provided either via argument or tool call.",
                )
            self._overlays.setdefault(final_id, {})["bio"] = bio
            handle = await self._contact_manager.update(
                f"Please set the bio for contact id {final_id} as follows:\n{bio}",
            )
            return await handle.result()

        # ------------------------------------------------------------------
        # Ensure the assistant (contact_id 0) is always described in 2nd person
        assistant_extra = (
            "IMPORTANT: This bio belongs to the assistant itself (contact_id 0). Always use **second-person** pronouns ('you') when describing the assistant. Never refer in the third person."  # noqa: E501
            if contact_id == 0
            else None
        )
        combined_guidance: Optional[str] = (
            "\n".join(g for g in (guidance, assistant_extra) if g) or None
        )

        tools: Dict[str, Callable[..., Any]] = {
            "transcript_ask": self._transcript_manager.ask,
            "contact_ask": self._contact_manager.ask,
            "set_bio": set_bio,
        }

        # Retrieve contact info for clearer context (run in thread to avoid nested loop)
        contacts = await asyncio.to_thread(
            self._contact_manager._search_contacts,
            filter=f"contact_id == {contact_id}",
            limit=1,
        )
        c0 = contacts[0] if contacts else None
        latest_bio_val = c0.bio
        contact_name_val = (
            " ".join(p for p in [c0.first_name, c0.surname] if p).strip() or None
        )

        self._llm.set_system_message(
            pb.build_bio_prompt(
                f"{contact_name_val} (id {contact_id})",
                tools,
                guidance=combined_guidance,
            ),
        )

        payload_dict = {
            "contact_id": contact_id,
            "latest_bio": latest_bio_val,
            "transcript": transcript,
        }
        if contact_name_val is not None:
            payload_dict["contact_name"] = contact_name_val

        payload = json.dumps(payload_dict, indent=2)

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
        *,
        contact_id: int,
        guidance: Optional[str] = None,
    ) -> str:
        """
        Generates a fresh ≤120-word rolling summary and stores it in RAM.

        The method now fetches the current rolling summary automatically so
        callers don’t need to provide it explicitly.
        """

        # --- scoped mutator --------------------------------------------------
        target_id = contact_id

        async def set_rolling_summary(contact_id: int, rolling_summary: str) -> str:
            final_id = contact_id or target_id
            if final_id is None:
                raise ValueError(
                    "contact_id must be provided either via argument or tool call.",
                )
            self._overlays.setdefault(final_id, {})["rolling_summary"] = rolling_summary
            handle = await self._contact_manager.update(
                f"Please set the bio for contact id {final_id} as follows:\n{rolling_summary}",
            )
            return await handle.result()

        tools: Dict[str, Callable[..., Any]] = {
            "transcript_ask": self._transcript_manager.ask,
            "contact_ask": self._contact_manager.ask,
            "set_rolling_summary": set_rolling_summary,
        }

        # Retrieve contact details (name + current rolling summary) – thread-safe
        # ----------------------------------------------------------------
        try:
            contacts = await asyncio.to_thread(
                self._contact_manager._search_contacts,
                filter=f"contact_id == {contact_id}",
                limit=1,
            )
            c0 = contacts[0] if contacts else None
            contact_name_val = (
                " ".join(p for p in [c0.first_name, c0.surname] if p).strip()
                if c0
                else None
            )
            latest_summary_val = c0.rolling_summary if c0 else None
        except Exception:
            contact_name_val = None
            latest_summary_val = None

        # Ensure assistant summaries are second-person
        assistant_extra = (
            "IMPORTANT: This rolling summary belongs to the assistant itself (contact_id 0). Always use **second-person** pronouns ('you') when describing the assistant. Never refer in the third person."  # noqa: E501
            if contact_id == 0
            else None
        )
        combined_guidance: Optional[str] = (
            "\n".join(g for g in (guidance, assistant_extra) if g) or None
        )

        contact_label = (
            f"{contact_name_val} (id {contact_id})"
            if contact_name_val
            else f"id {contact_id}"
        )
        self._llm.set_system_message(
            pb.build_rolling_prompt(
                contact_label,
                tools,
                guidance=combined_guidance,
            ),
        )

        # ------------------------------------------------------------------
        # Build payload with the rolling summary we already fetched.
        # ------------------------------------------------------------------
        payload = json.dumps(
            {
                "contact_id": contact_id,
                "latest_rolling_summary": latest_summary_val,
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

    async def update_contact_response_policy(
        self,
        transcript: str,
        *,
        contact_id: int,
        guidance: Optional[str] = None,
    ) -> str:  # noqa: D401 – imperative helper
        """Generate/refresh the response_policy field for the given contact."""

        target_id = contact_id

        async def set_response_policy(contact_id: int, response_policy: str) -> str:
            final_id = contact_id or target_id
            if final_id is None:
                raise ValueError(
                    "contact_id must be supplied either via argument or tool call.",
                )
            self._overlays.setdefault(final_id, {})["response_policy"] = response_policy
            handle = await self._contact_manager.update(
                f"Please set the response_policy for contact id {final_id} as follows:\n{response_policy}",
            )
            return await handle.result()

        tools: Dict[str, Callable[..., Any]] = {
            "transcript_ask": self._transcript_manager.ask,
            "contact_ask": self._contact_manager.ask,
            "set_response_policy": set_response_policy,
        }

        # Retrieve current policy & contact label (thread-safe)
        contacts = await asyncio.to_thread(
            self._contact_manager._search_contacts,
            filter=f"contact_id == {contact_id}",
            limit=1,
        )
        c0 = contacts[0] if contacts else None
        contact_name_val = (
            " ".join(p for p in [c0.first_name, c0.surname] if p).strip()
            if c0
            else None
        )
        latest_policy_val = c0.response_policy if c0 else None

        assistant_extra = (
            "IMPORTANT: This response policy belongs to the assistant itself (contact_id 0). Always use **second-person** pronouns ('you')."
            if contact_id == 0
            else None
        )
        combined_guidance = (
            "\n".join(g for g in (guidance, assistant_extra) if g) or None
        )

        contact_label = (
            f"{contact_name_val} (id {contact_id})"
            if contact_name_val
            else f"id {contact_id}"
        )

        self._llm.set_system_message(
            pb.build_response_policy_prompt(
                contact_label,
                tools,
                guidance=combined_guidance,
            ),
        )

        payload = json.dumps(
            {
                "contact_id": contact_id,
                "latest_response_policy": latest_policy_val,
                "transcript": transcript,
            },
            indent=2,
        )

        handle = start_async_tool_use_loop(
            self._llm,
            payload,
            tools,
            loop_id="SimulatedMemoryManager.update_contact_response_policy",
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

    async def update_tasks(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str:
        """
        Pretend to analyse the transcript and adjust the (simulated) task
        list accordingly. Returns a concise summary.
        """

        tools: Dict[str, Callable[..., Any]] = {
            "task_ask": self._task_scheduler.ask,
            "task_update": self._task_scheduler.update,
        }

        self._llm.set_system_message(
            pb.build_task_prompt(tools, guidance=guidance),
        )

        handle = start_async_tool_use_loop(
            self._llm,
            transcript,
            tools,
            loop_id="SimulatedMemoryManager.update_tasks",
            tool_policy=lambda i, _: ("required", _) if i < 2 else ("auto", _),
        )

        return await handle.result()
