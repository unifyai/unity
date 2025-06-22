# memory_manager/simulated.py
"""
A lightweight, *offline-only* stand-in for the real `MemoryManager`.

It keeps an **internal, in-memory dictionary** of “contacts” so that calls to
`update_contact_bio` and `update_contact_rolling_summary` appear to mutate
state across invocations – but nothing ever touches an external store.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import unify

from .base import BaseMemoryManager


class SimulatedMemoryManager(BaseMemoryManager):
    """
    Drop-in replacement for tests / demos that runs entirely in RAM and
    fabricates plausible replies with an LLM.
    """

    def __init__(self, description: str = "imaginary scenario") -> None:
        # Tiny fake “database” keyed by contact_id
        self._contacts: Dict[int, Dict[str, str]] = {}

        # Shared stateful LLM so the simulation feels coherent
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )
        self._llm.set_system_message(
            "You are a *simulated* MemoryManager.  There is **no** real DB – "
            "invent sensible answers and pretend state updates succeed.\n\n"
            f"Back-story: {description}",
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    async def update_contacts(self, transcript: str) -> str:  # noqa: D401
        """
        Pretend to parse the transcript and add / update contacts.
        Simply returns a short, human-readable summary.
        """
        prompt = (
            "You are pretending to scan a 50-message transcript and create or "
            "update contacts.  Return a 1-sentence summary of what changed.\n\n"
            f"Transcript:\n{transcript}"
        )
        return await self._llm.generate(prompt)

    async def update_contact_bio(
        self,
        transcript: str,
        latest_bio: Optional[str] = None,
    ) -> str:
        """
        Fabricates a new bio (or keeps the old one) and stores it in RAM.
        """
        prompt = (
            "You are updating ONE contact’s *bio* based on a 50-message chunk.\n"
            f"Existing bio (may be null): {latest_bio}\n\n"
            f"Transcript:\n{transcript}\n\n"
            "Return only the *new* bio text (≤ 80 words)."
        )
        new_bio = await self._llm.generate(prompt)
        # Simulated persistence – always assumes contact_id == 1 for demo
        self._contacts.setdefault(1, {})["bio"] = new_bio
        return new_bio

    async def update_contact_rolling_summary(
        self,
        transcript: str,
        latest_rolling_summary: Optional[str] = None,
    ) -> str:
        """
        Generates a fresh ≤120-word rolling summary and stores it in RAM.
        """
        prompt = (
            "Update the 50-message *rolling summary* for ONE contact.\n"
            f"Previous summary (may be null): {latest_rolling_summary}\n\n"
            f"Transcript:\n{transcript}\n\n"
            "Return the new concise rolling summary (≤ 120 words)."
        )
        new_summary = await self._llm.generate(prompt)
        # Again, pretend the contact_id is 1
        self._contacts.setdefault(1, {})["rolling_summary"] = new_summary
        return new_summary
