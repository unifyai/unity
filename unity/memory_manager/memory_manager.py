# memory_manager/memory_manager.py
from __future__ import annotations

import asyncio
import json
import os
from typing import Optional, Callable, Dict, Any

import unify

from ..contact_manager.contact_manager import ContactManager
from ..transcript_manager.transcript_manager import TranscriptManager
from ..common.llm_helpers import methods_to_tool_dict, start_async_tool_use_loop
from .prompt_builders import (
    build_contact_update_prompt,
    build_bio_prompt,
    build_rolling_prompt,
)
from .base import BaseMemoryManager


class MemoryManager(BaseMemoryManager):
    """
    Offline helper invoked by a scheduler every ~50 messages.
    """

    def __init__(
        self,
        *,
        contact_manager: Optional[ContactManager] = None,
        transcript_manager: Optional[TranscriptManager] = None,
    ):
        self._contact_manager = contact_manager or ContactManager()
        self._transcript_manager = transcript_manager or TranscriptManager(
            contact_manager=self._contact_manager,
        )

    # ------------------------------------------------------------------ #
    # 1  update_contacts                                                 #
    # ------------------------------------------------------------------ #
    async def update_contacts(self, transcript: str) -> str:
        """
        Scan the transcript, identify *new* contacts or modified details,
        and persist them.  Returns a short description of what changed.
        """

        # ─ 1.  Build live tool-set
        tools = methods_to_tool_dict(
            self._contact_manager.ask,
            self._contact_manager.update,  # full-power update allowed here
            self._transcript_manager.ask,
            include_class_name=False,
        )

        # ─ 2.  LLM client
        llm = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
        )
        llm.set_system_message(build_contact_update_prompt(tools))

        # ─ 3.  Kick off *single* tool-use loop
        handle = start_async_tool_use_loop(
            llm,
            transcript,
            tools,
            loop_id="MemoryManager.update_contacts",
            tool_policy=lambda i, _: ("required", _) if i < 2 else ("auto", _),
        )

        return await handle.result()  # a plain str

    # ------------------------------------------------------------------ #
    # 2  update_contact_bio                                              #
    # ------------------------------------------------------------------ #
    async def update_contact_bio(
        self,
        transcript: str,
        latest_bio: Optional[str] = None,
    ) -> str:
        """
        Refresh the *bio* column for ONE contact.
        Caller assembles the correct transcript slice & resolves the contact_id.
        """

        async def set_bio(contact_id: int, bio: str) -> str:
            """
            Restricted helper – only touches the `bio` column.
            """
            await asyncio.to_thread(
                self._contact_manager._update_contact,
                contact_id=contact_id,
                custom_fields={"bio": bio},
            )
            return "bio updated"

        tools: Dict[str, Callable[..., Any]] = {
            "transcript_ask": self._transcript_manager.ask,
            "contact_ask": self._contact_manager.ask,
            "set_bio": set_bio,
        }

        llm = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
        )
        llm.set_system_message(build_bio_prompt(tools))

        # Compose input blob
        user_blob = json.dumps(
            {
                "latest_bio": latest_bio,
                "transcript": transcript,
            },
            indent=2,
        )

        handle = start_async_tool_use_loop(
            llm,
            user_blob,
            tools,
            loop_id="MemoryManager.update_contact_bio",
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
        )

        return await handle.result()

    # ------------------------------------------------------------------ #
    # 3  update_contact_rolling_summary                                  #
    # ------------------------------------------------------------------ #
    async def update_contact_rolling_summary(
        self,
        transcript: str,
        latest_rolling_summary: Optional[str] = None,
    ) -> str:
        """
        Refresh the rolling_summary column for ONE contact.
        """

        async def set_rolling_summary(contact_id: int, rolling_summary: str) -> str:
            await asyncio.to_thread(
                self._contact_manager._update_contact,
                contact_id=contact_id,
                custom_fields={"rolling_summary": rolling_summary},
            )
            return "rolling_summary updated"

        tools: Dict[str, Callable[..., Any]] = {
            "transcript_ask": self._transcript_manager.ask,
            "contact_ask": self._contact_manager.ask,
            "set_rolling_summary": set_rolling_summary,
        }

        llm = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
        )
        llm.set_system_message(build_rolling_prompt(tools))

        user_blob = json.dumps(
            {
                "latest_rolling_summary": latest_rolling_summary,
                "transcript": transcript,
            },
            indent=2,
        )

        handle = start_async_tool_use_loop(
            llm,
            user_blob,
            tools,
            loop_id="MemoryManager.update_contact_rolling_summary",
            tool_policy=lambda i, _: ("required", _) if i < 1 else ("auto", _),
        )

        return await handle.result()
