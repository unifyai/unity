# memory_manager/memory_manager.py
from __future__ import annotations

import asyncio
import json
import os
from typing import Optional, Callable, Dict, Any

import unify

from ..contact_manager.contact_manager import ContactManager
from ..transcript_manager.transcript_manager import TranscriptManager
from ..knowledge_manager.knowledge_manager import KnowledgeManager
from ..common.llm_helpers import methods_to_tool_dict, start_async_tool_use_loop
from .prompt_builders import (
    build_contact_update_prompt,
    build_bio_prompt,
    build_rolling_prompt,
    build_knowledge_prompt,
)
from .base import BaseMemoryManager
from ..events.event_bus import EVENT_BUS, Event


class MemoryManager(BaseMemoryManager):
    """
    Offline helper invoked by a scheduler every ~50 messages.
    """

    # ─────────────────────────────  NEW CONSTANTS  ──────────────────────────
    _MANAGERS = {
        "ContactManager": "contact_manager",
        "TranscriptManager": "transcript_manager",
        "KnowledgeManager": "knowledge_manager",
        "TaskScheduler": "task_scheduler",
        "Conductor": "conductor",
    }

    _TIME_WINDOWS = {  # seconds
        "past_day": 60 * 60 * 24,
        "past_week": 60 * 60 * 24 * 7,
        "past_4_weeks": 60 * 60 * 24 * 7 * 4,
        "past_12_weeks": 60 * 60 * 24 * 7 * 12,
        "past_52_weeks": 60 * 60 * 24 * 7 * 52,
    }
    _COUNT_WINDOWS = {
        "past_interaction": 1,
        "past_10_interactions": 10,
        "past_40_interactions": 40,
        "past_120_interactions": 120,
        "past_520_interactions": 520,
    }
    _ALL_TIME = {"all_time": None}

    _tmp_cols = []
    for nick in _MANAGERS.values():
        for window in list(_TIME_WINDOWS) + list(_COUNT_WINDOWS) + list(_ALL_TIME):
            _tmp_cols.append(f"{nick}/{window}")

    _ROLLING_COLUMNS = tuple(_tmp_cols)
    del _tmp_cols

    # ---------------------------------------------------------------------- #
    def __init__(
        self,
        *,
        contact_manager: Optional[ContactManager] = None,
        transcript_manager: Optional[TranscriptManager] = None,
        knowledge_manager: Optional[KnowledgeManager] = None,
    ):
        self._contact_manager = contact_manager or ContactManager()
        self._transcript_manager = transcript_manager or TranscriptManager(
            contact_manager=self._contact_manager,
        )
        self._knowledge_manager = knowledge_manager or KnowledgeManager()

        # ── NEW: Rolling-Activity context & subscriptions ─────────────────
        self._rolling_ctx = self._ensure_rolling_context()
        asyncio.create_task(self._setup_rolling_callbacks())  # fire-and-forget

    # ------------------------------------------------------------------ #
    # 1  update_contacts                                                 #
    # ------------------------------------------------------------------ #
    async def update_contacts(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str:
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
        llm.set_system_message(build_contact_update_prompt(tools, guidance))

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
        guidance: Optional[str] = None,
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
        llm.set_system_message(build_bio_prompt(tools, guidance))

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
        guidance: Optional[str] = None,
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
        llm.set_system_message(build_rolling_prompt(tools, guidance))

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

    # ------------------------------------------------------------------ #
    # 4  update_knowledge                                                #
    # ------------------------------------------------------------------ #
    async def update_knowledge(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str:
        """
        Mine reusable information and persist to the long-term knowledge base.
        """

        tools: Dict[str, Callable[..., Any]] = methods_to_tool_dict(
            self._contact_manager.ask,
            self._transcript_manager.ask,
            self._knowledge_manager.ask,
            self._knowledge_manager.refactor,
            self._knowledge_manager.update,
            include_class_name=True,
        )

        llm = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
        )
        llm.set_system_message(build_knowledge_prompt(tools, guidance))

        handle = start_async_tool_use_loop(
            llm,
            transcript,
            tools,
            loop_id="MemoryManager.update_knowledge",
            tool_policy=lambda i, _: ("required", _) if i < 2 else ("auto", _),
        )

        return await handle.result()

    # ───────────────────────────  NEW HELPERS  ────────────────────────────
    # 1. Context & schema ---------------------------------------------------
    def _ensure_rolling_context(self) -> str:
        """Create the `RollingActivity` context (idempotent) and return its name."""
        active_ctx = unify.get_active_context()["write"] or ""
        ctx = f"{active_ctx}/RollingActivity" if active_ctx else "RollingActivity"
        if ctx not in unify.get_contexts():
            unify.create_context(ctx, unique_column_ids="row_id")
            fields = {
                col: {"type": "str", "mutable": True} for col in self._ROLLING_COLUMNS
            }
            unify.create_fields(fields, context=ctx)
        return ctx

    # 2. Callback registration ---------------------------------------------
    async def _setup_rolling_callbacks(self) -> None:
        """
        Register one EventBus subscription **per manager × window**.
        Each callback fires either
          • every *N* events  **or**
          • every *T* seconds since the previous trigger.
        """
        for mgr_cls, nick in self._MANAGERS.items():
            # baseline filter – narrow to one manager
            filt = f'evt.payload["manager"] == "{mgr_cls}" and evt.payload.get("phase") == "outgoing"'

            # count-based windows
            for window, n in self._COUNT_WINDOWS.items():
                col = f"{nick}/{window}"

                async def _cb(events, _col=col):  # bind column name now
                    await self._record_rolling_activity(_col, events)

                await EVENT_BUS.register_callback(
                    event_type="ManagerMethod",
                    callback=_cb,
                    filter=filt,
                    every_n=n,
                )

            # time-based windows
            for window, secs in self._TIME_WINDOWS.items():
                col = f"{nick}/{window}"

                async def _cb(events, _col=col):
                    await self._record_rolling_activity(_col, events)

                await EVENT_BUS.register_callback(
                    event_type="ManagerMethod",
                    callback=_cb,
                    filter=filt,
                    every_seconds=secs,
                )

            # all-time window is handled once at startup if empty
        # ensure at least one empty row exists
        if not unify.get_logs(context=self._rolling_ctx, limit=1):
            unify.log(context=self._rolling_ctx, new=True, mutable=True)

    # 3. Persisting the new snapshot ---------------------------------------
    async def _record_rolling_activity(
        self,
        column: str,
        events: list[Event],
    ) -> None:
        """
        Append a **new** row to RollingActivity, copying the previous one and
        updating *column* with the fresh LLM-generated summary.
        """
        # 0.  snapshot previous row (if any)
        prev = unify.get_logs(
            context=self._rolling_ctx,
            sorting={"row_id": "descending"},
            limit=1,
        )
        base_payload = prev[0].entries.copy() if prev else {}

        # 1.  strip event-bus boilerplate → compact text blob
        relevant = [
            {
                "manager": ev.payload.get("manager"),
                "method": ev.payload.get("method"),
                "details": {
                    k: v
                    for k, v in ev.payload.items()
                    if k not in {"manager", "method"}
                },
            }
            for ev in events
        ]
        events_blob = json.dumps(relevant, indent=2)

        # 2.  quick LLM summary
        llm = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
        )
        llm.set_system_message(
            "You are a concise assistant. Summarise the JSON array of manager "
            "events supplied by the user in max. 50 words.",
        )
        summary = (await llm.chat(events_blob)).strip()

        # 3.  assemble & persist new log
        base_payload[column] = summary
        unify.log(
            context=self._rolling_ctx,
            new=True,
            mutable=True,
            **base_payload,
        )
