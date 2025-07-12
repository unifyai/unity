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

    # ────────────────────────────────────────────────────────────────────
    #  NEW: hierarchy helpers so higher-level windows summarise the lower
    #       level rather than the raw ManagerMethod events
    # ────────────────────────────────────────────────────────────────────
    _TIME_ORDER = [
        "past_day",
        "past_week",
        "past_4_weeks",
        "past_12_weeks",
        "past_52_weeks",
    ]
    _COUNT_ORDER = [
        "past_interaction",
        "past_10_interactions",
        "past_40_interactions",
        "past_120_interactions",
        "past_520_interactions",
    ]

    # child_window → (immediate_lower_window, how_many_lower_summaries)
    _TIME_PARENT: dict[str, tuple[str, int]] = {}
    for i in range(1, len(_TIME_ORDER)):
        child, parent = _TIME_ORDER[i], _TIME_ORDER[i - 1]
        _TIME_PARENT[child] = (
            parent,
            _TIME_WINDOWS[child] // _TIME_WINDOWS[parent],
        )

    _COUNT_PARENT: dict[str, tuple[str, int]] = {}
    for i in range(1, len(_COUNT_ORDER)):
        child, parent = _COUNT_ORDER[i], _COUNT_ORDER[i - 1]
        _COUNT_PARENT[child] = (
            parent,
            _COUNT_WINDOWS[child] // _COUNT_WINDOWS[parent],
        )

    _tmp_cols = []
    for nick in _MANAGERS.values():
        for window in list(_TIME_WINDOWS) + list(_COUNT_WINDOWS) + list(_ALL_TIME):
            _tmp_cols.append(f"{nick}/{window}")

    # ───────────────────────────  SUMMARY COLS  ───────────────────────────
    _SUMMARY_TIME_COL = "time_based_activity"
    _SUMMARY_COUNT_COL = "count_based_activity"
    _tmp_cols.extend([_SUMMARY_TIME_COL, _SUMMARY_COUNT_COL])

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
        # Serialise updates to the RollingActivity context to avoid race conditions
        self._rolling_lock = asyncio.Lock()
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
            return f"Bio for contact with id {contact_id} successfully updated"

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
            return (
                f"Rolling summary for contact with id {contact_id} successfully updated"
            )

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
        Register callbacks that build the Rolling-Activity hierarchy.

        • Base-level snapshots
            – past_interaction (count based)
            – past_day         (time based)
          are still triggered directly from *ManagerMethod* events.

        • All higher-level windows are now triggered from the new
          `RollingSummary` event that each completed snapshot emits.
          The callback fires once *ratio* (= how many lower-level
          summaries constitute this window) such events have arrived.
        """

        async def _register_for_manager(mgr_cls: str, nick: str):
            # Register callbacks for one manager

            def _mk_cb(col_name: str):
                async def _cb(events, _col=col_name):
                    await self._record_rolling_activity(_col, events)

                    # schedule recording (silent now)

                return _cb

            mm_filter = (
                f'evt.payload["manager"] == "{mgr_cls}" '
                f'and evt.payload.get("phase") == "outgoing"'
            )

            async def _reg(cb_col: str, *, event_type: str, **kw):
                try:
                    await EVENT_BUS.register_callback(
                        event_type=event_type,
                        callback=_mk_cb(cb_col),
                        **kw,
                    )
                except Exception as e:
                    # propagate any registration error
                    raise

            # base-level
            await _reg(
                f"{nick}/past_interaction",
                event_type="ManagerMethod",
                filter=mm_filter,
                every_n=self._COUNT_WINDOWS["past_interaction"],
            )

            await _reg(
                f"{nick}/past_day",
                event_type="ManagerMethod",
                filter=mm_filter,
                every_seconds=self._TIME_WINDOWS["past_day"],
            )

            # count hierarchy
            for child, (parent, ratio) in self._COUNT_PARENT.items():
                await _reg(
                    f"{nick}/{child}",
                    event_type="RollingSummary",
                    filter=(
                        f'evt.payload["manager"] == "{nick}" '
                        f'and evt.payload["window"] == "{parent}"'
                    ),
                    every_n=ratio,
                )

            # time hierarchy
            for child, (parent, ratio) in self._TIME_PARENT.items():
                if child == "past_day":
                    continue
                await _reg(
                    f"{nick}/{child}",
                    event_type="RollingSummary",
                    filter=(
                        f'evt.payload["manager"] == "{nick}" '
                        f'and evt.payload["window"] == "{parent}"'
                    ),
                    every_n=ratio,
                )

        # launch all manager registrations concurrently
        await asyncio.gather(
            *[
                _register_for_manager(mgr_cls, nick)
                for mgr_cls, nick in self._MANAGERS.items()
            ],
        )

        # indicate readiness (useful for tests)
        try:
            self._callbacks_ready.set()  # type: ignore[attr-defined]
        except AttributeError:
            self._callbacks_ready = asyncio.Event()
            self._callbacks_ready.set()

    # 3. Persisting the new snapshot ---------------------------------------
    async def _record_rolling_activity_body(
        self,
        column: str,
        events: list[Event],
    ) -> None:
        """
        Append a **new** row to RollingActivity, copying the previous one and
        updating *column* with a fresh summary.

        • Base-level windows (``past_day`` / ``past_interaction``) are generated
          from the **raw** ManagerMethod events (unchanged behaviour).
        • Higher-level windows are created **only** from the summaries of the
          immediate lower-level window – thus forming a cascade:
              raw events → day → week → 4 weeks → 12 weeks → 52 weeks
              raw events → 1 interaction → 10 → 40 → 120 → 520
        """

        # ---- 0. previous snapshot ----------------------------------------
        prev = unify.get_logs(
            context=self._rolling_ctx,
            sorting={"row_id": "descending"},
            limit=1,
        )
        base_payload = prev[0].entries.copy() if prev else {}

        # ---- helper: concise LLM summary ---------------------------------
        async def _summarise(items: list[str | dict]) -> str:
            if not items:
                return ""
            llm = unify.AsyncUnify(
                "o4-mini@openai",
                cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
                traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            )
            llm.set_system_message(
                "You are a concise assistant. Summarise the JSON array supplied "
                "by the user in max. 50 words.",
            )
            return (await llm.generate(json.dumps(items, indent=2))).strip()

        # ------------------------------------------------------------------
        mgr_nick, window = column.split("/", 1)

        # ── 1.  Decide data-source (raw events vs. lower-level summaries) ──
        if window in {"past_day", "past_interaction"}:
            # base-level – summarise RAW events
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
            summary = await _summarise(relevant)

        else:
            # higher-level – derive from lower-level summaries
            if window in self._TIME_PARENT:
                lower_window, need = self._TIME_PARENT[window]
            elif window in self._COUNT_PARENT:
                lower_window, need = self._COUNT_PARENT[window]
            else:  # unexpected – fall back to raw events
                lower_window, need = None, 0

            if lower_window is None:
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
                summary = await _summarise(relevant)
            else:
                lower_col = f"{mgr_nick}/{lower_window}"
                rows = unify.get_logs(
                    context=self._rolling_ctx,
                    sorting={"row_id": "descending"},
                    limit=need * 5,  # generous buffer; we'll filter below
                )
                collected: list[str] = []
                for lg in rows:
                    txt = lg.entries.get(lower_col)
                    if txt:
                        collected.append(txt)
                    if len(collected) >= need:
                        break
                summary = await _summarise(collected)

        # ---- 2.  persist --------------------------------------------------
        # Ensure **new** row creation: remove any inherited `row_id` so Unify
        # allocates a fresh sequence number instead of silently updating the
        # previous snapshot (which would make successive interactions appear
        # as a single row and break tests expecting one row per call).
        base_payload.pop("row_id", None)

        base_payload[column] = summary

        # ──────────────────────────  NEW: Pre-compute summaries  ────────────────
        base_payload[self._SUMMARY_TIME_COL] = self._build_activity_summary(
            base_payload,
            "time",
        )
        base_payload[self._SUMMARY_COUNT_COL] = self._build_activity_summary(
            base_payload,
            "interaction",
        )

        # ------------------------------------------------------------------
        unify.log(
            context=self._rolling_ctx,
            new=True,
            mutable=True,
            **base_payload,
        )

        # ---- 3.  notify dependants ----------------------------------------
        # Emit a *RollingSummary* event so higher-level windows trigger only
        # after the lower-level snapshot is fully written.
        await EVENT_BUS.publish(
            Event(
                type="RollingSummary",
                payload={
                    "manager": mgr_nick,  # e.g. "contact_manager"
                    "window": window,  # e.g. "past_10_interactions"
                },
            ),
        )

    # ------------------------------------------------------------------ #
    #  Wrapper: guarantees single writer for RollingActivity             #
    # ------------------------------------------------------------------ #
    async def _record_rolling_activity(
        self,
        column: str,
        events: list[Event],
    ) -> None:
        """
        Thread-safe wrapper around :py:meth:`_record_rolling_activity_body` that
        ensures only *one* coroutine at a time can append a new snapshot to the
        ``RollingActivity`` context.  This prevents scenarios where two
        concurrent callbacks would both read the same *latest* row, apply their
        individual update, and then write out diverging successors derived from
        an inconsistent base state.
        """

        async with self._rolling_lock:
            await self._record_rolling_activity_body(column, events)

    # ------------------------------------------------------------------ #
    #  helper: build human-readable activity summary                     #
    # ------------------------------------------------------------------ #
    def _build_activity_summary(
        self,
        entries: dict[str, str],
        mode: str = "time",
    ) -> str:
        """Return the rolling activity summary Markdown for *entries*."""
        mode = mode.lower()
        if mode not in {"time", "interaction"}:
            raise ValueError("mode must be either 'time' or 'interaction'")

        windows: list[str] = (
            list(self._TIME_ORDER) if mode == "time" else list(self._COUNT_ORDER)
        )
        windows.append("all_time")

        def _pretty(w: str) -> str:
            if w == "all_time":
                return "All Time"
            parts = w.split("_")
            return "Past " + " ".join(
                p.capitalize() if not p.isdigit() else p for p in parts[1:]
            )

        _TITLE_DESC = {
            "task_scheduler": (
                "Tasks",
                "Overview of the tasks scheduled, updated, and performed.",
            ),
            "knowledge_manager": (
                "Knowledge",
                "Overview of the long-term memory (knowledge) added, updated, restructured, removed etc.",
            ),
            "contact_manager": (
                "Contacts",
                "Overview of contacts created or updated and related actions.",
            ),
            "transcript_manager": (
                "Transcripts",
                "Overview of messages and transcript summaries.",
            ),
            "conductor": (
                "Orchestration",
                "High-level orchestration and planning actions.",
            ),
        }

        lines: list[str] = []
        for mgr_cls, nick in self._MANAGERS.items():
            title, desc = _TITLE_DESC.get(
                nick,
                (mgr_cls.replace("Manager", ""), ""),
            )

            available: list[tuple[str, str]] = []
            for w in windows:
                col = f"{nick}/{w}"
                summary = entries.get(col)
                if summary:
                    available.append((w, summary))

            if not available:
                continue

            lines.append(f"# {title}")
            if desc:
                lines.append(desc)
            lines.append("")

            for w, summary in available:
                lines.append(f"## {_pretty(w)}")
                lines.append(summary)
                lines.append("")

        return "\n".join(lines).strip()

    # ------------------------------------------------------------------ #
    # 5  get_rolling_activity                                            #
    # ------------------------------------------------------------------ #
    def get_rolling_activity(self, mode: str = "time") -> str:
        """
        Return the **latest** Rolling-Activity snapshot as a human-readable
        Markdown string.
        """
        mode = mode.lower()
        if mode not in {"time", "interaction"}:
            raise ValueError("mode must be either 'time' or 'interaction'")

        rows = unify.get_logs(
            context=self._rolling_ctx,
            sorting={"row_id": "descending"},
            limit=1,
        )
        # If there is no stored rolling activity yet, return an *empty* string so
        # callers can completely omit the Historic Activity block.  This avoids
        # polluting system prompts with a verbose placeholder that carries no
        # useful information.
        if not rows:
            return ""

        latest = rows[0].entries
        key = self._SUMMARY_TIME_COL if mode == "time" else self._SUMMARY_COUNT_COL
        stored = latest.get(key)
        if stored:
            return stored

        # Fallback – build on the fly if snapshot predates summary columns
        return self._build_activity_summary(latest, mode)
