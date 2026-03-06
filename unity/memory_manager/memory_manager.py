# memory_manager/memory_manager.py
from __future__ import annotations

import asyncio
import functools
from typing import Optional, Callable, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass


from ..common.llm_client import new_llm_client
from ..manager_registry import ManagerRegistry
from ..common.llm_helpers import methods_to_tool_dict
from ..common.tool_spec import ToolSpec
from ..common.async_tool_loop import start_async_tool_loop
from . import prompt_builders as pb
from .base import BaseMemoryManager
from ..events.event_bus import EVENT_BUS, Event
from ..events.manager_event_logging import log_manager_result

if TYPE_CHECKING:
    from ..contact_manager.base import BaseContactManager
    from ..transcript_manager.base import BaseTranscriptManager
    from ..knowledge_manager.base import BaseKnowledgeManager
    from ..task_scheduler.base import BaseTaskScheduler


class MemoryManager(BaseMemoryManager):
    """
    Offline helper that processes transcripts in chunks (~50 messages by default).
    """

    # ------------------------------------------------------------------ #
    #  Shared helper: convert Message / event dicts to plain-text transcript
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_plain_transcript(
        messages: list[dict],
        contact_manager: Optional["BaseContactManager"] = None,
    ) -> str:
        """Return *plain-text* view of the message/event list.

        \u2022 Chat messages are rendered exactly like `TranscriptManager.build_plain_transcript`.
        \u2022 *ManagerMethod* events (kind == "manager_method") are preserved as **raw JSON** so
          downstream helpers/tests can still inspect all attributes (the tests assert the
          literal substring `"kind": "manager_method"` is present).
        """

        # Delegate chat messages to the central helper so naming etc. stays consistent
        from unity.transcript_manager.transcript_manager import (
            TranscriptManager as _TM,
        )

        plain_chat = _TM.build_plain_transcript(
            messages,
            contact_manager=contact_manager,
        )

        # Append serialised manager-method events (if any) *in the order they appear*
        extra_lines: list[str] = []
        for itm in messages:
            if itm.get("kind") == "manager_method":
                try:
                    import json  # local import to avoid polluting module namespace

                    dat = itm.get("data", {})
                    keys_to_omit = {
                        "row_id",
                        "event_id",
                        "calling_id",
                        "type",
                        "timestamp",
                        "event_timestamp",
                        "payload_cls",
                    }
                    concise = {
                        "kind": "manager_method",
                        **{k: v for k, v in dat.items() if k not in keys_to_omit},
                    }
                    extra_lines.append(json.dumps(concise))
                except Exception:
                    extra_lines.append(str(itm))

        if extra_lines:
            if plain_chat:
                return "\n".join([plain_chat] + extra_lines)
            return "\n".join(extra_lines)

        return plain_chat

    # ---------------------------------------------------------------------- #
    @dataclass(frozen=True)
    class MemoryConfig:
        """Configuration flags that control orchestration/runtime behaviour.

        enable_callbacks: When True, register EventBus callbacks for message
            ingestion and manager-method tracking. When False, skip callback
            registration.

        Per-capability flags gate which tools and prompt sections are included
        in the unified chunk-processing loop.
        """

        enable_callbacks: bool = True

        contacts: bool = True
        bios: bool = True
        rolling_summaries: bool = True
        response_policies: bool = True
        knowledge: bool = True
        tasks: bool = True

    def __init__(
        self,
        *,
        contact_manager: Optional["BaseContactManager"] = None,
        transcript_manager: Optional["BaseTranscriptManager"] = None,
        knowledge_manager: Optional["BaseKnowledgeManager"] = None,
        task_scheduler: Optional["BaseTaskScheduler"] = None,
        config: Optional["MemoryManager.MemoryConfig"] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self._contact_manager = (
            contact_manager
            if contact_manager is not None
            else ManagerRegistry.get_contact_manager()
        )
        self._transcript_manager = (
            transcript_manager
            if transcript_manager is not None
            else ManagerRegistry.get_transcript_manager(
                contact_manager=self._contact_manager,
            )
        )
        self._knowledge_manager = (
            knowledge_manager
            if knowledge_manager is not None
            else ManagerRegistry.get_knowledge_manager()
        )
        self._task_scheduler = (
            task_scheduler
            if task_scheduler is not None
            else ManagerRegistry.get_task_scheduler()
        )

        # \u2500\u2500 Config-controlled callback registration ----------------
        self._cfg: MemoryManager.MemoryConfig = (
            config if config is not None else MemoryManager.MemoryConfig()
        )
        self._register_update_callbacks: bool = self._cfg.enable_callbacks
        # \u2500\u2500 real-time 50-message trigger (update callbacks) --------------------
        self._CHUNK_SIZE: int = 50
        self._recent_messages: list[dict] = []
        self._messages_since_update: int = 0

        self._chunk_lock = asyncio.Lock()

        if self._register_update_callbacks:
            if loop is not None:
                loop.call_soon_threadsafe(
                    loop.create_task,
                    self._setup_message_callbacks(),
                )
            else:
                asyncio.create_task(self._setup_message_callbacks())

            EVENT_BUS.register_auto_pin(
                event_type="ManagerMethod",
                open_predicate=lambda e: (
                    e.payload.get("source") == "ConversationManager"
                    and e.payload.get("phase") == "incoming"
                ),
                close_predicate=lambda e: (
                    e.payload.get("source") == "ConversationManager"
                    and e.payload.get("phase") == "outgoing"
                ),
                key_fn=lambda e: e.calling_id,
            )

            if loop is not None:
                loop.call_soon_threadsafe(
                    loop.create_task,
                    self._setup_explicit_call_callbacks(),
                )
            else:
                asyncio.create_task(self._setup_explicit_call_callbacks())

    # ------------------------------------------------------------------ #
    #  Contact tool helpers (shared by update_contacts and process_chunk) #
    # ------------------------------------------------------------------ #

    def _build_contact_tools(self) -> Dict[str, Callable[..., Any]]:
        """Build the contact CRUD tool set with async wrappers."""

        @functools.wraps(self._contact_manager._create_contact, updated=())
        async def _create_contact(**kwargs):
            if kwargs.get("custom_fields"):
                raise ValueError(
                    "Creation of custom columns is not allowed.",
                )
            import inspect

            allowed = set(
                inspect.signature(self._contact_manager._create_contact).parameters,
            )
            cleaned_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            return await asyncio.to_thread(
                self._contact_manager._create_contact,
                **cleaned_kwargs,
            )

        @functools.wraps(self._contact_manager.update_contact, updated=())
        async def _update_contact(**kwargs):
            if kwargs.get("custom_fields"):
                raise ValueError(
                    "Modification involving custom columns is not allowed.",
                )
            import inspect

            allowed = set(
                inspect.signature(self._contact_manager.update_contact).parameters,
            )
            cleaned_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            return await asyncio.to_thread(
                self._contact_manager.update_contact,
                **cleaned_kwargs,
            )

        @functools.wraps(self._contact_manager._merge_contacts, updated=())
        async def _merge_contacts(**kwargs):
            import inspect

            allowed = set(
                inspect.signature(self._contact_manager._merge_contacts).parameters,
            )
            cleaned_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            return await asyncio.to_thread(
                self._contact_manager._merge_contacts,
                **cleaned_kwargs,
            )

        return {
            "contact_ask": ToolSpec(
                fn=self._contact_manager.ask,
                display_label="Querying contact book",
            ),
            "create_contact": ToolSpec(
                fn=_create_contact,
                display_label="Creating a contact",
            ),
            "update_contact": ToolSpec(
                fn=_update_contact,
                display_label="Updating a contact",
            ),
            "merge_contacts": ToolSpec(
                fn=_merge_contacts,
                display_label="Merging contacts",
            ),
        }

    # ------------------------------------------------------------------ #
    # 1  update_contacts                                                 #
    # ------------------------------------------------------------------ #
    @log_manager_result(
        "MemoryManager",
        "update_contacts",
        payload_key="transcript",
        display_label="Updating Contacts",
    )
    async def update_contacts(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str:
        """
        Scan the transcript, identify *new* contacts or modified details,
        and persist them.  Returns a short description of what changed.
        """
        tools = self._build_contact_tools()

        llm = new_llm_client()
        llm.set_system_message(
            pb.build_contact_update_prompt(
                tools,
                guidance=guidance,
            ),
        )

        handle = start_async_tool_loop(
            llm,
            transcript,
            tools,
            loop_id="MemoryManager.update_contacts",
        )

        return await handle.result()

    # ------------------------------------------------------------------ #
    # 2  update_knowledge                                                #
    # ------------------------------------------------------------------ #
    @log_manager_result(
        "MemoryManager",
        "update_knowledge",
        payload_key="transcript",
        display_label="Updating Knowledge",
    )
    async def update_knowledge(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str:
        """
        Mine reusable information and persist to the long-term knowledge base.
        """
        _km = self._knowledge_manager

        tools: Dict[str, Callable[..., Any]] = methods_to_tool_dict(
            ToolSpec(fn=self._contact_manager.ask, display_label="Looking up contacts"),
            ToolSpec(fn=_km.ask, display_label="Querying notes"),
            ToolSpec(fn=_km.refactor, display_label="Reorganising notes"),
            ToolSpec(fn=_km.update, display_label="Updating notes"),
            include_class_name=True,
        )

        llm = new_llm_client()
        llm.set_system_message(
            pb.build_knowledge_prompt(
                tools,
                guidance=guidance,
            ),
        )

        handle = start_async_tool_loop(
            llm,
            transcript,
            tools,
            loop_id="MemoryManager.update_knowledge",
        )

        return await handle.result()

    # ------------------------------------------------------------------ #
    # 3  update_tasks                                                    #
    # ------------------------------------------------------------------ #
    @log_manager_result(
        "MemoryManager",
        "update_tasks",
        payload_key="transcript",
        display_label="Updating Tasks",
    )
    async def update_tasks(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str:
        """
        Analyse the latest transcript chunk and update the task list using
        the TaskScheduler's public API (ask / update).  Returns a concise
        description of what was changed or 'no-op' when no updates were
        necessary.
        """
        tools: Dict[str, Callable[..., Any]] = methods_to_tool_dict(
            ToolSpec(fn=self._task_scheduler.ask, display_label="Querying tasks"),
            ToolSpec(fn=self._task_scheduler.update, display_label="Updating tasks"),
            include_class_name=True,
        )

        llm = new_llm_client()
        llm.set_system_message(
            pb.build_task_prompt(
                tools,
                guidance=guidance,
            ),
        )

        handle = start_async_tool_loop(
            llm,
            transcript,
            tools,
            loop_id="MemoryManager.update_tasks",
        )

        return await handle.result()

    # ------------------------------------------------------------------ #
    # 4  process_chunk  (unified single-loop for passive chunk trigger)  #
    # ------------------------------------------------------------------ #
    @log_manager_result(
        "MemoryManager",
        "process_chunk",
        payload_key="transcript",
        display_label="Processing Memory Chunk",
    )
    async def process_chunk(
        self,
        transcript: str,
        guidance: Optional[str] = None,
    ) -> str:
        """Run a single LLM tool loop that handles all enabled memory
        maintenance tasks (contacts, bios, rolling summaries, response
        policies, knowledge, tasks) in one pass.

        The ``self._cfg`` flags determine which tools and prompt sections
        are included.
        """
        tools: Dict[str, Callable[..., Any]] = {}

        # Contact tools (gated by contacts flag)
        if self._cfg.contacts:
            tools.update(self._build_contact_tools())

        # Knowledge tools
        if self._cfg.knowledge:
            _km = self._knowledge_manager
            tools.update(
                methods_to_tool_dict(
                    ToolSpec(fn=_km.ask, display_label="Querying notes"),
                    ToolSpec(fn=_km.refactor, display_label="Reorganising notes"),
                    ToolSpec(fn=_km.update, display_label="Updating notes"),
                    include_class_name=True,
                ),
            )
            # Also expose contact_ask for knowledge context lookups
            if "contact_ask" not in tools:
                tools["contact_ask"] = ToolSpec(
                    fn=self._contact_manager.ask,
                    display_label="Looking up contacts",
                )

        # Task tools
        if self._cfg.tasks:
            tools.update(
                methods_to_tool_dict(
                    ToolSpec(
                        fn=self._task_scheduler.ask,
                        display_label="Querying tasks",
                    ),
                    ToolSpec(
                        fn=self._task_scheduler.update,
                        display_label="Updating tasks",
                    ),
                    include_class_name=True,
                ),
            )

        if not tools:
            return "no-op (all memory capabilities disabled)"

        llm = new_llm_client()
        llm.set_system_message(
            pb.build_unified_prompt(
                tools,
                contacts=self._cfg.contacts,
                bios=self._cfg.bios,
                rolling_summaries=self._cfg.rolling_summaries,
                response_policies=self._cfg.response_policies,
                knowledge=self._cfg.knowledge,
                tasks=self._cfg.tasks,
                guidance=guidance,
            ),
        )

        handle = start_async_tool_loop(
            llm,
            transcript,
            tools,
            loop_id="MemoryManager.process_chunk",
        )

        return await handle.result()

    # ------------------------------------------------------------------ #
    # 5  reset                                                           #
    # ------------------------------------------------------------------ #
    async def reset(self) -> None:  # noqa: D401 \u2013 imperative name
        """Reset the event bus and re-register message-related callbacks."""
        EVENT_BUS.clear()

        if self._register_update_callbacks:
            await asyncio.gather(
                self._setup_message_callbacks(),
                self._setup_explicit_call_callbacks(),
            )

    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500  MESSAGE-BASED CALLBACKS  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    async def _setup_message_callbacks(self) -> None:
        """Register a callback that fires *every* incoming `message` event."""

        async def _cb(events):  # noqa: ANN001
            await self._on_new_message(events[0])

        try:
            await EVENT_BUS.register_callback(
                event_type="Message",
                callback=_cb,
                every_n=1,
            )
        except Exception:  # pragma: no cover
            pass

    async def _setup_explicit_call_callbacks(self) -> None:
        """Register a callback for ManagerMethod events tagged with
        `source == "ConversationManager"` (incoming & outgoing)."""

        async def _cb(events):  # noqa: ANN001
            await self._on_new_explicit_call(events[0])

        try:
            await EVENT_BUS.register_callback(
                event_type="ManagerMethod",
                callback=_cb,
                filter='evt.payload.get("source") == "ConversationManager" and evt.payload.get("manager") != "MemoryManager"',
                every_n=1,
            )
        except Exception:  # pragma: no cover
            pass

    async def _on_new_explicit_call(self, evt: Event) -> None:
        """Append explicit ManagerMethod events to the current buffer."""
        self._recent_messages.append(
            {
                "kind": "manager_method",
                "data": {
                    **(
                        evt.payload.model_dump(mode="json")
                        if hasattr(evt.payload, "model_dump")
                        else evt.payload
                    ),
                    "timestamp": evt.timestamp.isoformat(),
                    "calling_id": evt.calling_id,
                },
            },
        )

        self._messages_since_update += 1

        if self._messages_since_update >= self._CHUNK_SIZE:
            await self._flush_recent_items()

    async def _flush_recent_items(self) -> None:
        """Helper that triggers chunk processing & resets local counters."""
        self._messages_since_update = 0
        items = self._recent_messages.copy()
        self._recent_messages.clear()
        await self._process_message_chunk(items)

    async def _on_new_message(self, evt: Event) -> None:
        """Collect messages and trigger memory updates every *CHUNK_SIZE* messages."""
        payload = evt.payload

        # Payloads are standardised as dicts by the EventBus, but may still
        # arrive as Pydantic model instances from live publish paths.
        if hasattr(payload, "model_dump"):
            d = payload.model_dump(mode="json")
        elif isinstance(payload, dict):
            d = payload
        else:
            return

        # Minimal validation: a Message payload must have content.
        if "content" not in d:
            return

        ts_raw = d.get("timestamp")
        ts_str = (
            ts_raw.isoformat()
            if hasattr(ts_raw, "isoformat")
            else str(ts_raw) if ts_raw else ""
        )

        self._recent_messages.append(
            {
                "kind": "message",
                "data": {
                    "sender_id": d.get("sender_id"),
                    "receiver_ids": d.get("receiver_ids"),
                    "medium": d.get("medium"),
                    "timestamp": ts_str,
                    "content": d.get("content"),
                },
            },
        )

        self._messages_since_update += 1

        if self._messages_since_update >= self._CHUNK_SIZE:
            await self._flush_recent_items()

    async def _process_message_chunk(self, messages: list[dict]) -> None:
        """Run the unified memory update loop for one chunk."""
        async with self._chunk_lock:
            try:
                plain_transcript = self.build_plain_transcript(
                    messages,
                    contact_manager=self._contact_manager,
                )
                await self.process_chunk(plain_transcript)
            except Exception:  # pragma: no cover
                import traceback

                traceback.print_exc()

    # \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500  HELPERS  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    @classmethod
    def get_rolling_activity(cls, mode: str = "time") -> str:
        """Rolling activity has been temporarily removed; return an empty string."""
        return ""
