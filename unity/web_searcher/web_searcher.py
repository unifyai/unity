from tavily import TavilyClient
import functools
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel
import asyncio
import unify
import functools
from pathlib import Path
from unity.settings import SETTINGS
from unity.common.log_utils import log as unity_log
from unity.common.async_tool_loop import (
    start_async_tool_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from unity.common.read_only_ask_guard import ReadOnlyAskGuardHandle
from unity.common.llm_client import new_llm_client
from unity.common.llm_helpers import (
    methods_to_tool_dict,
    make_request_clarification_tool,
)
from unity.events.manager_event_logging import log_manager_call
from unity.events.event_bus import EVENT_BUS, Event
from unity.web_searcher import prompt_builders
from .base import BaseWebSearcher
from ..common.tool_outcome import ToolOutcome
from ..common.model_to_fields import model_to_fields
from ..common.embed_utils import ensure_vector_column
from ..common.filter_utils import normalize_filter_expr
from ..common.search_utils import table_search_top_k
from .types.website import Website
from ..common.context_registry import ContextRegistry, TableContext


class WebSearcher(BaseWebSearcher):
    """
    Manages web search and extraction.
    """

    class Config:
        required_contexts = [
            TableContext(
                name="Websites",
                description="Catalog of websites of interest for WebSearcher routing/policies.",
                fields=model_to_fields(Website),
                unique_keys={"website_id": "int", "host": "str", "name": "str"},
                auto_counting={"website_id": None},
            ),
        ]

    def __init__(self):
        super().__init__()
        self.tavily_client = TavilyClient(api_key=SETTINGS.web.TAVILY_API_KEY or None)
        self._hierarchical_actor = None
        self._default_function_id = None

        # Resolve context for Websites table (single-table store)
        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs.get("read"), ctxs.get("write")
        if not read_ctx:
            try:
                from .. import ensure_initialised as _ensure_initialised

                _ensure_initialised()
                ctxs = unify.get_active_context()
                read_ctx, write_ctx = ctxs.get("read"), ctxs.get("write")
            except Exception:
                pass
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must match for WebSearcher."
        self.include_in_multi_assistant_table = True
        self._websites_ctx = ContextRegistry.get_context(self, "Websites")
        # Build the tools mapping once; copy when used
        ask_tools: Dict[str, Any] = methods_to_tool_dict(
            self._search,
            self._extract,
            self._crawl,
            self._map,
            self._gated_website_search,
            self._filter_websites,
            self._search_websites,
            include_class_name=False,
        )
        self.add_tools("ask", ask_tools)
        update_tools: Dict[str, Any] = {
            **methods_to_tool_dict(
                self.ask,
                self._create_website,
                self._update_website,
                self._delete_website,
                include_class_name=False,
            ),
        }
        self.add_tools("update", update_tools)
        # Ensure any internal caches/storage are present
        self._provision_storage()

    @property
    def hierarchical_actor(self):
        """Lazily initialize and return the HierarchicalActor instance."""
        if self._hierarchical_actor is None:
            from ..actor.hierarchical_actor import HierarchicalActor

            self._hierarchical_actor = HierarchicalActor()
        return self._hierarchical_actor

    def _ensure_default_function_exists(self) -> None:
        """Ensure the default website entrypoint exists and record its function_id.

        On any error, sets ``self._default_function_id`` to ``None``.
        """
        if self._default_function_id is not None:
            return

        try:
            fm = self.hierarchical_actor.function_manager
            fn_path = Path(__file__).parent / "functions" / "search_website_for_info.py"
            source = fn_path.read_text(encoding="utf-8")
            fm.add_functions(
                implementations=[source],
                overwrite=True,
                verify={"search_website_for_info": False},
            )
            # Re-check presence and set id if now present
            check = fm.filter_functions(
                filter="name == 'search_website_for_info'",
                limit=1,
            )
            self._default_function_id = (
                int(check[0].get("function_id")) if check else None
            )
            print("default function id:", self._default_function_id)
        except Exception:
            self._default_function_id = None

    @functools.wraps(BaseWebSearcher.ask, updated=())
    @log_manager_call("WebSearcher", "ask", payload_key="question")
    async def ask(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:
        client = new_llm_client()

        tools = dict(self.get_tools("ask"))
        if _clarification_up_q is not None and _clarification_down_q is not None:

            async def _on_request(q: str):
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "WebSearcher",
                            "method": "ask",
                            "action": "clarification_request",
                            "question": q,
                        },
                    ),
                )

            async def _on_answer(ans: str):
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "WebSearcher",
                            "method": "ask",
                            "action": "clarification_answer",
                            "answer": ans,
                        },
                    ),
                )

            tools["request_clarification"] = make_request_clarification_tool(
                _clarification_up_q,
                _clarification_down_q,
                on_request=_on_request,
                on_answer=_on_answer,
            )

        client.set_system_message(
            prompt_builders.build_ask_prompt(tools=tools).to_list(),
        )

        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            response_format=response_format,
            handle_cls=(
                ReadOnlyAskGuardHandle if SETTINGS.UNITY_READONLY_ASK_GUARD else None
            ),
            timeout=1200,
        )

        # If the caller requests reasoning steps, wrap the handle's result
        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore[attr-defined]

        return handle

    # ------------------------------------------------------------------ #
    #  Storage and lifecycle helpers                                     #
    # ------------------------------------------------------------------ #

    def _provision_storage(self) -> None:
        """
        Ensure internal caches and the Websites table exist (idempotent).

        Caches are kept process-local. Websites are persisted via Unify.
        """
        try:
            # Simple placeholders for last operation snapshots
            if not hasattr(self, "_last_results"):
                self._last_results: List[Dict[str, Any]] = []
            else:
                # keep existing values; provisioning is idempotent
                pass
            if not hasattr(self, "_last_extractions"):
                self._last_extractions: Dict[str, Any] = {}
            if not hasattr(self, "_last_crawls"):
                self._last_crawls: Dict[str, Any] = {}
            if not hasattr(self, "_last_maps"):
                self._last_maps: Dict[str, Any] = {}
        except Exception:
            # Best-effort only; callers operate without caches if needed
            pass

    @functools.cache
    def _ensure_notes_vector(self) -> None:
        # Ensure vector for notes (best-effort)
        try:
            ensure_vector_column(
                self._websites_ctx,
                embed_column="notes_emb",
                source_column="notes",
                derived_expr=None,
            )
        except Exception:
            pass

    @functools.wraps(BaseWebSearcher.clear, updated=())
    def clear(self) -> None:
        # Best-effort cache flush
        try:
            self._last_results = []
        except Exception:
            pass
        try:
            self._last_extractions = {}
        except Exception:
            pass
        try:
            self._last_crawls = {}
        except Exception:
            pass
        try:
            self._last_maps = {}
        except Exception:
            pass

        # Re-provision storage to a clean slate
        unify.delete_context(self._websites_ctx)

        ContextRegistry.refresh(self, "Websites")

        # Attempt to ensure context visibility before reads
        try:
            import time as _time

            for _ in range(3):
                try:
                    unify.get_fields(context=self._websites_ctx)
                    break
                except Exception:
                    _time.sleep(0.05)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Public update orchestration                                       #
    # ------------------------------------------------------------------ #

    @functools.wraps(BaseWebSearcher.update, updated=())
    @log_manager_call("WebSearcher", "update", payload_key="request")
    async def update(
        self,
        text: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:
        client = new_llm_client()

        tools = dict(self.get_tools("update"))
        if _clarification_up_q is not None and _clarification_down_q is not None:

            async def _on_request(q: str):
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "WebSearcher",
                            "method": "update",
                            "action": "clarification_request",
                            "question": q,
                        },
                    ),
                )

            async def _on_answer(ans: str):
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "WebSearcher",
                            "method": "update",
                            "action": "clarification_answer",
                            "answer": ans,
                        },
                    ),
                )

            tools["request_clarification"] = make_request_clarification_tool(
                _clarification_up_q,
                _clarification_down_q,
                on_request=_on_request,
                on_answer=_on_answer,
            )

        client.set_system_message(
            prompt_builders.build_update_prompt(tools=tools).to_list(),
        )

        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.update.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            response_format=response_format,
            tool_policy=self._default_update_tool_policy,
            timeout=1200,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore[attr-defined]

        return handle

    # ------------------------------------------------------------------ #
    #  Websites table tools                                              #
    # ------------------------------------------------------------------ #

    def _create_website(
        self,
        *,
        name: str,
        host: str,
        gated: bool,
        subscribed: bool,
        credentials: Optional[List[int]] = None,
        actor_entrypoint: Optional[int] = None,
        notes: str = "",
    ) -> ToolOutcome:
        """
        Create a new `Website` row and persist it to the Websites catalog.

        This tool registers a website of interest for the WebSearcher. The row is
        unique by `host` (and also by `name`), and captures whether the site is
        gated or subscription‑based, optional credential references, an optional
        Actor function entrypoint for gated navigation, and free‑form notes that
        inform routing and semantic search. Use this to onboard new sources that
        future `ask` or `update` operations may reference via catalog tools.

        Parameters
        ----------
        name : str
            Human‑friendly display name for the website (unique across rows).
        host : str
            Canonical host/domain used as the uniqueness key (e.g., "example.com").
        gated : bool
            True when the source requires login or access steps beyond public pages.
        subscribed : bool
            True when access relies on an active paid subscription/license.
        credentials : list[int] | None
            Optional foreign‑key references to secrets storing login materials.
        actor_entrypoint : int | None
            Optional function id for an Actor entrypoint used to navigate the site.
        notes : str
            Free‑form operational notes; also used for semantic catalog search.

        Returns
        -------
        ToolOutcome
            A short outcome summary including the created host identifier.
        """
        assert host, "host is required"
        assert (
            isinstance(name, str) and name.strip()
        ), "name is required and must be non-empty"

        existing = unify.get_logs(
            context=self._websites_ctx,
            filter=f"host == {host!r}",
            limit=1,
            return_ids_only=True,
        )
        assert not existing, f"Website with host '{host}' already exists."

        existing_name = unify.get_logs(
            context=self._websites_ctx,
            filter=f"name == {name!r}",
            limit=1,
            return_ids_only=True,
        )
        assert not existing_name, f"Website with name '{name}' already exists."

        entries: Dict[str, Any] = {
            "name": name,
            "host": host,
            "gated": bool(gated),
            "subscribed": bool(subscribed),
            "credentials": credentials if credentials else None,
            "actor_entrypoint": actor_entrypoint,
            "notes": notes or "",
        }
        unity_log(
            context=self._websites_ctx,
            **entries,
            new=True,
            mutable=True,
            add_to_all_context=self.include_in_multi_assistant_table,
        )
        return {"outcome": "website created", "details": {"host": host}}

    def _filter_websites(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Website]:
        """
        Return Website rows matching a boolean filter expression over columns.

        Use this read‑only catalog tool to retrieve stored website records based
        on attributes such as `host`, `name`, `gated`, `subscribed`, or
        `website_id`. The `filter` argument accepts a concise boolean expression
        with `==`, `!=`, `and`, and `or` over these fields. Results are paginated
        with `offset` and `limit`. This does not perform web access; it only reads
        the persisted Websites table to support configuration inspection and
        validation within the tool loop.

        Parameters
        ----------
        filter : str | None
            Optional expression to constrain rows (e.g., "gated == true and subscribed == false").
        offset : int
            Number of rows to skip from the start of the result set.
        limit : int
            Maximum number of rows to return (default 100).

        Returns
        -------
        list[Website]
            Materialised Website models for downstream reasoning or display.
        """
        normalized = normalize_filter_expr(filter)
        logs = unify.get_logs(
            context=self._websites_ctx,
            filter=normalized,
            offset=offset,
            limit=limit,
        )
        result: List[Website] = []
        for lg in logs:
            ent = lg.entries or {}
            result.append(
                Website(
                    website_id=(
                        int(ent.get("website_id"))
                        if ent.get("website_id") is not None
                        else -1
                    ),
                    name=ent.get("name", ""),
                    host=ent.get("host"),
                    gated=bool(ent.get("gated", False)),
                    subscribed=bool(ent.get("subscribed", False)),
                    credentials=ent.get("credentials"),
                    actor_entrypoint=ent.get("actor_entrypoint"),
                    notes=ent.get("notes", ""),
                ),
            )
        return result

    def _search_websites(
        self,
        *,
        notes: str,
        k: int = 10,
    ) -> List[Website]:
        """
        Semantic similarity search over the Websites catalog using `notes` text.

        Provide a short description of the desired source and this tool returns
        the top‑k catalog rows whose `notes` are most semantically similar. This
        is useful for routing (e.g., selecting which site to query) and for
        quickly recalling previously added sources. It operates only on stored
        metadata; it does not contact external websites. The `k` parameter bounds
        the number of results returned.

        Parameters
        ----------
        notes : str
            Natural‑language description used as the search query.
        k : int
            Maximum number of similar rows to return (1–1000; default 10).

        Returns
        -------
        list[Website]
            Ranked subset of Website rows relevant to the provided description.
        """
        if not isinstance(notes, str) or not notes.strip():
            return []
        self._ensure_notes_vector()
        rows = table_search_top_k(
            context=self._websites_ctx,
            references={"notes": notes},
            k=max(1, min(int(k), 1000)),
            allowed_fields=[
                "website_id",
                "name",
                "host",
                "gated",
                "subscribed",
                "credentials",
                "actor_entrypoint",
                "notes",
            ],
            row_filter=None,
            unique_id_field="website_id",
        )
        return [
            Website(
                website_id=(
                    int(r.get("website_id")) if r.get("website_id") is not None else -1
                ),
                name=r.get("name", ""),
                host=r.get("host"),
                gated=bool(r.get("gated", False)),
                subscribed=bool(r.get("subscribed", False)),
                credentials=r.get("credentials"),
                actor_entrypoint=r.get("actor_entrypoint"),
                notes=r.get("notes", ""),
            )
            for r in rows
        ]

    async def _gated_website_search(
        self,
        *,
        queries: str | list[str],
        website: Dict[str, Any] | Website,
    ) -> str:
        """Search a gated website using the Actor entrypoint with Website data.

        Parameters
        ----------
        queries : str | list[str]
            One or more search queries to run on the target site.
            Pass multiple queries to search for different topics in a single browser session.
        website : Website | dict
            Website record containing host, credentials, actor_entrypoint, notes.
        """
        print("Searching gated website:", website)
        # Normalize queries to a list
        if isinstance(queries, str):
            queries = [queries]

        # Normalise website record
        host: str = (
            website.get("host")
            if isinstance(website, dict)
            else getattr(website, "host", "")
        )
        creds_field = (
            website.get("credentials")
            if isinstance(website, dict)
            else getattr(website, "credentials", None)
        )
        actor_fn_id = (
            website.get("actor_entrypoint")
            if isinstance(website, dict)
            else getattr(website, "actor_entrypoint", None)
        )

        creds: List[int] = []
        if isinstance(creds_field, list):
            try:
                creds = [int(x) for x in creds_field]
            except Exception:
                creds = []

        # Resolve function id: prefer site-specific entrypoint; else default
        self._ensure_default_function_exists()
        function_id = (
            actor_fn_id
            if actor_fn_id is not None and actor_fn_id >= 0
            else self._default_function_id
        )
        if function_id is None:
            return "Failed gated website search: Both actor entrypoint and default function are unavailable. Unable to resolve."

        # Start the actor plan with explicit entrypoint args
        queries_desc = ", ".join(queries[:3]) + ("..." if len(queries) > 3 else "")
        plan = await self.hierarchical_actor.act(
            description=f"Search website for information: {queries_desc}. Start with {host}",
            entrypoint=function_id,
            entrypoint_args=[queries, host, creds],
            persist=False,
            new_session=True,
        )
        result = await plan.result()
        return str(result)

    def _delete_website(
        self,
        *,
        name: Optional[str] = None,
        host: Optional[str] = None,
        website_id: Optional[int] = None,
    ) -> ToolOutcome:
        """
        Delete a single `Website` row identified by `website_id`, `host`, or `name`.

        Exactly one row must match the provided identifier(s); if none are found
        this tool raises an error, and if multiple rows match it fails to avoid
        accidental mass deletion. Use this to remove stale or incorrect catalog
        entries from the Websites table. Deletions are permanent and affect only
        the configuration store; they do not interact with external websites.

        Parameters
        ----------
        name : str | None
            Optional row selector by human‑friendly name (unique when present).
        host : str | None
            Optional row selector by canonical host/domain.
        website_id : int | None
            Optional selector by internal integer identifier.

        Returns
        -------
        ToolOutcome
            Outcome summary containing the identifier that was used for deletion.
        """
        exprs: List[str] = []
        if name is not None:
            exprs.append(f"name == {name!r}")
        if host is not None:
            exprs.append(f"host == {host!r}")
        if website_id is not None:
            exprs.append(f"website_id == {int(website_id)}")
        filt = " and ".join(exprs) if exprs else None

        ids = unify.get_logs(
            context=self._websites_ctx,
            filter=filt,
            limit=2,
            return_ids_only=True,
        )
        if not ids:
            raise ValueError("No website found matching the provided identifier.")
        if len(ids) > 1:
            raise RuntimeError("Multiple websites match the provided identifier.")

        unify.delete_logs(context=self._websites_ctx, logs=ids[0])
        return {
            "outcome": "website deleted",
            "details": {"host": host, "website_id": website_id},
        }

    def _update_website(
        self,
        *,
        website_id: Optional[int] = None,
        match_host: Optional[str] = None,
        match_name: Optional[str] = None,
        name: Optional[str] = None,
        host: Optional[str] = None,
        gated: Optional[bool] = None,
        subscribed: Optional[bool] = None,
        credentials: Optional[List[int]] = None,
        actor_entrypoint: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> ToolOutcome:
        """Update fields of an existing Website.

        Identify the target by one of: website_id, match_host, or match_name.
        Only provided fields are updated; others remain unchanged. Uniqueness
        of updated `host` and `name` is enforced.
        """
        # Identify target row
        filt_exprs: List[str] = []
        if website_id is not None:
            filt_exprs.append(f"website_id == {int(website_id)}")
        if match_host is not None:
            filt_exprs.append(f"host == {match_host!r}")
        if match_name is not None:
            filt_exprs.append(f"name == {match_name!r}")
        if not filt_exprs:
            raise ValueError(
                "Provide one identifier: website_id, match_host, or match_name.",
            )
        filt = " and ".join(filt_exprs)

        ids = unify.get_logs(
            context=self._websites_ctx,
            filter=filt,
            limit=2,
            return_ids_only=True,
        )
        if not ids:
            raise ValueError("No website found matching the provided identifier.")
        if len(ids) > 1:
            raise RuntimeError("Multiple websites match the provided identifier.")
        target_id = ids[0]

        # Build updates from non-None fields
        updates: Dict[str, Any] = {}
        if name is not None:
            updates["name"] = name
        if host is not None:
            updates["host"] = host
        if gated is not None:
            updates["gated"] = bool(gated)
        if subscribed is not None:
            updates["subscribed"] = bool(subscribed)
        if credentials is not None:
            updates["credentials"] = credentials
        if actor_entrypoint is not None:
            updates["actor_entrypoint"] = actor_entrypoint
        if notes is not None:
            updates["notes"] = notes

        if not updates:
            raise ValueError("No updates provided.")

        # Enforce uniqueness for host/name when changing them (allow same-row)
        if "host" in updates:
            dupe = unify.get_logs(
                context=self._websites_ctx,
                filter=f"host == {updates['host']!r}",
                limit=1,
                return_ids_only=True,
            )
            if dupe and dupe[0] != target_id:
                raise AssertionError(
                    f"Website with host '{updates['host']}' already exists.",
                )
        if "name" in updates:
            dupe = unify.get_logs(
                context=self._websites_ctx,
                filter=f"name == {updates['name']!r}",
                limit=1,
                return_ids_only=True,
            )
            if dupe and dupe[0] != target_id:
                raise AssertionError(
                    f"Website with name '{updates['name']}' already exists.",
                )

        unify.update_logs(
            logs=[target_id],
            context=self._websites_ctx,
            entries=updates,
            overwrite=True,
        )

        details: Dict[str, Any] = {
            "website_id": website_id,
            "match_host": match_host,
            "match_name": match_name,
        }
        details.update(
            {
                k: updates.get(k)
                for k in ("name", "host", "gated", "subscribed")
                if k in updates
            },
        )
        return {"outcome": "website updated", "details": details}

    def _search(
        self,
        query: str,
        *,
        max_results: int = 5,
        start_date: str = None,
        end_date: str = None,
        include_images: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Perform a web search and return a structured result.

        Parameters
        ----------
        query : str
            The search query.
        max_results : int, default 5
            Maximum number of results to return.
        start_date : str, default None
            Will return all results after the specified start date ( publish date ). Required to be written in the format YYYY-MM-DD.
        end_date : str, default None
            Will return all results before the specified end date ( publish date ). Required to be written in the format YYYY-MM-DD.
        include_images : bool, default False
            Also perform an image search and include the results in the response.

        Returns
        -------
        Dict[str, Any]
            Structured search output with keys:
            - "answer": Concise summary string.
            - "results": Ranked list of sources with titles, URLs and snippets.
            - "images": When requested, a list of related images (may be empty).
        """
        response = self.tavily_client.search(
            query=query,
            max_results=max_results,
            start_date=start_date,
            end_date=end_date,
            include_images=include_images,
            include_answer=True,
        )
        return {
            "answer": response.get("answer", ""),
            "results": response.get("results", []),
            "images": response.get("images", []),
        }

    def _extract(
        self,
        urls: str | List[str],
        *,
        include_images: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract cleaned content from webpage URL.

        Parameters
        ----------
        urls : str | List[str]
            The URL to extract content from.
        include_images : bool, default False
            Also extract images from the URLs.

        Returns
        -------
        Dict[str, Any]
            Parsed content payload with keys:
            - "results": Successful extractions with cleaned content/metadata.
            - "failed_results": Any URLs that could not be extracted.
        """
        response = self.tavily_client.extract(urls=urls, include_images=include_images)
        return {
            "results": response.get("results", []),
            "failed_results": response.get("failed_results", []),
        }

    def _crawl(
        self,
        start_url: str,
        *,
        instructions: str | None = None,
        max_depth: int | None = None,
        max_breadth: int | None = None,
        limit: int | None = None,
        include_images: bool | None = None,
    ) -> Dict[str, Any]:
        """
        Graph-based website traversal.

        Parameters
        ----------
        start_url : str
            The root URL to begin the crawl.
        instructions : str, default None
            Natural language instructions for the crawler.
        max_depth : int | None, default None
            Maximum crawl depth (uses service defaults when None).
        max_breadth : int | None, default None
            Maximum number of links to follow per page (uses service defaults when None).
        limit : int | None, default None
            Overall limit on number of pages to crawl (uses service defaults when None).
        include_images : bool | None, default None
            Whether to include images in crawl results (uses service defaults when None).

        Returns
        -------
        Dict[str, Any]
            Crawl summary with keys:
            - "base_url": Normalised base host for the crawl session.
            - "results": List of discovered pages and associated content.
        """
        response = self.tavily_client.crawl(
            url=start_url,
            instructions=instructions,
            max_depth=max_depth,
            max_breadth=max_breadth,
            limit=limit,
            include_images=include_images,
        )
        return {
            "base_url": response.get("base_url"),
            "results": response.get("results", []),
        }

    # ------------------------------------------------------------------ #
    #  Tool policies                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_update_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """
        Require an initial read-only ask on the first turn (if enabled); auto thereafter.

        This mirrors ContactManager behaviour so that update flows that depend
        on catalog inspection begin with a deterministic nested ask handle,
        making nested-structure behaviour stable.
        """
        from unity.settings import SETTINGS

        if (
            SETTINGS.FIRST_MUTATION_TOOL_IS_ASK
            and step_index < 1
            and "ask" in current_tools
        ):
            return ("required", {"ask": current_tools["ask"]})
        return ("auto", current_tools)

    def _map(
        self,
        url: str,
        *,
        instructions: str | None = None,
        max_depth: int | None = None,
        max_breadth: int | None = None,
        limit: int | None = None,
        include_images: bool | None = None,
    ) -> Dict[str, Any]:
        """
        Structured mapping over sources for complex research queries.

        Parameters
        ----------
        url : str
            The root URL to begin the mapping.
        instructions : str | None, default None
            Natural language guidance for the mapping process.
        max_depth : int | None, default None
            Maximum traversal depth (uses service defaults when None).
        max_breadth : int | None, default None
            Maximum number of branches to explore per step (uses service defaults when None).
        limit : int | None, default None
            Overall limit on the number of items to consider (uses service defaults when None).
        include_images : bool | None, default None
            Whether to include images in the mapped results (uses service defaults when None).

        Returns
        -------
        Dict[str, Any]
            Mapping summary with keys:
            - "base_url": Normalised base host when applicable.
            - "results": List of mapped items/pages relevant to the query.
        """
        response = self.tavily_client.map(
            url=url,
            instructions=instructions,
            max_depth=max_depth,
            max_breadth=max_breadth,
            limit=limit,
            include_images=include_images,
        )
        return {
            "base_url": response.get("base_url"),
            "results": response.get("results", []),
        }
