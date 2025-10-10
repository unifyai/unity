from __future__ import annotations

import asyncio
import functools
import json
import os
from typing import Any, Callable, Dict, List, Optional

import unify
from ..common.llm_helpers import (
    methods_to_tool_dict,
    inject_broader_context,
    make_request_clarification_tool,
)
from ..common.async_tool_loop import (
    start_async_tool_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from ..constants import is_readonly_ask_guard_enabled
from ..common.read_only_ask_guard import ReadOnlyAskGuardHandle
from ..events.event_bus import EVENT_BUS, Event
from ..events.manager_event_logging import (
    log_manager_call,
    new_call_id,
    publish_manager_method_event,
)
from ..common.tool_outcome import ToolOutcome
from ..common.embed_utils import ensure_vector_column
from ..common.context_store import TableStore
from ..common.model_to_fields import model_to_fields
from .types import Secret
from .base import BaseSecretManager
from .prompt_builders import build_ask_prompt, build_update_prompt


class SecretManager(BaseSecretManager):
    """
    Manages a fixed-schema table of secrets. Ensures secrets are never exposed
    to LLMs directly. Public methods mirror other managers' design.
    """

    def __init__(self) -> None:
        # Resolve context and construct a single-table store
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
        ), "read and write contexts must match for SecretManager."
        self._ctx = f"{read_ctx}/Secrets"

        # Ensure storage/schema exists deterministically (idempotent)
        self._provision_storage()

        # Public tools
        self._ask_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                self._list_columns,
                self._filter_secrets,
                self._search_secrets,
                self._list_secret_keys,
                include_class_name=False,
            ),
        }
        self._update_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                self.ask,
                self._create_secret,
                self._update_secret,
                self._delete_secret,
                include_class_name=False,
            ),
        }

        # .env sync: create file if missing and backfill existing secrets as KEY=VALUE
        try:
            self._ensure_dotenv_synced_on_init()
        except Exception:
            # Best-effort – local file sync must never break construction
            pass

    # --------------------- Storage provisioning helper --------------------- #
    def _provision_storage(self) -> None:
        """Ensure Secrets context and schema exist and required vectors are present."""
        # Fixed schema derived from Secret model
        self._store = TableStore(
            self._ctx,
            unique_keys={"name": "str"},
            description="Key-value secrets with descriptions and embeddings.",
            fields=model_to_fields(Secret),
        )
        self._store.ensure_context()
        # Ensure vector for description (best-effort)
        try:
            ensure_vector_column(
                self._ctx,
                embed_column="description_emb",
                source_column="description",
                derived_expr=None,
            )
        except Exception:
            pass

    @functools.wraps(BaseSecretManager.clear, updated=())
    def clear(self) -> None:
        try:
            unify.delete_context(self._ctx)
        except Exception:
            # Proceed even if deletion fails (context may already be absent)
            pass

        # Force re-provisioning even if previously ensured
        try:
            from ..common.context_store import TableStore as _TS  # local import

            try:
                _TS._ENSURED.discard((unify.active_project(), self._ctx))
            except Exception:
                pass
        except Exception:
            pass

        # Re-create schema and vectors
        self._provision_storage()

        # Verify the context is visible before attempting reads
        try:
            import time as _time  # local import

            for _ in range(3):
                try:
                    unify.get_fields(context=self._ctx)
                    break
                except Exception:
                    _time.sleep(0.05)
        except Exception:
            pass

    # --------------------- Internal helpers (LLM client/policies) --------------------- #
    def _new_llm_client(self, model: str) -> "unify.AsyncUnify":
        """Construct a configured AsyncUnify client for the given model."""
        return unify.AsyncUnify(
            model,
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "false")),
            reasoning_effort="high",
            service_tier="priority",
        )

    @staticmethod
    def _default_ask_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Default ask-side tool policy (no-op, retain current tools)."""
        return ("auto", current_tools)

    @staticmethod
    def _default_update_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Default update-side tool policy (no-op, retain current tools)."""
        return ("auto", current_tools)

    # --------------------- Internal helpers (.env sync) --------------------- #
    def _dotenv_path(self) -> str:
        """Return the path to the .env file used for local sync.

        Honors SECRET_DOTENV_PATH for tests/overrides; defaults to ".env" in CWD.
        """
        return os.environ.get("SECRET_DOTENV_PATH") or os.path.join(os.getcwd(), ".env")

    def _ensure_dotenv_synced_on_init(self) -> None:
        """Create .env if missing and merge existing Unify secrets into it."""
        path = self._dotenv_path()
        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        except Exception:
            pass
        # Ensure file exists
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("")

        # Build name->value map from current storage
        try:
            rows = unify.get_logs(context=self._ctx)
        except Exception:
            rows = []
        existing: Dict[str, str] = {}
        for lg in rows:
            try:
                nm = (lg.entries or {}).get("name")
                val = (lg.entries or {}).get("value")
                if isinstance(nm, str) and nm and isinstance(val, str):
                    existing[nm] = val
            except Exception:
                continue

        if existing:
            self._env_merge_and_write(add_or_update=existing, remove_keys=None)

    @staticmethod
    def _parse_env_lines(lines: List[str]) -> Dict[str, int]:
        """Return mapping of existing KEY -> line index for a simple .env file."""
        import re

        key_to_idx: Dict[str, int] = {}
        for idx, raw in enumerate(lines):
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", raw)
            if m:
                key_to_idx[m.group(1)] = idx
        return key_to_idx

    def _env_merge_and_write(
        self,
        add_or_update: Dict[str, str] | None,
        remove_keys: List[str] | None,
    ) -> None:
        """Merge provided updates/removals into the .env file atomically."""
        path = self._dotenv_path()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        except FileNotFoundError:
            lines = []

        key_to_idx = self._parse_env_lines(lines)

        # Remove keys first
        if remove_keys:
            rm = set(remove_keys)

            def _keep(i: int, s: str) -> bool:
                for k, j in key_to_idx.items():
                    if j == i and k in rm:
                        return False
                return True

            lines = [s for i, s in enumerate(lines) if _keep(i, s)]
            key_to_idx = self._parse_env_lines(lines)

        # Add or update keys
        if add_or_update:
            for key, value in add_or_update.items():
                line = f"{key}={value}"
                if key in key_to_idx:
                    lines[key_to_idx[key]] = line
                else:
                    lines.append(line)

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))

    def _env_set(self, name: str, value: str) -> None:
        """Set or update one KEY=VALUE line in .env."""
        self._env_merge_and_write({name: value}, remove_keys=None)

    def _env_remove(self, name: str) -> None:
        """Remove one KEY from .env (if present)."""
        self._env_merge_and_write(add_or_update=None, remove_keys=[name])

    # --------------------- Public API --------------------- #
    async def from_placeholder(self, text: str) -> str:
        """Resolve ${name} placeholders in text to raw secret values (no LLM).

        Notes
        -----
        - Logs a single incoming ManagerMethod event and returns the resolved
          string without publishing any outgoing event to avoid leaking values.
        - Never persists or logs secret values.

        Parameters
        ----------
        text : str
            Input string that may contain placeholders like "${api_key}".

        Returns
        -------
        str
            String with placeholders substituted with their secret values.
        """
        call_id: str | None = None
        try:
            call_id = new_call_id()
            await publish_manager_method_event(
                call_id,
                "SecretManager",
                "from_placeholder",
                phase="incoming",
                query=text,
            )
        except Exception:
            # Logging is best-effort – failures must not impact resolution
            pass

        resolved = self._resolve_placeholders(text)

        # Publish an outgoing event that does NOT include sensitive data
        try:
            if call_id is not None:
                await publish_manager_method_event(
                    call_id,
                    "SecretManager",
                    "from_placeholder",
                    phase="outgoing",
                    status="resolved",
                )
        except Exception:
            pass

        return resolved

    async def to_placeholder(self, text: str) -> str:
        """Convert a secret values in text to a placeholder.

        Parameters
        ----------
        text : str
            The text to convert secret values to placeholders.

        Returns
        -------
        str
            The text with secret values converted to placeholders.
        """
        # Best-effort metadata-only logging; never include raw text or values
        call_id: str | None = None
        try:
            call_id = new_call_id()
            await publish_manager_method_event(
                call_id,
                "SecretManager",
                "to_placeholder",
                phase="incoming",
                info="start",
            )
        except Exception:
            pass

        # Build a mapping from raw value → name using current storage
        try:
            rows = unify.get_logs(
                context=self._ctx,
                from_fields=["name", "value"],
            )
            if not rows:
                rows = unify.get_logs(context=self._ctx)
        except Exception:
            rows = []

        value_to_name: Dict[str, str] = {}
        for lg in rows:
            try:
                nm = (lg.entries or {}).get("name")
                val = (lg.entries or {}).get("value")
                if isinstance(nm, str) and nm and isinstance(val, str) and val:
                    # If duplicate values exist, prefer lexicographically smallest name
                    if val in value_to_name:
                        if nm < value_to_name[val]:
                            value_to_name[val] = nm
                    else:
                        value_to_name[val] = nm
            except Exception:
                continue

        # Replace longer values first to avoid partial overlaps
        import re

        ordered_values = sorted(value_to_name.keys(), key=len, reverse=True)
        replaced_names: set[str] = set()
        result = text
        total_replacements = 0
        for val in ordered_values:
            name = value_to_name[val]
            pattern = re.escape(val)
            placeholder = f"${{{name}}}"
            result, count = re.subn(pattern, placeholder, result)
            if count:
                total_replacements += count
                replaced_names.add(name)

        try:
            if call_id is not None:
                await publish_manager_method_event(
                    call_id,
                    "SecretManager",
                    "to_placeholder",
                    phase="outgoing",
                    status="converted",
                    replacements=total_replacements,
                    names=sorted(replaced_names),
                )
        except Exception:
            pass

        return result

    @functools.wraps(BaseSecretManager.ask, updated=())
    @log_manager_call("SecretManager", "ask", payload_key="question")
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:
        # First, replace any known raw secret values with placeholders
        try:
            text = await self.to_placeholder(text)
        except Exception:
            pass

        client = self._new_llm_client("gpt-5@openai")

        # Build tools for read-only inspection
        tools = dict(self._ask_tools)
        if clarification_up_q is not None and clarification_down_q is not None:

            async def _on_request(q: str):
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "SecretManager",
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
                            "manager": "SecretManager",
                            "method": "ask",
                            "action": "clarification_answer",
                            "answer": ans,
                        },
                    ),
                )

            tools["request_clarification"] = make_request_clarification_tool(
                clarification_up_q,
                clarification_down_q,
                on_request=_on_request,
                on_answer=_on_answer,
            )

        # System message via prompt builder
        client.set_system_message(
            build_ask_prompt(tools=tools),
        )

        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=parent_chat_context,
            tool_policy=self._default_ask_tool_policy,
            preprocess_msgs=inject_broader_context,
            handle_cls=(
                ReadOnlyAskGuardHandle if is_readonly_ask_guard_enabled() else None
            ),
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    @functools.wraps(BaseSecretManager.update, updated=())
    @log_manager_call("SecretManager", "update", payload_key="request")
    async def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:
        # First, replace any known raw secret values with placeholders
        try:
            text = await self.to_placeholder(text)
        except Exception:
            pass

        client = self._new_llm_client("gpt-5@openai")

        tools = dict(self._update_tools)
        if clarification_up_q is not None and clarification_down_q is not None:

            async def _on_request(q: str):
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "SecretManager",
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
                            "manager": "SecretManager",
                            "method": "update",
                            "action": "clarification_answer",
                            "answer": ans,
                        },
                    ),
                )

            tools["request_clarification"] = make_request_clarification_tool(
                clarification_up_q,
                clarification_down_q,
                on_request=_on_request,
                on_answer=_on_answer,
            )

        client.set_system_message(
            build_update_prompt(tools=tools),
        )

        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.update.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=parent_chat_context,
            tool_policy=self._default_update_tool_policy,
            preprocess_msgs=inject_broader_context,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    # --------------------- Tools (read-only) --------------------- #
    def _resolve_placeholders(self, text: str) -> str:
        """Return a copy of text with ${name} placeholders replaced by values.

        This helper performs direct Unify reads and never emits logs/events.
        Unknown names are left unchanged.
        """
        import re

        def repl(match: "re.Match[str]") -> str:
            name = match.group(1)
            try:
                rows = unify.get_logs(
                    context=self._ctx,
                    filter=f"name == {name!r}",
                    limit=1,
                )
                if rows:
                    val = (rows[0].entries or {}).get("value")
                    if isinstance(val, str):
                        return val
            except Exception:
                pass
            return match.group(0)  # leave placeholder as-is when missing

        return re.sub(r"\$\{([^}]+)\}", repl, text)

    def _list_columns(
        self,
        *,
        include_types: bool = True,
    ) -> Dict[str, Any] | List[str]:
        """Return available columns for the secrets table.

        Parameters
        ----------
        include_types : bool, default True
            When True, returns a mapping ``{column_name: column_type}``.
            When False, returns a list of column names only.

        Returns
        -------
        Dict[str, Any] | List[str]
            Column map when ``include_types=True``; otherwise a list of names.
        """
        cols = self._store.get_columns()
        return cols if include_types else list(cols)

    def _search_secrets(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[Secret]:
        """Semantic search over secrets using the description embedding.

        Parameters
        ----------
        references : Dict[str, str] | None, default None
            Mapping of source expressions to reference text. For this manager
            use a column name like ``"description"`` to search over secret descriptions.
            When None or empty, returns most-recent rows.
        k : int, default 10
            Maximum number of results to return.

        Returns
        -------
        List[Secret]
            Up to ``k`` redacted Secret models (``value`` is never populated).
        """
        # Simple implementation: prefer description vector; fallback to recent
        try:
            from ..common.semantic_search import (
                fetch_top_k_by_references,
                backfill_rows,
            )

            rows = fetch_top_k_by_references(
                self._ctx,
                references,
                k=k,
                allowed_fields=["name", "description"],
                row_filter=None,
            )
            filled = backfill_rows(
                self._ctx,
                rows,
                k,
                row_filter=None,
                unique_id_field="name",
                allowed_fields=["name", "description"],
            )
            return [
                Secret(
                    name=r.get("name"),
                    value="",
                    description=r.get("description", ""),
                )
                for r in filled
            ]
        except Exception:
            logs = unify.get_logs(context=self._ctx, limit=k)
            return [
                Secret(
                    name=lg.entries.get("name"),
                    value="",
                    description=lg.entries.get("description", ""),
                )
                for lg in logs
            ]

    def _filter_secrets(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Secret]:
        """Filter secrets using a boolean expression evaluated per row.

        Parameters
        ----------
        filter : str | None, default None
            A Python expression evaluated with column names in scope (e.g.,
            ``"name == 'unify_key'"``). When None, returns all rows.
        offset : int, default 0
            Zero-based index of the first result to include.
        limit : int, default 100
            Maximum number of rows to return.

        Returns
        -------
        List[Secret]
            Matching Secret models with ``value`` redacted.
        """
        logs = unify.get_logs(
            context=self._ctx,
            filter=filter,
            offset=offset,
            limit=limit,
            from_fields=["name", "description"],
        )
        # Never expose values in read tools
        return [
            Secret(
                name=lg.entries.get("name"),
                value="",
                description=lg.entries.get("description", ""),
            )
            for lg in logs
        ]

    def _list_secret_keys(self) -> List[str]:
        """Return all available secret names (keys) stored in Unify.

        Returns
        -------
        List[str]
            Sorted, unique list of secret names currently present in storage.
        """
        try:
            rows = unify.get_logs(context=self._ctx)
        except Exception:
            rows = []
        names: set[str] = set()
        for lg in rows:
            nm = (lg.entries or {}).get("name")
            if isinstance(nm, str) and nm:
                names.add(nm)
        return sorted(names)

    # --------------------- Tools (mutations) --------------------- #
    def _create_secret(
        self,
        *,
        name: str,
        value: str,
        description: Optional[str] = None,
    ) -> ToolOutcome:
        """Create and persist a new secret.

        Parameters
        ----------
        name : str
            Unique identifier for the secret. Used as the placeholder name.
        value : str
            Raw secret value to store (never exposed to LLMs).
        description : str | None, default None
            Optional human-readable description.

        Returns
        -------
        ToolOutcome
            A standard outcome dict: ``{"outcome": "secret created", "details": {"name": <str>}}``.
        """
        assert name and value, "Both name and value are required."
        # Enforce uniqueness of name
        existing = unify.get_logs(
            context=self._ctx,
            filter=f"name == {name!r}",
            limit=1,
            return_ids_only=True,
        )
        assert not existing, f"Secret with name '{name}' already exists."

        # Write secret (store raw value in backend, but never surface to LLM)
        entries = {
            "name": name,
            "value": value,
            "description": description or "",
        }
        log = unify.log(context=self._ctx, **entries, new=True, mutable=True)

        # .env sync (best-effort)
        try:
            self._env_set(name, value)
        except Exception:
            pass

        return {"outcome": "secret created", "details": {"name": name}}

    def _update_secret(
        self,
        *,
        name: str,
        value: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ToolOutcome:
        """Update fields of an existing secret.

        Parameters
        ----------
        name : str
            Secret name to update.
        value : str | None, default None
            New raw value (optional). When provided it overwrites the existing value.
        description : str | None, default None
            New description (optional).

        Returns
        -------
        ToolOutcome
            Outcome dict: ``{"outcome": "secret updated", "details": {"name": <str>}}``.
        """
        # Find target log id
        ids = unify.get_logs(
            context=self._ctx,
            filter=f"name == {name!r}",
            limit=2,
            return_ids_only=True,
        )
        if not ids:
            raise ValueError(f"No secret found with name '{name}'.")
        if len(ids) > 1:
            raise RuntimeError(f"Multiple secrets found with name '{name}'.")
        log_id = ids[0]

        updates: Dict[str, Any] = {}
        if description is not None:
            updates["description"] = description
        if value is not None:
            updates["value"] = value

        if not updates:
            raise ValueError("No updates provided.")

        unify.update_logs(
            logs=[log_id],
            context=self._ctx,
            entries=updates,
            overwrite=True,
        )

        # .env sync when value provided (best-effort)
        try:
            if value is not None:
                self._env_set(name, value)
        except Exception:
            pass

        return {"outcome": "secret updated", "details": {"name": name}}

    def _delete_secret(self, *, name: str) -> ToolOutcome:
        """Delete a secret by name.

        Parameters
        ----------
        name : str
            The secret name to remove.

        Returns
        -------
        ToolOutcome
            Outcome dict: ``{"outcome": "secret deleted", "details": {"name": <str>}}``.
        """
        ids = unify.get_logs(
            context=self._ctx,
            filter=f"name == {name!r}",
            limit=2,
            return_ids_only=True,
        )
        if not ids:
            raise ValueError(f"No secret found with name '{name}'.")
        if len(ids) > 1:
            raise RuntimeError(f"Multiple secrets found with name '{name}'.")
        unify.delete_logs(context=self._ctx, logs=ids[0])
        # .env sync (best-effort)
        try:
            self._env_remove(name)
        except Exception:
            pass
        return {"outcome": "secret deleted", "details": {"name": name}}
