from __future__ import annotations

import asyncio
import functools
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel

import unify
from unity.common.llm_client import new_llm_client
from unity.common.log_utils import log as unity_log
from ..common.llm_helpers import methods_to_tool_dict
from ..common.tool_spec import ToolSpec
from ..common.async_tool_loop import (
    start_async_tool_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from ..settings import SETTINGS
from ..common.read_only_ask_guard import ReadOnlyAskGuardHandle
from ..events.event_bus import EVENT_BUS, Event
from ..events.manager_event_logging import log_manager_call
from ..common.tool_outcome import ToolOutcome
from ..common.embed_utils import ensure_vector_column
from ..common.context_store import TableStore
from ..common.model_to_fields import model_to_fields
from .types import OAuthToken, Secret, SecretBinding, SecretDefault
from .base import BaseSecretManager
from .prompt_builders import build_ask_prompt, build_update_prompt
from ..common.filter_utils import normalize_filter_expr
from ..common.search_utils import table_search_top_k, is_plain_identifier
from ..common.context_registry import ContextRegistry, TableContext
from ..common.authoring import authoring_assistant_id
from ..session_details import SESSION_DETAILS
from .settings import (
    OAUTH_TOKENS_TABLE,
    SECRET_BINDING_TABLE,
    SECRET_DEFAULT_TABLE,
    SECRETS_TABLE,
)


class SecretBackendReadError(RuntimeError):
    """Raised when SecretManager cannot read an authoritative backend slice."""


logger = logging.getLogger(__name__)


class SecretManager(BaseSecretManager):
    """Shared credential vault plus per-body binding overlay.

    The SecretManager owns three distinct storage surfaces:

    * :data:`SECRETS_TABLE` — the shared credential vault. In a Hive,
      every member reads and writes one ``Hives/{hive_id}/Secrets``
      table so a shared credential lives in exactly one row.
    * :data:`SECRET_BINDING_TABLE` — a per-body overlay. One row per
      integration expresses *which* shared credential *this* body uses
      when that integration runs. Bodies in the same Hive can bind
      ``salesforce`` to different :class:`Secret` rows without
      fighting over a single value.
    * :data:`SECRET_DEFAULT_TABLE` — a Hive-wide fallback overlay.
      Bodies in a Hive that have not yet materialized their own
      :class:`SecretBinding` for an integration fall back to the
      default expressed here before the resolver gives up.

    OAuth tokens synced from Orchestra live on a fourth, per-body
    context (:data:`OAUTH_TOKENS_TABLE`) rather than in the shared
    vault — they are identity material bound to the body that
    completed the OAuth flow and must never leak across the Hive.
    """

    class Config:
        required_contexts = [
            TableContext(
                name=SECRETS_TABLE,
                description=(
                    "Shared credential vault. Hive-scoped: every body "
                    "in a Hive reads and writes the same rows."
                ),
                fields=model_to_fields(Secret),
                unique_keys={"secret_id": "int", "name": "str"},
                auto_counting={"secret_id": None},
            ),
            # Per-body overlay mapping integrations → credentials. The FK
            # is declared against the relative ``Secrets.secret_id``; the
            # ContextRegistry rewrites it to the correct base (Hive root
            # for Hive members, per-body for solo bodies) so the overlay
            # always points at the shared vault that backs this body.
            TableContext(
                name=SECRET_BINDING_TABLE,
                description=(
                    "Per-body overlay selecting which shared credential "
                    "this body uses for each integration."
                ),
                fields=model_to_fields(SecretBinding),
                unique_keys={"integration": "str"},
                auto_counting=None,
                foreign_keys=[
                    {
                        "name": "secret_id",
                        "references": f"{SECRETS_TABLE}.secret_id",
                        "on_delete": "SET NULL",
                    },
                ],
            ),
            # Hive-wide default for each integration. Also Hive-scoped,
            # so solo bodies never write or read this table in practice;
            # they fall back to ``Secret.name == integration`` instead.
            TableContext(
                name=SECRET_DEFAULT_TABLE,
                description=(
                    "Hive-wide default credential selection per "
                    "integration. Consulted when a body has no "
                    ":class:`SecretBinding` of its own."
                ),
                fields=model_to_fields(SecretDefault),
                unique_keys={"integration": "str"},
                auto_counting=None,
                foreign_keys=[
                    {
                        "name": "secret_id",
                        "references": f"{SECRETS_TABLE}.secret_id",
                        "on_delete": "SET NULL",
                    },
                ],
            ),
            # OAuth tokens synced from Orchestra. Always per-body: a
            # Hive member's Google/Microsoft refresh token is identity
            # material for that specific body and must never share a
            # row with another body's tokens.
            TableContext(
                name=OAUTH_TOKENS_TABLE,
                description=(
                    "Per-body OAuth tokens synced from Orchestra. Kept "
                    "off the shared credential vault so one body's "
                    "identity material never leaks to its peers."
                ),
                fields=model_to_fields(OAuthToken),
                unique_keys={"name": "str"},
                auto_counting=None,
            ),
        ]

    def __init__(self) -> None:
        super().__init__()
        self.include_in_multi_assistant_table = True

        self._ctx = ContextRegistry.get_context(self, SECRETS_TABLE)
        self._binding_ctx = ContextRegistry.get_context(self, SECRET_BINDING_TABLE)
        self._default_ctx = ContextRegistry.get_context(self, SECRET_DEFAULT_TABLE)
        self._oauth_ctx = ContextRegistry.get_context(self, OAUTH_TOKENS_TABLE)

        # Keys this manager last exported into ``os.environ`` / ``.env``
        # via ``_sync_dotenv``. Tracked so an unbind (or a Hive default
        # revocation) can remove the previously-exported credential
        # before pooled venvs respawn under the refreshed environment.
        self._exported_env_keys: set[str] = set()
        # Retain scheduled venv-invalidation tasks so the asyncio GC
        # does not reap them before they run.
        self._pending_tasks: "set[asyncio.Task[Any]]" = set()

        self._provision_storage()

        ask_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                ToolSpec(
                    fn=self._list_columns,
                    display_label="Listing credential fields",
                ),
                ToolSpec(
                    fn=self._filter_secrets,
                    display_label="Filtering credentials",
                ),
                ToolSpec(
                    fn=self._search_secrets,
                    display_label="Searching credentials",
                ),
                ToolSpec(
                    fn=self._list_secret_keys,
                    display_label="Listing credential names",
                ),
                ToolSpec(
                    fn=self._list_secret_bindings,
                    display_label="Listing integration bindings",
                ),
                include_class_name=False,
            ),
        }
        self.add_tools("ask", ask_tools)
        update_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                ToolSpec(fn=self.ask, display_label="Querying credentials"),
                ToolSpec(
                    fn=self._create_secret,
                    display_label="Storing a new credential",
                ),
                ToolSpec(fn=self._update_secret, display_label="Updating a credential"),
                ToolSpec(fn=self._delete_secret, display_label="Deleting a credential"),
                ToolSpec(
                    fn=self._bind_secret,
                    display_label="Binding an integration to a credential",
                ),
                ToolSpec(
                    fn=self._unbind_secret,
                    display_label="Removing an integration binding",
                ),
                ToolSpec(
                    fn=self._set_hive_default,
                    display_label="Setting a Hive-wide default credential",
                ),
                include_class_name=False,
            ),
        }
        self.add_tools("update", update_tools)

        try:
            self._ensure_dotenv_synced_on_init()
        except Exception:
            # Best-effort – local file sync must never break construction
            pass

    def _provision_storage(self) -> None:
        """Bind a ``TableStore`` for the shared credential vault.

        The overlay contexts (``SecretBinding``, ``SecretDefault``,
        ``OAuthTokens``) are provisioned by :class:`ContextRegistry`
        via their :class:`TableContext` declarations and don't need a
        local store handle — :meth:`_list_columns` only reflects the
        shared vault columns, and every other code path reads or
        writes through the Unify SDK directly.
        """
        self._store = TableStore(
            self._ctx,
            unique_keys={"secret_id": "int", "name": "str"},
            auto_counting={"secret_id": None},
            description="Shared credential vault.",
            fields=model_to_fields(Secret),
        )

    def warm_embeddings(self) -> None:
        self._ensure_description_vector()

    @functools.cache
    def _ensure_description_vector(self) -> None:
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
        for ctx in (self._ctx, self._binding_ctx, self._default_ctx, self._oauth_ctx):
            try:
                unify.delete_context(ctx)
            except Exception:
                pass

        for table in (
            SECRETS_TABLE,
            SECRET_BINDING_TABLE,
            SECRET_DEFAULT_TABLE,
            OAUTH_TOKENS_TABLE,
        ):
            ContextRegistry.refresh(self, table)

        self._ctx = ContextRegistry.get_context(self, SECRETS_TABLE)
        self._binding_ctx = ContextRegistry.get_context(self, SECRET_BINDING_TABLE)
        self._default_ctx = ContextRegistry.get_context(self, SECRET_DEFAULT_TABLE)
        self._oauth_ctx = ContextRegistry.get_context(self, OAUTH_TOKENS_TABLE)

        self._provision_storage()

        try:
            import time as _time

            for _ in range(3):
                try:
                    unify.get_fields(context=self._ctx)
                    break
                except Exception:
                    _time.sleep(0.05)
        except Exception:
            pass

    # --------------------- Internal helpers (LLM client/policies) --------------------- #

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
        """Require 'ask' on the first step (if enabled); auto thereafter."""
        from unity.settings import SETTINGS

        if (
            SETTINGS.FIRST_MUTATION_TOOL_IS_ASK
            and step_index < 1
            and "ask" in current_tools
        ):
            return ("required", {"ask": current_tools["ask"]})
        return ("auto", current_tools)

    # --------------------- Internal helpers (assistant OAuth sync) ---------- #

    OAUTH_SECRET_ALLOWLIST = frozenset(
        {
            "GOOGLE_ACCESS_TOKEN",
            "GOOGLE_REFRESH_TOKEN",
            "GOOGLE_TOKEN_EXPIRES_AT",
            "GOOGLE_GRANTED_SCOPES",
            "MICROSOFT_ACCESS_TOKEN",
            "MICROSOFT_REFRESH_TOKEN",
            "MICROSOFT_TOKEN_EXPIRES_AT",
            "MICROSOFT_GRANTED_SCOPES",
        },
    )

    def _sync_assistant_secrets(self) -> None:
        """Sync OAuth tokens from Orchestra into the per-body OAuth context.

        OAuth access/refresh tokens are identity material tied to the
        specific body that completed the OAuth flow. Storing them on
        the per-body :data:`OAUTH_TOKENS_TABLE` context keeps them off
        the Hive-shared credential vault so another body in the same
        Hive never sees, exports, or overwrites them.

        Best-effort: transient failures are swallowed.
        """
        agent_id = SESSION_DETAILS.assistant.agent_id
        if agent_id is None:
            return

        base_url = SETTINGS.ORCHESTRA_URL
        admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()
        if not base_url or not admin_key:
            return

        try:
            from unify.utils import http

            resp = http.get(
                f"{base_url}/admin/assistant",
                headers={"Authorization": f"Bearer {admin_key}"},
                params={"agent_id": int(agent_id), "from_fields": "secrets"},
                timeout=15,
            )
            if resp.status_code != 200:
                return
            payload = resp.json()
            items = payload.get("info", [])
            if not items:
                return
            secrets_dict: dict = items[0].get("secrets") or {}
        except Exception:
            return

        description = "System-managed OAuth credential (auto-synced)"
        for name, value in secrets_dict.items():
            if name not in self.OAUTH_SECRET_ALLOWLIST:
                continue
            if not isinstance(value, str) or not value:
                continue
            try:
                existing = unify.get_logs(
                    context=self._oauth_ctx,
                    filter=f"name == {name!r}",
                    limit=1,
                    return_ids_only=True,
                )
                if existing:
                    unify.update_logs(
                        logs=[existing[0]],
                        context=self._oauth_ctx,
                        entries={"value": value, "description": description},
                        overwrite=True,
                    )
                else:
                    unity_log(
                        context=self._oauth_ctx,
                        name=name,
                        value=value,
                        description=description,
                        new=True,
                        mutable=True,
                        add_to_all_context=False,
                    )
                self._env_set(name, value)
            except Exception:
                continue

        for stale_name in self.OAUTH_SECRET_ALLOWLIST - secrets_dict.keys():
            try:
                ids = unify.get_logs(
                    context=self._oauth_ctx,
                    filter=f"name == {stale_name!r}",
                    limit=1,
                    return_ids_only=True,
                )
                if ids:
                    unify.delete_logs(context=self._oauth_ctx, logs=ids[0])
                    self._env_remove(stale_name)
            except Exception:
                continue

    # --------------------- Internal helpers (.env sync) --------------------- #
    def _dotenv_path(self) -> str:
        """Return the path to the .env file used for local sync.

        Honors UNITY_SECRET_DOTENV_PATH from SETTINGS; defaults to ".env" in CWD.
        """
        if SETTINGS.secret.DOTENV_PATH:
            return SETTINGS.secret.DOTENV_PATH
        from unity.file_manager.settings import get_local_root

        return os.path.join(get_local_root(), ".env")

    def _is_hive_member(self) -> bool:
        """Return True when the active body belongs to a Hive."""
        return SESSION_DETAILS.hive_id is not None

    def _resolved_vault_entries(self) -> Dict[str, str]:
        """Return the ``{name: value}`` vault slice this body should export.

        The slice is computed differently depending on whether the body
        lives in a Hive:

        * **Hive member**: every :class:`SecretBinding` row on *this*
          body plus every :class:`SecretDefault` row the Hive declares
          that the body has not overridden with its own binding.
          ``os.environ`` (and the mirrored ``.env`` file) therefore
          carry exactly the credentials the Hive has sanctioned for
          this body — never the full shared vault.
        * **Solo body**: the full per-body vault — every stored
          credential is visible to the body that owns the single
          :data:`SECRETS_TABLE`.
        """
        if not self._is_hive_member():
            return self._all_vault_entries()

        selected_ids: Dict[int, None] = {}
        try:
            binding_rows = unify.get_logs(
                context=self._binding_ctx,
                from_fields=["integration", "secret_id"],
            )
        except Exception as exc:
            raise SecretBackendReadError(
                f"Failed to read SecretBinding rows from {self._binding_ctx!r}.",
            ) from exc
        bound_integrations: set[str] = set()
        for row in binding_rows:
            entries = row.entries or {}
            integration = entries.get("integration")
            secret_id = entries.get("secret_id")
            if isinstance(integration, str) and integration:
                bound_integrations.add(integration)
            if isinstance(secret_id, int) and secret_id >= 0:
                selected_ids[int(secret_id)] = None

        try:
            default_rows = unify.get_logs(
                context=self._default_ctx,
                from_fields=["integration", "secret_id"],
            )
        except Exception as exc:
            raise SecretBackendReadError(
                f"Failed to read SecretDefault rows from {self._default_ctx!r}.",
            ) from exc
        for row in default_rows:
            entries = row.entries or {}
            integration = entries.get("integration")
            if isinstance(integration, str) and integration in bound_integrations:
                continue
            secret_id = entries.get("secret_id")
            if isinstance(secret_id, int) and secret_id >= 0:
                selected_ids[int(secret_id)] = None

        if not selected_ids:
            return {}

        return self._vault_entries_by_id(list(selected_ids.keys()))

    def _all_vault_entries(self) -> Dict[str, str]:
        """Return every ``{name: value}`` from the shared vault (solo path)."""
        try:
            rows = unify.get_logs(context=self._ctx)
        except Exception as exc:
            raise SecretBackendReadError(
                f"Failed to read Secret rows from {self._ctx!r}.",
            ) from exc
        out: Dict[str, str] = {}
        for lg in rows:
            entries = lg.entries or {}
            name = entries.get("name")
            value = entries.get("value")
            if isinstance(name, str) and name and isinstance(value, str):
                out[name] = value
        return out

    def _vault_entries_by_id(self, secret_ids: List[int]) -> Dict[str, str]:
        """Fetch the ``{name: value}`` map for a set of vault rows by id."""
        if not secret_ids:
            return {}
        id_list = ",".join(str(int(sid)) for sid in secret_ids)
        try:
            rows = unify.get_logs(
                context=self._ctx,
                filter=f"secret_id in [{id_list}]",
                from_fields=["secret_id", "name", "value"],
            )
        except Exception as exc:
            raise SecretBackendReadError(
                f"Failed to read Secret rows from {self._ctx!r}.",
            ) from exc
        out: Dict[str, str] = {}
        for lg in rows:
            entries = lg.entries or {}
            name = entries.get("name")
            value = entries.get("value")
            if isinstance(name, str) and name and isinstance(value, str):
                out[name] = value
        return out

    def _oauth_entries(self) -> Dict[str, str]:
        """Return every ``{name: value}`` from the per-body OAuth store."""
        try:
            rows = unify.get_logs(
                context=self._oauth_ctx,
                from_fields=["name", "value"],
            )
        except Exception as exc:
            raise SecretBackendReadError(
                f"Failed to read OAuth token rows from {self._oauth_ctx!r}.",
            ) from exc
        out: Dict[str, str] = {}
        for lg in rows:
            entries = lg.entries or {}
            name = entries.get("name")
            value = entries.get("value")
            if isinstance(name, str) and name and isinstance(value, str):
                out[name] = value
        return out

    def _sync_dotenv(self) -> None:
        """Export this body's credential slice to ``.env`` and ``os.environ``.

        Hive members export only the subset of vault rows their
        :class:`SecretBinding` overlay (plus any Hive-wide defaults)
        has selected; solo bodies export the full vault. OAuth tokens
        from the per-body :data:`OAUTH_TOKENS_TABLE` are always
        layered on top so Google/Microsoft-backed tools find them in
        ``os.environ`` regardless of binding state.

        Keys that were previously exported by this manager but no
        longer appear in the resolved slice (for example after
        ``_unbind_secret`` retires an integration) are removed from
        both ``os.environ`` and the ``.env`` file so pooled venvs
        spawned after the next invalidation pass don't inherit a
        stale credential.
        """
        try:
            vault_entries = self._resolved_vault_entries()
            oauth_entries = self._oauth_entries()
        except SecretBackendReadError as exc:
            logger.warning(
                "Skipping credential export after backend read failure: %s",
                exc,
            )
            return

        combined: Dict[str, str] = {}
        combined.update(vault_entries)
        combined.update(oauth_entries)

        stale = sorted(self._exported_env_keys - combined.keys())
        if not combined and not stale:
            return
        self._env_merge_and_write(
            add_or_update=combined or None,
            remove_keys=stale or None,
        )
        self._exported_env_keys = set(combined)

    def _schedule_venv_invalidation(self) -> None:
        """Best-effort retirement of pooled venv subprocesses.

        Called after a credential mutation so already-spawned venv
        subprocesses — which captured the old ``os.environ`` at spawn
        time — are retired and the next execution spawns with the
        refreshed environment. When no event loop is running (e.g.
        test or script context) the invalidation is skipped: there
        are no pooled subprocesses to retire in that case.

        The scheduled task is retained on :attr:`_pending_tasks` so
        the asyncio garbage collector cannot reap it before it runs;
        the done-callback removes it once complete.
        """
        from unity.function_manager.function_manager import VenvPool

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(VenvPool.invalidate_all_pools())
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    def _republish_credentials(self) -> None:
        """Re-export the current credential slice and retire stale venvs.

        Called after every binding/default mutation so this body's
        ``.env`` and ``os.environ`` track the latest resolution and
        any pooled venv subprocesses that captured the previous
        environment at spawn time are retired.
        """
        try:
            self._sync_dotenv()
        except Exception:
            pass
        self._schedule_venv_invalidation()

    def _upsert_integration_overlay(
        self,
        context: str,
        *,
        integration: str,
        secret_id: int,
    ) -> None:
        """Upsert a per-integration overlay row keyed on ``integration``.

        Writes to either the :data:`SECRET_BINDING_TABLE` (per-body)
        or :data:`SECRET_DEFAULT_TABLE` (Hive-wide) context, depending
        on the caller. The ``authoring_assistant_id`` audit stamp is
        preserved across overwrites so the body that first installed
        the row remains attributed even when a different body later
        repoints it.
        """
        existing = unify.get_logs(
            context=context,
            filter=f"integration == {integration!r}",
            limit=1,
            from_fields=["authoring_assistant_id"],
        )
        if existing:
            entries = existing[0].entries or {}
            updates: Dict[str, Any] = {"secret_id": int(secret_id)}
            if entries.get("authoring_assistant_id") is None:
                updates["authoring_assistant_id"] = authoring_assistant_id()
            unify.update_logs(
                logs=[existing[0].id],
                context=context,
                entries=updates,
                overwrite=True,
            )
            return
        unity_log(
            context=context,
            integration=integration,
            secret_id=int(secret_id),
            authoring_assistant_id=authoring_assistant_id(),
            new=True,
            mutable=True,
            add_to_all_context=False,
        )

    def _ensure_dotenv_synced_on_init(self) -> None:
        """Create .env if missing, pull OAuth tokens, and merge resolved secrets."""
        path = self._dotenv_path()
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        except Exception:
            pass
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("")

        self._sync_assistant_secrets()
        self._sync_dotenv()

    @staticmethod
    def _parse_env_lines(lines: List[str]) -> Dict[str, int]:
        """Return mapping of existing KEY -> line index for a simple .env file."""
        import re

        key_to_idx: Dict[str, int] = {}
        for idx, raw in enumerate(lines):
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_/]*)\s*=", raw)
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

        if remove_keys:
            rm = set(remove_keys)

            def _keep(i: int, s: str) -> bool:
                for k, j in key_to_idx.items():
                    if j == i and k in rm:
                        return False
                return True

            lines = [s for i, s in enumerate(lines) if _keep(i, s)]
            key_to_idx = self._parse_env_lines(lines)

        if add_or_update:
            for key, value in add_or_update.items():
                line = f"{key}={value}"
                if key in key_to_idx:
                    lines[key_to_idx[key]] = line
                else:
                    lines.append(line)
                os.environ[key] = value

        if remove_keys:
            for key in remove_keys:
                os.environ.pop(key, None)

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
        """Resolve ${name} placeholders in text to raw secret values (no LLM)."""
        return self._resolve_placeholders(text)

    async def to_placeholder(self, text: str) -> str:
        """Convert known secret values in text to ``${name}`` placeholders."""
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
                    if val in value_to_name:
                        if nm < value_to_name[val]:
                            value_to_name[val] = nm
                    else:
                        value_to_name[val] = nm
            except Exception:
                continue

        import re

        ordered_values = sorted(value_to_name.keys(), key=len, reverse=True)
        result = text
        for val in ordered_values:
            name = value_to_name[val]
            pattern = re.escape(val)
            placeholder = f"${{{name}}}"
            result = re.sub(pattern, placeholder, result)

        return result

    def get_secret_for_integration(self, integration: str) -> Secret:
        """Resolve the :class:`Secret` the active body should use for an integration.

        Resolution order:

        1. The body's :class:`SecretBinding` row for *integration* (if any).
        2. The Hive-wide :class:`SecretDefault` for *integration* (Hive
           members only).
        3. Solo bodies fall back to a vault row whose ``name`` equals
           *integration*, matching the single-body convention where the
           integration name doubles as the credential name.

        Raises
        ------
        LookupError
            When no binding, Hive default, nor matching vault row is
            found. Callers should treat this as a configuration issue
            (the integration has not been wired up on this body) rather
            than a transient failure.
        """
        if not integration:
            raise ValueError("integration name must be a non-empty string")

        try:
            binding_rows = unify.get_logs(
                context=self._binding_ctx,
                filter=f"integration == {integration!r}",
                limit=1,
                from_fields=["integration", "secret_id"],
            )
        except Exception as exc:
            raise SecretBackendReadError(
                f"Failed to read SecretBinding rows from {self._binding_ctx!r}.",
            ) from exc
        if binding_rows:
            secret_id = (binding_rows[0].entries or {}).get("secret_id")
            if isinstance(secret_id, int) and secret_id >= 0:
                resolved = self._load_secret(f"secret_id == {int(secret_id)}")
                if resolved is not None:
                    return resolved

        if self._is_hive_member():
            try:
                default_rows = unify.get_logs(
                    context=self._default_ctx,
                    filter=f"integration == {integration!r}",
                    limit=1,
                    from_fields=["integration", "secret_id"],
                )
            except Exception as exc:
                raise SecretBackendReadError(
                    f"Failed to read SecretDefault rows from {self._default_ctx!r}.",
                ) from exc
            if default_rows:
                secret_id = (default_rows[0].entries or {}).get("secret_id")
                if isinstance(secret_id, int) and secret_id >= 0:
                    resolved = self._load_secret(f"secret_id == {int(secret_id)}")
                    if resolved is not None:
                        return resolved
        else:
            fallback = self._load_secret(f"name == {integration!r}")
            if fallback is not None:
                return fallback

        raise LookupError(
            f"No credential found for integration {integration!r}: no "
            "SecretBinding on this body"
            + (
                " and no Hive-wide SecretDefault"
                if self._is_hive_member()
                else " and no vault row matches the integration name"
            )
            + ".",
        )

    def _load_secret(self, filter_expr: str) -> Optional[Secret]:
        """Return the :class:`Secret` matching ``filter_expr`` or ``None``.

        Callers pass a single-row Unify filter expression against
        :attr:`_ctx` (for example ``"secret_id == 7"`` or
        ``"name == 'Salesforce Admin'"``) and get back the full
        credential row — or ``None`` when the backend returns no match.
        Backend read failures raise :class:`SecretBackendReadError` so
        callers do not mistake a transient outage for a missing
        credential configuration.
        """
        try:
            rows = unify.get_logs(
                context=self._ctx,
                filter=filter_expr,
                limit=1,
                from_fields=[
                    "secret_id",
                    "name",
                    "value",
                    "description",
                    "authoring_assistant_id",
                ],
            )
        except Exception as exc:
            raise SecretBackendReadError(
                f"Failed to read Secret rows from {self._ctx!r}.",
            ) from exc
        if not rows:
            return None
        entries = rows[0].entries or {}
        return Secret(
            secret_id=int(entries.get("secret_id", -1)),
            name=entries.get("name") or "",
            value=entries.get("value") or "",
            description=entries.get("description") or "",
            authoring_assistant_id=entries.get("authoring_assistant_id"),
        )

    def _resolve_binding_target(self, secret_name: str) -> int:
        """Resolve a user-supplied secret name to its stable ``secret_id``.

        The public binding surface takes human-friendly names so the
        caller never has to look up an id. This helper enforces name
        uniqueness on the *read* side — a caller cannot bind to an
        ambiguous name.
        """
        rows = unify.get_logs(
            context=self._ctx,
            filter=f"name == {secret_name!r}",
            limit=2,
            from_fields=["secret_id"],
        )
        if not rows:
            raise ValueError(f"No credential named {secret_name!r} exists.")
        if len(rows) > 1:
            raise RuntimeError(
                f"Multiple credentials named {secret_name!r}; cannot resolve.",
            )
        secret_id = (rows[0].entries or {}).get("secret_id")
        if not isinstance(secret_id, int) or secret_id < 0:
            raise RuntimeError(
                f"Credential {secret_name!r} has no valid secret_id on record.",
            )
        return int(secret_id)

    @functools.wraps(BaseSecretManager.ask, updated=())
    @log_manager_call(
        "SecretManager",
        "ask",
        payload_key="question",
        display_label="Checking credentials",
    )
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
        try:
            self._sync_assistant_secrets()
        except Exception:
            pass
        try:
            self._sync_dotenv()
        except Exception:
            pass

        try:
            text = await self.to_placeholder(text)
        except Exception:
            pass

        client = new_llm_client()

        tools = dict(self.get_tools("ask"))
        _clar_queues = None
        _on_clar_req = None
        _on_clar_ans = None
        if _clarification_up_q is not None and _clarification_down_q is not None:
            from ..common.llm_helpers import make_request_clarification_tool

            _clar_queues = (_clarification_up_q, _clarification_down_q)
            tools["request_clarification"] = make_request_clarification_tool(None, None)

            async def _on_clar_req(q: str):
                try:
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
                except Exception:
                    pass

            async def _on_clar_ans(ans: str):
                try:
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
                except Exception:
                    pass

        client.set_system_message(
            build_ask_prompt(tools=tools).to_list(),
        )

        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=self._default_ask_tool_policy,
            response_format=response_format,
            handle_cls=(
                ReadOnlyAskGuardHandle if SETTINGS.UNITY_READONLY_ASK_GUARD else None
            ),
            clarification_queues=_clar_queues,
            on_clarification_request=_on_clar_req,
            on_clarification_answer=_on_clar_ans,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    @functools.wraps(BaseSecretManager.update, updated=())
    @log_manager_call(
        "SecretManager",
        "update",
        payload_key="request",
        display_label="Updating credentials",
    )
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
        try:
            text = await self.to_placeholder(text)
        except Exception:
            pass

        client = new_llm_client()

        tools = dict(self.get_tools("update"))
        _clar_queues = None
        _on_clar_req = None
        _on_clar_ans = None
        if _clarification_up_q is not None and _clarification_down_q is not None:
            from ..common.llm_helpers import make_request_clarification_tool

            _clar_queues = (_clarification_up_q, _clarification_down_q)
            tools["request_clarification"] = make_request_clarification_tool(None, None)

            async def _on_clar_req(q: str):
                try:
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
                except Exception:
                    pass

            async def _on_clar_ans(ans: str):
                try:
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
                except Exception:
                    pass

        client.set_system_message(
            build_update_prompt(tools=tools).to_list(),
        )

        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.update.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=self._default_update_tool_policy,
            response_format=response_format,
            clarification_queues=_clar_queues,
            on_clarification_request=_on_clar_req,
            on_clarification_answer=_on_clar_ans,
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
        """Replace ``${name}`` with the matching vault value (unknown names untouched)."""
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
            return match.group(0)

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
        """
        cols = self._store.get_columns()
        return cols if include_types else list(cols)

    def _sanitize_secret_references(
        self,
        references: Optional[Dict[str, str]],
    ) -> Optional[Dict[str, str]]:
        """Return a safe subset of references limited to description-based terms."""
        if not references:
            return references

        allowed: Dict[str, str] = {}
        for source_expr, ref_text in references.items():
            try:
                if is_plain_identifier(source_expr):
                    if source_expr == "description":
                        allowed[source_expr] = ref_text
                    continue

                import re as _re

                placeholders = _re.findall(
                    r"\{\s*([a-zA-Z_][\w]*)\s*\}",
                    source_expr or "",
                )
                if placeholders and all(ph == "description" for ph in placeholders):
                    allowed[source_expr] = ref_text
            except Exception:
                continue

        return allowed or None

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
            Mapping of source expressions to reference text (use
            ``"description"``). When None, returns most-recent rows.
        k : int, default 10
            Maximum number of results (<= 1000).
        """
        self._ensure_description_vector()
        safe_refs = self._sanitize_secret_references(references)

        rows = table_search_top_k(
            context=self._ctx,
            references=safe_refs,
            k=k,
            allowed_fields=[
                "secret_id",
                "name",
                "description",
                "authoring_assistant_id",
            ],
            row_filter=None,
            unique_id_field="name",
        )
        return [
            Secret(
                secret_id=(
                    int(r.get("secret_id")) if r.get("secret_id") is not None else -1
                ),
                name=r.get("name"),
                value="",
                description=r.get("description") or "",
                authoring_assistant_id=r.get("authoring_assistant_id"),
            )
            for r in rows
        ]

    def _filter_secrets(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Secret]:
        """Filter secrets using a boolean expression evaluated per row."""
        normalized = normalize_filter_expr(filter)
        logs = unify.get_logs(
            context=self._ctx,
            filter=normalized,
            offset=offset,
            limit=limit,
            from_fields=[
                "secret_id",
                "name",
                "description",
                "authoring_assistant_id",
            ],
        )
        return [
            Secret(
                secret_id=(
                    int(lg.entries.get("secret_id"))
                    if lg.entries.get("secret_id") is not None
                    else -1
                ),
                name=lg.entries.get("name"),
                value="",
                description=lg.entries.get("description") or "",
                authoring_assistant_id=lg.entries.get("authoring_assistant_id"),
            )
            for lg in logs
        ]

    def _list_secret_keys(self) -> List[str]:
        """Return every credential name stored in the shared vault."""
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

    def _list_secret_bindings(self) -> List[SecretBinding]:
        """Return every integration binding held by this body."""
        try:
            rows = unify.get_logs(
                context=self._binding_ctx,
                from_fields=["integration", "secret_id", "authoring_assistant_id"],
            )
        except Exception:
            rows = []
        out: List[SecretBinding] = []
        for lg in rows:
            entries = lg.entries or {}
            try:
                out.append(SecretBinding.model_validate(entries))
            except Exception:
                continue
        return out

    # --------------------- Tools (mutations) --------------------- #
    def _create_secret(
        self,
        *,
        name: str,
        value: str,
        description: Optional[str] = None,
    ) -> ToolOutcome:
        """Create a credential in the shared vault.

        The shared vault is Hive-scoped: every body in a Hive writes
        to the same ``Hives/{hive_id}/Secrets`` table. Two rows with
        distinct ``name`` values may represent alternate credentials
        for the same service (for example ``"Salesforce Admin"`` and
        ``"Salesforce User"``); which one a body uses for a given
        integration is expressed through :meth:`_bind_secret` rather
        than by rewriting this row.
        """
        assert name and value, "Both name and value are required."
        existing = unify.get_logs(
            context=self._ctx,
            filter=f"name == {name!r}",
            limit=1,
            return_ids_only=True,
        )
        assert not existing, f"Secret with name '{name}' already exists."

        entries = {
            "name": name,
            "value": value,
            "description": description or "",
            "authoring_assistant_id": authoring_assistant_id(),
        }
        unity_log(
            context=self._ctx,
            **entries,
            new=True,
            mutable=True,
            add_to_all_context=self.include_in_multi_assistant_table,
        )

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
        """Update the value or description of an existing credential.

        ``authoring_assistant_id`` is a write-once audit stamp and is
        never rewritten — the original author remains attributed even
        when another body in the Hive rotates the credential.
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

        try:
            if value is not None:
                self._env_set(name, value)
        except Exception:
            pass

        return {"outcome": "secret updated", "details": {"name": name}}

    def _delete_secret(self, *, name: str) -> ToolOutcome:
        """Delete a credential by name."""
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
        try:
            self._env_remove(name)
        except Exception:
            pass
        return {"outcome": "secret deleted", "details": {"name": name}}

    def _bind_secret(
        self,
        *,
        integration: str,
        secret_name: str,
    ) -> ToolOutcome:
        """Bind *integration* on this body to an existing credential.

        Takes the human-friendly ``secret_name`` and resolves it to a
        stable ``secret_id`` before writing the overlay row, so
        renaming a vault credential later does not silently retarget
        any binding.

        A subsequent call with the same *integration* overwrites the
        overlay in place. The ``authoring_assistant_id`` stamp is
        preserved across overwrites so the body that first installed
        the binding remains attributed.
        """
        if not integration:
            raise ValueError("integration name must be a non-empty string.")
        if not secret_name:
            raise ValueError("secret_name must be a non-empty string.")

        secret_id = self._resolve_binding_target(secret_name)
        self._upsert_integration_overlay(
            self._binding_ctx,
            integration=integration,
            secret_id=secret_id,
        )
        self._republish_credentials()

        return {
            "outcome": "integration bound",
            "details": {"integration": integration, "secret_name": secret_name},
        }

    def _unbind_secret(self, *, integration: str) -> ToolOutcome:
        """Remove this body's binding for *integration*.

        After unbinding, :meth:`get_secret_for_integration` falls back
        to the Hive-wide :class:`SecretDefault` (for Hive members) or
        to ``name == integration`` (for solo bodies).
        """
        if not integration:
            raise ValueError("integration name must be a non-empty string.")
        ids = unify.get_logs(
            context=self._binding_ctx,
            filter=f"integration == {integration!r}",
            limit=1,
            return_ids_only=True,
        )
        if not ids:
            return {
                "outcome": "no binding to remove",
                "details": {"integration": integration},
            }
        unify.delete_logs(context=self._binding_ctx, logs=ids[0])
        self._republish_credentials()

        return {
            "outcome": "integration unbound",
            "details": {"integration": integration},
        }

    def _set_hive_default(
        self,
        *,
        integration: str,
        secret_name: str,
    ) -> ToolOutcome:
        """Install a Hive-wide default credential for *integration*.

        Only meaningful inside a Hive — solo bodies do not consult
        :class:`SecretDefault` and the method therefore raises.
        """
        if not self._is_hive_member():
            raise RuntimeError(
                "SecretDefault is Hive-wide and cannot be installed on a solo body.",
            )
        if not integration:
            raise ValueError("integration name must be a non-empty string.")
        if not secret_name:
            raise ValueError("secret_name must be a non-empty string.")

        secret_id = self._resolve_binding_target(secret_name)
        self._upsert_integration_overlay(
            self._default_ctx,
            integration=integration,
            secret_id=secret_id,
        )
        self._republish_credentials()

        return {
            "outcome": "hive default set",
            "details": {"integration": integration, "secret_name": secret_name},
        }
