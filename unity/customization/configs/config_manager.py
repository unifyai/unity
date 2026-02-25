from __future__ import annotations

import functools
import logging
from typing import Any, Tuple

import unify

from ...common.log_utils import log as unity_log
from ...common.model_to_fields import model_to_fields
from ...common.context_store import TableStore
from ...common.context_registry import TableContext, ContextRegistry
from .base import BaseConfigManager
from .types.actor_config import ActorConfig

logger = logging.getLogger(__name__)


class ConfigManager(BaseConfigManager):
    """Concrete ConfigManager backed by Unify contexts."""

    class Config:
        required_contexts = [
            TableContext(
                name="Configs/Actor",
                description="Stored actor configuration for CodeActActor construction.",
                fields=model_to_fields(ActorConfig),
            ),
        ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs.get("read"), ctxs.get("write")
        if not read_ctx:
            try:
                from ... import ensure_initialised as _ensure_initialised

                _ensure_initialised()
                ctxs = unify.get_active_context()
                read_ctx, write_ctx = ctxs.get("read"), ctxs.get("write")
            except Exception:
                pass

        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a ConfigManager."

        self.include_in_multi_assistant_table = True
        self._ctx = ContextRegistry.get_context(self, "Configs/Actor")
        self._BUILTIN_FIELDS: Tuple[str, ...] = tuple(ActorConfig.model_fields.keys())

        self._provision_storage()

    def _provision_storage(self) -> None:
        self._store = TableStore(
            self._ctx,
            description="Stored actor configuration for CodeActActor construction.",
            fields=model_to_fields(ActorConfig),
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    @functools.wraps(BaseConfigManager.save_config, updated=())
    def save_config(self, config: ActorConfig) -> None:
        self._delete_existing()
        payload = config.to_post_json()
        if not payload:
            return
        unity_log(
            context=self._ctx,
            **payload,
            new=True,
            mutable=True,
            add_to_all_context=self.include_in_multi_assistant_table,
        )
        logger.info("Saved actor config: %s", payload)

    @functools.wraps(BaseConfigManager.load_config, updated=())
    def load_config(self) -> ActorConfig:
        logs = unify.get_logs(
            context=self._ctx,
            limit=1,
            from_fields=list(self._BUILTIN_FIELDS),
        )
        if not logs:
            return ActorConfig()
        return ActorConfig(**logs[0].entries)

    @functools.wraps(BaseConfigManager.clear, updated=())
    def clear(self) -> None:
        unify.delete_context(self._ctx)
        ContextRegistry.refresh(self, "Configs/Actor")
        self._provision_storage()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _delete_existing(self) -> None:
        ids = unify.get_logs(
            context=self._ctx,
            limit=100,
            return_ids_only=True,
        )
        if ids:
            for log_id in ids:
                unify.delete_logs(context=self._ctx, logs=log_id)
