from __future__ import annotations

from typing import Any

import unify

from unity.settings import SETTINGS

# Backward-compatible constant (now sourced from settings)
DEFAULT_MODEL = SETTINGS.UNIFY_MODEL


def get_cache_setting() -> bool | str:
    """Return the cache setting from SETTINGS.

    Backward-compatible wrapper. New code should use SETTINGS.UNIFY_CACHE directly.
    """
    return SETTINGS.UNIFY_CACHE


def new_llm_client(
    model: str | None = None,
    *,
    async_client: bool = True,
    stateful: bool = False,
    **kwargs: Any,
) -> "unify.AsyncUnify | unify.Unify":
    """
    Create a configured Unify client.

    If model is not specified, uses UNIFY_MODEL from settings (default: gpt-5.2@openai).
    Defaults to high reasoning_effort and priority service_tier where applicable.
    Returns an AsyncUnify client by default, or a synchronous Unify client when
    async_client=False.
    """
    if model is None:
        model = SETTINGS.UNIFY_MODEL

    config = {
        "cache": SETTINGS.UNIFY_CACHE,
        "reasoning_effort": "high",
        "service_tier": "priority",
        "stateful": stateful,
    }
    config.update(kwargs)

    if async_client:
        return unify.AsyncUnify(model, **config)
    return unify.Unify(model, **config)
