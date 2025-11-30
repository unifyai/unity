from __future__ import annotations


import os
from typing import Any

import unify


DEFAULT_MODEL = "gpt-5.1@openai"


def get_cache_setting(default: bool | str = True) -> bool | str:
    """
    Parse the UNIFY_CACHE environment variable.

    Returns:
        - True if the value is "true", "yes", or "1" (case-insensitive) or not set
        - False if the value is "false", "no", or "0" (case-insensitive)
        - The string value as-is for any other cache mode (e.g., "read", "write",
          "read-only", "both", and their "-closest" variants)
    """
    raw = os.environ.get("UNIFY_CACHE")
    if raw is None:
        return default
    lower = raw.lower()
    if lower in ("true", "yes", "1"):
        return True
    if lower in ("false", "no", "0"):
        return False
    return raw


def new_llm_client(
    model: str = DEFAULT_MODEL,
    *,
    async_client: bool = True,
    stateful: bool = False,
    **kwargs: Any,
) -> "unify.AsyncUnify | unify.Unify":
    """
    Create a configured Unify client.

    Defaults to high reasoning_effort and priority service_tier where applicable (otherwise dropped).
    Returns an AsyncUnify client by default, or a synchronous Unify client when
    async_client=False.
    """
    config = {
        "cache": get_cache_setting(),
        "reasoning_effort": "high",
        "service_tier": "priority",
        "stateful": stateful,
    }
    config.update(kwargs)

    if async_client:
        return unify.AsyncUnify(model, **config)
    return unify.Unify(model, **config)
