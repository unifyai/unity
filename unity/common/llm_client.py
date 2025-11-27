from __future__ import annotations


import json
import os
from typing import Any

import unify


DEFAULT_MODEL = "gemini-3-pro@vertex-ai"


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
        "cache": json.loads(os.environ.get("UNIFY_CACHE", "true")),
        "traced": json.loads(os.environ.get("UNIFY_TRACED", "false")),
        "reasoning_effort": "high",
        "service_tier": "priority",
        "stateful": stateful,
    }
    config.update(kwargs)

    if async_client:
        return unify.AsyncUnify(model, **config)
    return unify.Unify(model, **config)
