from __future__ import annotations


import json
import os
from typing import Any

import unify


DEFAULT_MODEL = "gpt-5@openai"


def new_llm_client(
    model: str = DEFAULT_MODEL,
    *,
    stateful: bool = False,
    **kwargs: Any,
) -> "unify.AsyncUnify":
    """
    Create a configured AsyncUnify client.

    Defaults to "gpt-5@openai" with sane defaults for reasoning effort and service tier.
    """
    config = {
        "cache": json.loads(os.environ.get("UNIFY_CACHE", "true")),
        "traced": json.loads(os.environ.get("UNIFY_TRACED", "false")),
        "reasoning_effort": "high",
        "service_tier": "priority",
        "stateful": stateful,
    }
    config.update(kwargs)

    return unify.AsyncUnify(model, **config)
