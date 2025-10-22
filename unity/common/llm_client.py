from __future__ import annotations


import json
import os

import unify


def new_llm_client(
    model: str = "gpt-5@openai",
    *,
    stateful: bool = False,
) -> "unify.AsyncUnify":
    """
    Create a configured AsyncUnify client for gpt-5@openai with sane defaults.

    The model is intentionally hard-coded to "gpt-5@openai" to align with
    manager expectations around reasoning_effort and service_tier semantics.
    """
    selected_model = "gpt-5@openai"
    return unify.AsyncUnify(
        selected_model,
        cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
        traced=json.loads(os.environ.get("UNIFY_TRACED", "false")),
        reasoning_effort="high",
        service_tier="priority",
        stateful=stateful,
    )
