from __future__ import annotations

import os
import json
import unify


def new_llm_client(model: str) -> "unify.AsyncUnify":
    """Construct a configured AsyncUnify client for the given model."""
    return unify.AsyncUnify(
        model,
        cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
        traced=json.loads(os.environ.get("UNIFY_TRACED", "false")),
        reasoning_effort="high",
        service_tier="priority",
    )
