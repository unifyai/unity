from __future__ import annotations

import copy
from typing import Any

import unillm
from pydantic import BaseModel

from unity.settings import SETTINGS

# Backward-compatible constant (now sourced from settings)
DEFAULT_MODEL = SETTINGS.UNIFY_MODEL


def new_llm_client(
    model: str | None = None,
    *,
    async_client: bool = True,
    stateful: bool = False,
    **kwargs: Any,
) -> "unillm.AsyncUnify | unillm.Unify":
    """
    Create a configured Unify client.

    If model is not specified, uses UNIFY_MODEL from settings (default: claude-4.6-opus@anthropic).
    Defaults to low reasoning_effort and priority service_tier where applicable.
    Caching is controlled by the UNILLM_CACHE env var (owned by unillm).
    Returns an AsyncUnify client by default, or a synchronous Unify client when
    async_client=False.
    """
    if model is None:
        model = SETTINGS.UNIFY_MODEL

    config = {
        "reasoning_effort": "low",
        "service_tier": "priority",
        "stateful": stateful,
    }
    config.update(kwargs)

    if async_client:
        return unillm.AsyncUnify(model, **config)
    return unillm.Unify(model, **config)


def _make_openai_strict_json_schema_compatible(node: Any) -> None:
    """Mutate a JSON schema in-place to satisfy OpenAI strict requirements.

    OpenAI's strict JSON-schema mode requires that:
    - For any schema object with explicit `properties`, the `required` array
      must exist and include *every* property key.
    - `additionalProperties` must be false (to forbid extra keys).

    Pydantic excludes fields with default values from `required` (because they're
    optional at validation time). This helper normalizes the schema to the strict
    subset that OpenAI enforces.
    """
    if isinstance(node, dict):
        props = node.get("properties")
        if node.get("type") == "object" and isinstance(props, dict):
            node["additionalProperties"] = False
            node["required"] = list(props.keys())
        for v in node.values():
            _make_openai_strict_json_schema_compatible(v)
        return

    if isinstance(node, list):
        for v in node:
            _make_openai_strict_json_schema_compatible(v)


def pydantic_to_json_schema_response_format(
    response_model: type[BaseModel],
    *,
    name: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Build an OpenAI-style `response_format` dict from a Pydantic model.

    This returns the JSON-schema response format shape used by OpenAI:

        {"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": ...}}

    When `strict=True`, the schema is post-processed to satisfy OpenAI's strict
    constraints (see `_make_openai_strict_json_schema_compatible`).
    """
    schema = response_model.model_json_schema()
    if strict:
        schema = copy.deepcopy(schema)
        _make_openai_strict_json_schema_compatible(schema)
    else:
        # Keep the existing behaviour of forbidding unknown keys where possible.
        schema.setdefault("additionalProperties", False)
        for def_schema in schema.get("$defs", {}).values():
            if isinstance(def_schema, dict):
                def_schema.setdefault("additionalProperties", False)

    return {
        "type": "json_schema",
        "json_schema": {
            "name": name or response_model.__name__,
            "schema": schema,
            "strict": strict,
        },
    }
