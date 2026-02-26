"""
Code-first client customization registry.

Client configs and environments are defined in Python code under per-client
subpackages (e.g. ``colliers/``, ``midland_heart/``).  Each subpackage
registers its org-level, user-level, and/or assistant-level customizations
into the module-level dicts below.

At Actor construction time, ``resolve()`` is called with the current
org_id / user_id / assistant_id to produce a merged ``ActorConfig`` and
a combined list of ``BaseEnvironment`` instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from unity.customization.configs.types.actor_config import ActorConfig

if TYPE_CHECKING:
    from unity.actor.environments.base import BaseEnvironment

# ---------------------------------------------------------------------------
# Registry dicts — populated by client subpackages at import time
# ---------------------------------------------------------------------------

_ORG_CONFIGS: dict[int, ActorConfig] = {}
_ORG_ENVIRONMENTS: dict[int, list[BaseEnvironment]] = {}

_USER_CONFIGS: dict[str, ActorConfig] = {}
_USER_ENVIRONMENTS: dict[str, list[BaseEnvironment]] = {}

_ASSISTANT_CONFIGS: dict[int, ActorConfig] = {}
_ASSISTANT_ENVIRONMENTS: dict[int, list[BaseEnvironment]] = {}


# ---------------------------------------------------------------------------
# Public helpers for client subpackages to register themselves
# ---------------------------------------------------------------------------


def register_org(
    org_id: int,
    *,
    config: ActorConfig | None = None,
    environments: list[BaseEnvironment] | None = None,
) -> None:
    if config is not None:
        _ORG_CONFIGS[org_id] = config
    if environments:
        _ORG_ENVIRONMENTS[org_id] = environments


def register_user(
    user_id: str,
    *,
    config: ActorConfig | None = None,
    environments: list[BaseEnvironment] | None = None,
) -> None:
    if config is not None:
        _USER_CONFIGS[user_id] = config
    if environments:
        _USER_ENVIRONMENTS[user_id] = environments


def register_assistant(
    assistant_id: int,
    *,
    config: ActorConfig | None = None,
    environments: list[BaseEnvironment] | None = None,
) -> None:
    if config is not None:
        _ASSISTANT_CONFIGS[assistant_id] = config
    if environments:
        _ASSISTANT_ENVIRONMENTS[assistant_id] = environments


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _merge_configs(configs: list[ActorConfig]) -> ActorConfig:
    """Merge a list of configs in precedence order (first = least specific).

    For scalar fields, the last non-None value wins (more specific overrides
    less specific).  For ``guidelines``, all non-None values are concatenated
    with newlines (additive).
    """
    if not configs:
        return ActorConfig()

    merged: dict = {}
    guidelines_parts: list[str] = []

    for cfg in configs:
        for field_name in cfg.model_fields:
            value = getattr(cfg, field_name)
            if value is None:
                continue
            if field_name == "guidelines":
                guidelines_parts.append(value)
            else:
                merged[field_name] = value

    if guidelines_parts:
        merged["guidelines"] = "\n".join(guidelines_parts)

    return ActorConfig(**merged)


def resolve(
    org_id: int | None = None,
    user_id: str | None = None,
    assistant_id: int | None = None,
) -> tuple[ActorConfig, list[BaseEnvironment]]:
    """Resolve merged config and environments for the given identity.

    Cascade order (least to most specific): org -> user -> assistant.
    Configs are merged (scalars: last non-None wins; guidelines: concatenated).
    Environments are purely additive across all levels, deduplicated by name.
    """
    configs: list[ActorConfig] = []
    environments: list[BaseEnvironment] = []

    if org_id is not None:
        if org_id in _ORG_CONFIGS:
            configs.append(_ORG_CONFIGS[org_id])
        if org_id in _ORG_ENVIRONMENTS:
            environments.extend(_ORG_ENVIRONMENTS[org_id])

    if user_id is not None:
        if user_id in _USER_CONFIGS:
            configs.append(_USER_CONFIGS[user_id])
        if user_id in _USER_ENVIRONMENTS:
            environments.extend(_USER_ENVIRONMENTS[user_id])

    if assistant_id is not None:
        if assistant_id in _ASSISTANT_CONFIGS:
            configs.append(_ASSISTANT_CONFIGS[assistant_id])
        if assistant_id in _ASSISTANT_ENVIRONMENTS:
            environments.extend(_ASSISTANT_ENVIRONMENTS[assistant_id])

    merged_config = _merge_configs(configs)

    # Deduplicate environments by name, keeping the last occurrence
    # (more specific level wins if names collide).
    seen: dict[str, BaseEnvironment] = {}
    for env in environments:
        seen[env.name] = env
    deduped = list(seen.values())

    return merged_config, deduped


# ---------------------------------------------------------------------------
# Import client subpackages so they self-register.
# Add new clients here.
# ---------------------------------------------------------------------------

from unity.customization.clients import colliers as _colliers  # noqa: F401, E402
from unity.customization.clients import (
    midland_heart as _midland_heart,
)  # noqa: F401, E402
