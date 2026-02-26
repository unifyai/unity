"""
Code-first client customization registry.

Client configs, environments, and custom functions are defined in Python
code under per-client subpackages (e.g. ``colliers/``, ``midland_heart/``).
Each subpackage registers its org-level, user-level, and/or assistant-level
customizations into the module-level dicts below.

At Actor construction time, ``resolve()`` is called with the current
org_id / user_id / assistant_id to produce a merged ``ActorConfig``,
a combined list of ``BaseEnvironment`` instances, and the directories
containing ``@custom_function`` definitions and custom venvs to sync.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from unity.customization.configs.types.actor_config import ActorConfig

if TYPE_CHECKING:
    from unity.actor.environments.base import BaseEnvironment

# ---------------------------------------------------------------------------
# Registry dicts — populated by client subpackages at import time
# ---------------------------------------------------------------------------

_ORG_CONFIGS: dict[int, ActorConfig] = {}
_ORG_ENVIRONMENTS: dict[int, list[BaseEnvironment]] = {}
_ORG_FUNCTION_DIRS: dict[int, list[Path]] = {}
_ORG_VENV_DIRS: dict[int, list[Path]] = {}

_USER_CONFIGS: dict[str, ActorConfig] = {}
_USER_ENVIRONMENTS: dict[str, list[BaseEnvironment]] = {}
_USER_FUNCTION_DIRS: dict[str, list[Path]] = {}
_USER_VENV_DIRS: dict[str, list[Path]] = {}

_ASSISTANT_CONFIGS: dict[int, ActorConfig] = {}
_ASSISTANT_ENVIRONMENTS: dict[int, list[BaseEnvironment]] = {}
_ASSISTANT_FUNCTION_DIRS: dict[int, list[Path]] = {}
_ASSISTANT_VENV_DIRS: dict[int, list[Path]] = {}


# ---------------------------------------------------------------------------
# Public helpers for client subpackages to register themselves
# ---------------------------------------------------------------------------


def register_org(
    org_id: int,
    *,
    config: ActorConfig | None = None,
    environments: list[BaseEnvironment] | None = None,
    function_dir: Path | None = None,
    venv_dir: Path | None = None,
) -> None:
    if config is not None:
        _ORG_CONFIGS[org_id] = config
    if environments:
        _ORG_ENVIRONMENTS[org_id] = environments
    if function_dir is not None:
        _ORG_FUNCTION_DIRS.setdefault(org_id, []).append(function_dir)
    if venv_dir is not None:
        _ORG_VENV_DIRS.setdefault(org_id, []).append(venv_dir)


def register_user(
    user_id: str,
    *,
    config: ActorConfig | None = None,
    environments: list[BaseEnvironment] | None = None,
    function_dir: Path | None = None,
    venv_dir: Path | None = None,
) -> None:
    if config is not None:
        _USER_CONFIGS[user_id] = config
    if environments:
        _USER_ENVIRONMENTS[user_id] = environments
    if function_dir is not None:
        _USER_FUNCTION_DIRS.setdefault(user_id, []).append(function_dir)
    if venv_dir is not None:
        _USER_VENV_DIRS.setdefault(user_id, []).append(venv_dir)


def register_assistant(
    assistant_id: int,
    *,
    config: ActorConfig | None = None,
    environments: list[BaseEnvironment] | None = None,
    function_dir: Path | None = None,
    venv_dir: Path | None = None,
) -> None:
    if config is not None:
        _ASSISTANT_CONFIGS[assistant_id] = config
    if environments:
        _ASSISTANT_ENVIRONMENTS[assistant_id] = environments
    if function_dir is not None:
        _ASSISTANT_FUNCTION_DIRS.setdefault(assistant_id, []).append(function_dir)
    if venv_dir is not None:
        _ASSISTANT_VENV_DIRS.setdefault(assistant_id, []).append(venv_dir)


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


def _collect_dirs(
    registries: list[dict],
    keys: list,
) -> list[Path]:
    """Gather directory paths from multiple registry dicts in cascade order."""
    dirs: list[Path] = []
    for registry, key in zip(registries, keys):
        if key is not None and key in registry:
            dirs.extend(registry[key])
    return dirs


def resolve(
    org_id: int | None = None,
    user_id: str | None = None,
    assistant_id: int | None = None,
) -> tuple[ActorConfig, list[BaseEnvironment], list[Path], list[Path]]:
    """Resolve merged customizations for the given identity.

    Cascade order (least to most specific): org -> user -> assistant.

    Returns:
        A 4-tuple of:
        - Merged ``ActorConfig`` (scalars: last non-None wins; guidelines: concatenated)
        - Environments (additive, deduplicated by name)
        - Function directories (ordered org -> user -> assistant)
        - Venv directories (ordered org -> user -> assistant)
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

    seen: dict[str, BaseEnvironment] = {}
    for env in environments:
        seen[env.name] = env
    deduped_envs = list(seen.values())

    function_dirs = _collect_dirs(
        [_ORG_FUNCTION_DIRS, _USER_FUNCTION_DIRS, _ASSISTANT_FUNCTION_DIRS],
        [org_id, user_id, assistant_id],
    )
    venv_dirs = _collect_dirs(
        [_ORG_VENV_DIRS, _USER_VENV_DIRS, _ASSISTANT_VENV_DIRS],
        [org_id, user_id, assistant_id],
    )

    return merged_config, deduped_envs, function_dirs, venv_dirs


# ---------------------------------------------------------------------------
# Import client subpackages so they self-register.
# Add new clients here.
# ---------------------------------------------------------------------------

from unity.customization.clients import colliers as _colliers  # noqa: F401, E402
from unity.customization.clients import (
    midland_heart as _midland_heart,
)  # noqa: F401, E402
