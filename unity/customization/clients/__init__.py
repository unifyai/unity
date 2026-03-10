"""
Code-first client customization registry.

Client configs, environments, custom functions, and seed data are defined
in Python code under per-client subpackages (e.g. ``colliers/``).  Each
subpackage registers its org-level, team-level, user-level, and/or
assistant-level customizations into the module-level dicts below.

During manager initialization, ``resolve()`` is called with the current
org_id / team_ids / user_id / assistant_id to produce a
``ResolvedCustomization`` containing merged configs, environments,
function dirs, and seed data.  Cross-cutting seed data (contacts,
guidance, knowledge, secrets, blacklist) is synced separately by
``_init_managers()``, while actor-specific fields (config, environments)
are forwarded to the ``CodeActActor`` constructor.

Cascade order (least to most specific): org -> team(s) -> user -> assistant.
When a user belongs to multiple teams, team customizations are merged in
ascending team_id order before being fed into the cascade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from unity.customization.configs.types.actor_config import ActorConfig

if TYPE_CHECKING:
    from unity.actor.environments.base import BaseEnvironment


# ---------------------------------------------------------------------------
# Resolved result
# ---------------------------------------------------------------------------


@dataclass
class ResolvedCustomization:
    config: ActorConfig
    environments: list[BaseEnvironment]
    function_dirs: list[Path]
    venv_dirs: list[Path]
    contacts: list[dict[str, Any]]
    guidance: list[dict[str, Any]]
    knowledge: dict[str, dict[str, Any]]
    blacklist: list[dict[str, Any]]
    secrets: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Registry dicts — populated by client subpackages at import time
# ---------------------------------------------------------------------------

_ORG_CONFIGS: dict[int, ActorConfig] = {}
_ORG_ENVIRONMENTS: dict[int, list[BaseEnvironment]] = {}
_ORG_FUNCTION_DIRS: dict[int, list[Path]] = {}
_ORG_VENV_DIRS: dict[int, list[Path]] = {}
_ORG_CONTACTS: dict[int, list[dict]] = {}
_ORG_GUIDANCE: dict[int, list[dict]] = {}
_ORG_KNOWLEDGE: dict[int, dict[str, dict]] = {}
_ORG_BLACKLIST: dict[int, list[dict]] = {}
_ORG_SECRETS: dict[int, list[dict]] = {}

_TEAM_CONFIGS: dict[int, ActorConfig] = {}
_TEAM_ENVIRONMENTS: dict[int, list[BaseEnvironment]] = {}
_TEAM_FUNCTION_DIRS: dict[int, list[Path]] = {}
_TEAM_VENV_DIRS: dict[int, list[Path]] = {}
_TEAM_CONTACTS: dict[int, list[dict]] = {}
_TEAM_GUIDANCE: dict[int, list[dict]] = {}
_TEAM_KNOWLEDGE: dict[int, dict[str, dict]] = {}
_TEAM_BLACKLIST: dict[int, list[dict]] = {}
_TEAM_SECRETS: dict[int, list[dict]] = {}

_USER_CONFIGS: dict[str, ActorConfig] = {}
_USER_ENVIRONMENTS: dict[str, list[BaseEnvironment]] = {}
_USER_FUNCTION_DIRS: dict[str, list[Path]] = {}
_USER_VENV_DIRS: dict[str, list[Path]] = {}
_USER_CONTACTS: dict[str, list[dict]] = {}
_USER_GUIDANCE: dict[str, list[dict]] = {}
_USER_KNOWLEDGE: dict[str, dict[str, dict]] = {}
_USER_BLACKLIST: dict[str, list[dict]] = {}
_USER_SECRETS: dict[str, list[dict]] = {}

_ASSISTANT_CONFIGS: dict[int, ActorConfig] = {}
_ASSISTANT_ENVIRONMENTS: dict[int, list[BaseEnvironment]] = {}
_ASSISTANT_FUNCTION_DIRS: dict[int, list[Path]] = {}
_ASSISTANT_VENV_DIRS: dict[int, list[Path]] = {}
_ASSISTANT_CONTACTS: dict[int, list[dict]] = {}
_ASSISTANT_GUIDANCE: dict[int, list[dict]] = {}
_ASSISTANT_KNOWLEDGE: dict[int, dict[str, dict]] = {}
_ASSISTANT_BLACKLIST: dict[int, list[dict]] = {}
_ASSISTANT_SECRETS: dict[int, list[dict]] = {}


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
    contacts: list[dict] | None = None,
    guidance: list[dict] | None = None,
    knowledge: dict[str, dict] | None = None,
    blacklist: list[dict] | None = None,
    secrets: list[dict] | None = None,
) -> None:
    if config is not None:
        _ORG_CONFIGS[org_id] = config
    if environments:
        _ORG_ENVIRONMENTS[org_id] = environments
    if function_dir is not None:
        _ORG_FUNCTION_DIRS.setdefault(org_id, []).append(function_dir)
    if venv_dir is not None:
        _ORG_VENV_DIRS.setdefault(org_id, []).append(venv_dir)
    if contacts:
        _ORG_CONTACTS.setdefault(org_id, []).extend(contacts)
    if guidance:
        _ORG_GUIDANCE.setdefault(org_id, []).extend(guidance)
    if knowledge:
        _ORG_KNOWLEDGE.setdefault(org_id, {}).update(knowledge)
    if blacklist:
        _ORG_BLACKLIST.setdefault(org_id, []).extend(blacklist)
    if secrets:
        _ORG_SECRETS.setdefault(org_id, []).extend(secrets)


def register_team(
    team_id: int,
    *,
    config: ActorConfig | None = None,
    environments: list[BaseEnvironment] | None = None,
    function_dir: Path | None = None,
    venv_dir: Path | None = None,
    contacts: list[dict] | None = None,
    guidance: list[dict] | None = None,
    knowledge: dict[str, dict] | None = None,
    blacklist: list[dict] | None = None,
    secrets: list[dict] | None = None,
) -> None:
    if config is not None:
        _TEAM_CONFIGS[team_id] = config
    if environments:
        _TEAM_ENVIRONMENTS[team_id] = environments
    if function_dir is not None:
        _TEAM_FUNCTION_DIRS.setdefault(team_id, []).append(function_dir)
    if venv_dir is not None:
        _TEAM_VENV_DIRS.setdefault(team_id, []).append(venv_dir)
    if contacts:
        _TEAM_CONTACTS.setdefault(team_id, []).extend(contacts)
    if guidance:
        _TEAM_GUIDANCE.setdefault(team_id, []).extend(guidance)
    if knowledge:
        _TEAM_KNOWLEDGE.setdefault(team_id, {}).update(knowledge)
    if blacklist:
        _TEAM_BLACKLIST.setdefault(team_id, []).extend(blacklist)
    if secrets:
        _TEAM_SECRETS.setdefault(team_id, []).extend(secrets)


def register_user(
    user_id: str,
    *,
    config: ActorConfig | None = None,
    environments: list[BaseEnvironment] | None = None,
    function_dir: Path | None = None,
    venv_dir: Path | None = None,
    contacts: list[dict] | None = None,
    guidance: list[dict] | None = None,
    knowledge: dict[str, dict] | None = None,
    blacklist: list[dict] | None = None,
    secrets: list[dict] | None = None,
) -> None:
    if config is not None:
        _USER_CONFIGS[user_id] = config
    if environments:
        _USER_ENVIRONMENTS[user_id] = environments
    if function_dir is not None:
        _USER_FUNCTION_DIRS.setdefault(user_id, []).append(function_dir)
    if venv_dir is not None:
        _USER_VENV_DIRS.setdefault(user_id, []).append(venv_dir)
    if contacts:
        _USER_CONTACTS.setdefault(user_id, []).extend(contacts)
    if guidance:
        _USER_GUIDANCE.setdefault(user_id, []).extend(guidance)
    if knowledge:
        _USER_KNOWLEDGE.setdefault(user_id, {}).update(knowledge)
    if blacklist:
        _USER_BLACKLIST.setdefault(user_id, []).extend(blacklist)
    if secrets:
        _USER_SECRETS.setdefault(user_id, []).extend(secrets)


def register_assistant(
    assistant_id: int,
    *,
    config: ActorConfig | None = None,
    environments: list[BaseEnvironment] | None = None,
    function_dir: Path | None = None,
    venv_dir: Path | None = None,
    contacts: list[dict] | None = None,
    guidance: list[dict] | None = None,
    knowledge: dict[str, dict] | None = None,
    blacklist: list[dict] | None = None,
    secrets: list[dict] | None = None,
) -> None:
    if config is not None:
        _ASSISTANT_CONFIGS[assistant_id] = config
    if environments:
        _ASSISTANT_ENVIRONMENTS[assistant_id] = environments
    if function_dir is not None:
        _ASSISTANT_FUNCTION_DIRS.setdefault(assistant_id, []).append(function_dir)
    if venv_dir is not None:
        _ASSISTANT_VENV_DIRS.setdefault(assistant_id, []).append(venv_dir)
    if contacts:
        _ASSISTANT_CONTACTS.setdefault(assistant_id, []).extend(contacts)
    if guidance:
        _ASSISTANT_GUIDANCE.setdefault(assistant_id, []).extend(guidance)
    if knowledge:
        _ASSISTANT_KNOWLEDGE.setdefault(assistant_id, {}).update(knowledge)
    if blacklist:
        _ASSISTANT_BLACKLIST.setdefault(assistant_id, []).extend(blacklist)
    if secrets:
        _ASSISTANT_SECRETS.setdefault(assistant_id, []).extend(secrets)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _merge_configs(configs: list[ActorConfig]) -> ActorConfig:
    """Merge a list of configs in precedence order (first = least specific).

    For scalar fields, the last non-None value wins (more specific overrides
    less specific).  For ``guidelines``, all non-None values are concatenated
    with newlines (additive).  For ``url_mappings``, dicts are merged
    additively (more specific origins override less specific ones).
    """
    if not configs:
        return ActorConfig()

    merged: dict = {}
    guidelines_parts: list[str] = []
    url_mappings_merged: dict[str, str] = {}

    for cfg in configs:
        for field_name in cfg.model_fields:
            value = getattr(cfg, field_name)
            if value is None:
                continue
            if field_name == "guidelines":
                guidelines_parts.append(value)
            elif field_name == "url_mappings":
                url_mappings_merged.update(value)
            else:
                merged[field_name] = value

    if guidelines_parts:
        merged["guidelines"] = "\n".join(guidelines_parts)
    if url_mappings_merged:
        merged["url_mappings"] = url_mappings_merged

    return ActorConfig(**merged)


def _collect_dirs(
    registries: list[dict],
    keys: list,
) -> list[Path]:
    dirs: list[Path] = []
    for registry, key in zip(registries, keys):
        if key is not None and key in registry:
            dirs.extend(registry[key])
    return dirs


def _collect_dirs_multi(
    registries: list[tuple[dict, list[int] | int | str | None]],
) -> list[Path]:
    """Collect directories from registries, supporting multi-key lookups for teams."""
    dirs: list[Path] = []
    for registry, key_or_keys in registries:
        if key_or_keys is None:
            continue
        keys = key_or_keys if isinstance(key_or_keys, list) else [key_or_keys]
        for k in keys:
            if k in registry:
                dirs.extend(registry[k])
    return dirs


def _merge_list_seed(
    registries: list[dict],
    keys: list,
    natural_key_fn: Any = None,
) -> list[dict]:
    """Merge list-based seed data across cascade levels.

    Later (more specific) records override earlier ones by natural key
    if ``natural_key_fn`` is provided.  Otherwise just concatenate.
    """
    merged: list[dict] = []
    for registry, key in zip(registries, keys):
        if key is not None and key in registry:
            merged.extend(registry[key])
    if natural_key_fn is not None and merged:
        seen: dict[str, dict] = {}
        for rec in merged:
            try:
                seen[natural_key_fn(rec)] = rec
            except Exception:
                seen[id(rec)] = rec
        return list(seen.values())
    return merged


def _merge_list_seed_multi(
    registries: list[tuple[dict, list[int] | int | str | None]],
    natural_key_fn: Any = None,
) -> list[dict]:
    """Like _merge_list_seed but supports multi-key lookups for teams."""
    merged: list[dict] = []
    for registry, key_or_keys in registries:
        if key_or_keys is None:
            continue
        keys = key_or_keys if isinstance(key_or_keys, list) else [key_or_keys]
        for k in keys:
            if k in registry:
                merged.extend(registry[k])
    if natural_key_fn is not None and merged:
        seen: dict[str, dict] = {}
        for rec in merged:
            try:
                seen[natural_key_fn(rec)] = rec
            except Exception:
                seen[id(rec)] = rec
        return list(seen.values())
    return merged


def _merge_knowledge_seed(
    registries: list[dict],
    keys: list,
) -> dict[str, dict]:
    """Merge knowledge table specs across cascade levels.

    Later levels can add new tables or override rows in existing tables.
    Row merging within a table uses ``seed_key`` for dedup.
    """
    merged: dict[str, dict] = {}
    for registry, key in zip(registries, keys):
        if key is not None and key in registry:
            for table_name, spec in registry[key].items():
                if table_name not in merged:
                    merged[table_name] = dict(spec)
                else:
                    existing = merged[table_name]
                    seed_key = spec.get("seed_key") or existing.get("seed_key")
                    if spec.get("columns"):
                        existing.setdefault("columns", {}).update(spec["columns"])
                    if spec.get("description"):
                        existing["description"] = spec["description"]
                    if seed_key:
                        existing["seed_key"] = seed_key
                    if spec.get("rows"):
                        all_rows = existing.get("rows", []) + spec["rows"]
                        if seed_key:
                            seen: dict[str, dict] = {}
                            for row in all_rows:
                                seen[str(row.get(seed_key, id(row)))] = row
                            existing["rows"] = list(seen.values())
                        else:
                            existing["rows"] = all_rows
    return merged


def _merge_knowledge_seed_multi(
    registries: list[tuple[dict, list[int] | int | str | None]],
) -> dict[str, dict]:
    """Like _merge_knowledge_seed but supports multi-key lookups for teams."""
    merged: dict[str, dict] = {}
    for registry, key_or_keys in registries:
        if key_or_keys is None:
            continue
        keys = key_or_keys if isinstance(key_or_keys, list) else [key_or_keys]
        for k in keys:
            if k not in registry:
                continue
            for table_name, spec in registry[k].items():
                if table_name not in merged:
                    merged[table_name] = dict(spec)
                else:
                    existing = merged[table_name]
                    seed_key = spec.get("seed_key") or existing.get("seed_key")
                    if spec.get("columns"):
                        existing.setdefault("columns", {}).update(spec["columns"])
                    if spec.get("description"):
                        existing["description"] = spec["description"]
                    if seed_key:
                        existing["seed_key"] = seed_key
                    if spec.get("rows"):
                        all_rows = existing.get("rows", []) + spec["rows"]
                        if seed_key:
                            seen: dict[str, dict] = {}
                            for row in all_rows:
                                seen[str(row.get(seed_key, id(row)))] = row
                            existing["rows"] = list(seen.values())
                        else:
                            existing["rows"] = all_rows
    return merged


def resolve(
    org_id: int | None = None,
    team_ids: list[int] | None = None,
    user_id: str | None = None,
    assistant_id: int | None = None,
) -> ResolvedCustomization:
    """Resolve merged customizations for the given identity.

    Cascade order (least to most specific):
    org -> team(s) -> user -> assistant.

    When a user belongs to multiple teams, team customizations are merged
    in ascending team_id order before being fed into the cascade.
    """
    sorted_team_ids = sorted(team_ids) if team_ids else []

    configs: list[ActorConfig] = []
    environments: list[BaseEnvironment] = []

    if org_id is not None:
        if org_id in _ORG_CONFIGS:
            configs.append(_ORG_CONFIGS[org_id])
        if org_id in _ORG_ENVIRONMENTS:
            environments.extend(_ORG_ENVIRONMENTS[org_id])

    for tid in sorted_team_ids:
        if tid in _TEAM_CONFIGS:
            configs.append(_TEAM_CONFIGS[tid])
        if tid in _TEAM_ENVIRONMENTS:
            environments.extend(_TEAM_ENVIRONMENTS[tid])

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
        seen[env.namespace] = env
    deduped_envs = list(seen.values())

    cascade = [
        (_ORG_FUNCTION_DIRS, org_id),
        (_TEAM_FUNCTION_DIRS, sorted_team_ids or None),
        (_USER_FUNCTION_DIRS, user_id),
        (_ASSISTANT_FUNCTION_DIRS, assistant_id),
    ]
    function_dirs = _collect_dirs_multi(cascade)

    venv_cascade = [
        (_ORG_VENV_DIRS, org_id),
        (_TEAM_VENV_DIRS, sorted_team_ids or None),
        (_USER_VENV_DIRS, user_id),
        (_ASSISTANT_VENV_DIRS, assistant_id),
    ]
    venv_dirs = _collect_dirs_multi(venv_cascade)

    seed_cascade = [
        (_ORG_CONTACTS, org_id),
        (_TEAM_CONTACTS, sorted_team_ids or None),
        (_USER_CONTACTS, user_id),
        (_ASSISTANT_CONTACTS, assistant_id),
    ]
    contacts = _merge_list_seed_multi(
        seed_cascade,
        natural_key_fn=lambda r: (
            f"{r.get('first_name', '')}|{r.get('surname', '')}"
        ).lower(),
    )
    guidance = _merge_list_seed_multi(
        [
            (_ORG_GUIDANCE, org_id),
            (_TEAM_GUIDANCE, sorted_team_ids or None),
            (_USER_GUIDANCE, user_id),
            (_ASSISTANT_GUIDANCE, assistant_id),
        ],
        natural_key_fn=lambda r: str(r.get("title", "")),
    )
    blacklist = _merge_list_seed_multi(
        [
            (_ORG_BLACKLIST, org_id),
            (_TEAM_BLACKLIST, sorted_team_ids or None),
            (_USER_BLACKLIST, user_id),
            (_ASSISTANT_BLACKLIST, assistant_id),
        ],
        natural_key_fn=lambda r: f"{r.get('medium', '')}|{r.get('contact_detail', '')}",
    )
    knowledge = _merge_knowledge_seed_multi(
        [
            (_ORG_KNOWLEDGE, org_id),
            (_TEAM_KNOWLEDGE, sorted_team_ids or None),
            (_USER_KNOWLEDGE, user_id),
            (_ASSISTANT_KNOWLEDGE, assistant_id),
        ],
    )

    code_secrets = _merge_list_seed_multi(
        [
            (_ORG_SECRETS, org_id),
            (_TEAM_SECRETS, sorted_team_ids or None),
            (_USER_SECRETS, user_id),
            (_ASSISTANT_SECRETS, assistant_id),
        ],
        natural_key_fn=lambda r: str(r.get("name", "")),
    )

    from unity.customization.secrets_file import load_secrets

    file_secrets = load_secrets(
        org_id=org_id,
        team_ids=sorted_team_ids or None,
        user_id=user_id,
        assistant_id=assistant_id,
    )

    secrets_by_name: dict[str, dict[str, Any]] = {}
    for s in code_secrets:
        secrets_by_name[s["name"]] = s
    for s in file_secrets:
        secrets_by_name[s["name"]] = {**secrets_by_name.get(s["name"], {}), **s}
    secrets = list(secrets_by_name.values())

    return ResolvedCustomization(
        config=merged_config,
        environments=deduped_envs,
        function_dirs=function_dirs,
        venv_dirs=venv_dirs,
        contacts=contacts,
        guidance=guidance,
        knowledge=knowledge,
        blacklist=blacklist,
        secrets=secrets,
    )


# ---------------------------------------------------------------------------
# Import client subpackages so they self-register.
# Add new clients here.
# ---------------------------------------------------------------------------

from unity.customization.clients import colliers as _colliers  # noqa: F401, E402
from unity.customization.clients import (
    midland_heart as _midland_heart,
)  # noqa: F401, E402
from unity.customization.clients import vantage as _vantage  # noqa: F401, E402
