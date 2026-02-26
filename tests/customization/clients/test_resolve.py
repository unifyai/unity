"""Tests for the code-first client customization resolve() function."""

from __future__ import annotations

import pytest

from unity.customization.configs.types.actor_config import ActorConfig
from unity.customization.clients import (
    _ORG_CONFIGS,
    _ORG_ENVIRONMENTS,
    _ORG_FUNCTION_DIRS,
    _ORG_VENV_DIRS,
    _USER_CONFIGS,
    _USER_ENVIRONMENTS,
    _USER_FUNCTION_DIRS,
    _USER_VENV_DIRS,
    _ASSISTANT_CONFIGS,
    _ASSISTANT_ENVIRONMENTS,
    _ASSISTANT_FUNCTION_DIRS,
    _ASSISTANT_VENV_DIRS,
    _merge_configs,
    resolve,
)
from unity.actor.code_act_actor import _resolve_param, _UNSET

# ---------------------------------------------------------------------------
# Fixtures: save/restore the global registry dicts around each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot and restore all registry dicts so tests don't leak."""
    snapshots = [
        _ORG_CONFIGS,
        _ORG_ENVIRONMENTS,
        _ORG_FUNCTION_DIRS,
        _ORG_VENV_DIRS,
        _USER_CONFIGS,
        _USER_ENVIRONMENTS,
        _USER_FUNCTION_DIRS,
        _USER_VENV_DIRS,
        _ASSISTANT_CONFIGS,
        _ASSISTANT_ENVIRONMENTS,
        _ASSISTANT_FUNCTION_DIRS,
        _ASSISTANT_VENV_DIRS,
    ]
    saved = [dict(d) for d in snapshots]
    for d in snapshots:
        d.clear()
    yield
    for d, s in zip(snapshots, saved):
        d.clear()
        d.update(s)


# ---------------------------------------------------------------------------
# _merge_configs tests
# ---------------------------------------------------------------------------


class TestMergeConfigs:
    def test_empty_list_returns_default_config(self):
        result = _merge_configs([])
        assert result == ActorConfig()

    def test_single_config_passed_through(self):
        cfg = ActorConfig(can_compose=False, model="gpt-4o@openai")
        result = _merge_configs([cfg])
        assert result.can_compose is False
        assert result.model == "gpt-4o@openai"
        assert result.timeout is None

    def test_scalar_override_last_wins(self):
        org = ActorConfig(model="org-model", timeout=60.0)
        user = ActorConfig(model="user-model")
        result = _merge_configs([org, user])
        assert result.model == "user-model"
        assert result.timeout == 60.0

    def test_guidelines_concatenated(self):
        org = ActorConfig(guidelines="Be professional.")
        user = ActorConfig(guidelines="Speak French.")
        assistant = ActorConfig(guidelines="Use formal tone.")
        result = _merge_configs([org, user, assistant])
        assert result.guidelines == "Be professional.\nSpeak French.\nUse formal tone."

    def test_guidelines_skips_none(self):
        org = ActorConfig(guidelines="Org rules.")
        user = ActorConfig()  # no guidelines
        assistant = ActorConfig(guidelines="Assistant rules.")
        result = _merge_configs([org, user, assistant])
        assert result.guidelines == "Org rules.\nAssistant rules."

    def test_all_fields_merge_correctly(self):
        org = ActorConfig(
            can_compose=True,
            can_store=True,
            timeout=120.0,
            model="org-model",
            prompt_caching=["system"],
            guidelines="Org guidelines.",
        )
        user = ActorConfig(
            model="user-model",
            prompt_caching=["system", "tools"],
        )
        assistant = ActorConfig(
            can_compose=False,
            guidelines="Assistant guidelines.",
        )
        result = _merge_configs([org, user, assistant])
        assert result.can_compose is False
        assert result.can_store is True
        assert result.timeout == 120.0
        assert result.model == "user-model"
        assert result.prompt_caching == ["system", "tools"]
        assert result.guidelines == "Org guidelines.\nAssistant guidelines."


# ---------------------------------------------------------------------------
# resolve() tests
# ---------------------------------------------------------------------------


class TestResolve:
    def test_no_match_returns_defaults(self):
        config, envs, fn_dirs, venv_dirs = resolve(
            org_id=999,
            user_id="nobody",
            assistant_id=999,
        )
        assert config == ActorConfig()
        assert envs == []
        assert fn_dirs == []
        assert venv_dirs == []

    def test_none_ids_returns_defaults(self):
        config, envs, fn_dirs, venv_dirs = resolve()
        assert config == ActorConfig()
        assert envs == []
        assert fn_dirs == []
        assert venv_dirs == []

    def test_org_config_resolved(self):
        _ORG_CONFIGS[1] = ActorConfig(model="org-model")
        config, *_ = resolve(org_id=1)
        assert config.model == "org-model"

    def test_user_overrides_org(self):
        _ORG_CONFIGS[1] = ActorConfig(model="org-model", timeout=60.0)
        _USER_CONFIGS["u1"] = ActorConfig(model="user-model")
        config, *_ = resolve(org_id=1, user_id="u1")
        assert config.model == "user-model"
        assert config.timeout == 60.0

    def test_assistant_overrides_user_and_org(self):
        _ORG_CONFIGS[1] = ActorConfig(model="org-model")
        _USER_CONFIGS["u1"] = ActorConfig(model="user-model")
        _ASSISTANT_CONFIGS[10] = ActorConfig(model="assistant-model")
        config, *_ = resolve(org_id=1, user_id="u1", assistant_id=10)
        assert config.model == "assistant-model"

    def test_guidelines_concatenated_across_levels(self):
        _ORG_CONFIGS[1] = ActorConfig(guidelines="Org.")
        _USER_CONFIGS["u1"] = ActorConfig(guidelines="User.")
        _ASSISTANT_CONFIGS[10] = ActorConfig(guidelines="Assistant.")
        config, *_ = resolve(org_id=1, user_id="u1", assistant_id=10)
        assert config.guidelines == "Org.\nUser.\nAssistant."

    def test_function_dirs_cascade(self):
        from pathlib import Path

        _ORG_FUNCTION_DIRS[1] = [Path("/org/functions")]
        _USER_FUNCTION_DIRS["u1"] = [Path("/user/functions")]
        _ASSISTANT_FUNCTION_DIRS[10] = [Path("/asst/functions")]
        _, _, fn_dirs, _ = resolve(org_id=1, user_id="u1", assistant_id=10)
        assert fn_dirs == [
            Path("/org/functions"),
            Path("/user/functions"),
            Path("/asst/functions"),
        ]

    def test_venv_dirs_cascade(self):
        from pathlib import Path

        _ORG_VENV_DIRS[1] = [Path("/org/venvs")]
        _, _, _, venv_dirs = resolve(org_id=1)
        assert venv_dirs == [Path("/org/venvs")]


# ---------------------------------------------------------------------------
# _resolve_param tests (ported from deleted test_config_manager.py)
# ---------------------------------------------------------------------------


class TestResolveParam:
    def test_explicit_wins_over_code_and_default(self):
        assert _resolve_param(42, 99, 0) == 42

    def test_explicit_false_wins(self):
        assert _resolve_param(False, True, True) is False

    def test_explicit_zero_wins(self):
        assert _resolve_param(0, 42, 100) == 0

    def test_code_value_used_when_unset(self):
        assert _resolve_param(_UNSET, False, True) is False

    def test_code_value_zero_used_when_unset(self):
        assert _resolve_param(_UNSET, 0, 42) == 0

    def test_code_string_used_when_unset(self):
        assert _resolve_param(_UNSET, "code-model", "default") == "code-model"

    def test_default_used_when_both_unset_and_none(self):
        assert _resolve_param(_UNSET, None, True) is True

    def test_default_string_used_when_both_unset_and_none(self):
        assert _resolve_param(_UNSET, None, "default") == "default"

    def test_code_list_used_when_unset(self):
        code_val = ["system", "tools"]
        assert _resolve_param(_UNSET, code_val, []) == code_val

    def test_default_list_used_when_both_unset_and_none(self):
        default = ["system", "tools", "messages"]
        assert _resolve_param(_UNSET, None, default) == default
