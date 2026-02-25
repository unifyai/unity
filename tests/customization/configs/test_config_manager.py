"""Unit tests for ConfigManager using SimulatedConfigManager.

These tests validate core functionality without requiring Unify backend
connectivity.
"""

from __future__ import annotations

import pytest

from unity.customization.configs.simulated import SimulatedConfigManager
from unity.customization.configs.types.actor_config import ActorConfig
from unity.actor.code_act_actor import _resolve_param, _UNSET

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def manager():
    return SimulatedConfigManager()


# ── Tests: ActorConfig model ─────────────────────────────────────────────────


class TestActorConfig:
    def test_defaults_are_all_none(self):
        cfg = ActorConfig()
        assert cfg.can_compose is None
        assert cfg.can_store is None
        assert cfg.timeout is None
        assert cfg.model is None
        assert cfg.prompt_caching is None

    def test_partial_config(self):
        cfg = ActorConfig(can_compose=False)
        assert cfg.can_compose is False
        assert cfg.can_store is None
        assert cfg.timeout is None

    def test_full_config(self):
        cfg = ActorConfig(
            can_compose=True,
            can_store=False,
            timeout=120.0,
            model="gpt-4o@openai",
            prompt_caching=["system", "messages"],
        )
        assert cfg.can_compose is True
        assert cfg.can_store is False
        assert cfg.timeout == 120.0
        assert cfg.model == "gpt-4o@openai"
        assert cfg.prompt_caching == ["system", "messages"]

    def test_to_post_json_excludes_none(self):
        cfg = ActorConfig(can_compose=False, timeout=60.0)
        payload = cfg.to_post_json()
        assert payload == {"can_compose": False, "timeout": 60.0}
        assert "can_store" not in payload
        assert "model" not in payload
        assert "prompt_caching" not in payload

    def test_to_post_json_empty_config(self):
        cfg = ActorConfig()
        assert cfg.to_post_json() == {}

    def test_timeout_must_be_positive(self):
        with pytest.raises(Exception):
            ActorConfig(timeout=0)
        with pytest.raises(Exception):
            ActorConfig(timeout=-10)


# ── Tests: SimulatedConfigManager CRUD ────────────────────────────────────────


class TestSimulatedCRUD:
    def test_load_returns_empty_when_nothing_saved(self, manager):
        cfg = manager.load_config()
        assert isinstance(cfg, ActorConfig)
        assert cfg.can_compose is None
        assert cfg.can_store is None

    def test_save_and_load_roundtrip(self, manager):
        original = ActorConfig(can_compose=False, timeout=120.0)
        manager.save_config(original)
        loaded = manager.load_config()
        assert loaded.can_compose is False
        assert loaded.timeout == 120.0
        assert loaded.can_store is None

    def test_save_upserts(self, manager):
        manager.save_config(ActorConfig(can_compose=True))
        manager.save_config(ActorConfig(can_compose=False, model="gpt-4o@openai"))
        loaded = manager.load_config()
        assert loaded.can_compose is False
        assert loaded.model == "gpt-4o@openai"
        # The first config's values are fully replaced, not merged.
        assert loaded.can_store is None

    def test_clear(self, manager):
        manager.save_config(ActorConfig(can_compose=False))
        manager.clear()
        cfg = manager.load_config()
        assert cfg.can_compose is None

    def test_save_full_config(self, manager):
        full = ActorConfig(
            can_compose=True,
            can_store=True,
            timeout=300.0,
            model="claude-4.5-opus@anthropic",
            prompt_caching=["system", "tools"],
        )
        manager.save_config(full)
        loaded = manager.load_config()
        assert loaded == full


# ── Tests: _resolve_param (three-tier precedence) ────────────────────────────


class TestResolveParam:
    def test_explicit_wins_over_db_and_default(self):
        assert _resolve_param(False, True, True) is False

    def test_db_wins_over_default_when_unset(self):
        assert _resolve_param(_UNSET, False, True) is False

    def test_default_used_when_both_unset_and_db_none(self):
        assert _resolve_param(_UNSET, None, True) is True

    def test_explicit_none_wins(self):
        """Explicit None should be returned (e.g. for model=None)."""
        assert _resolve_param(None, "some-model", "default-model") is None

    def test_db_value_zero_is_not_none(self):
        assert _resolve_param(_UNSET, 0, 42) == 0

    def test_db_false_is_not_none(self):
        assert _resolve_param(_UNSET, False, True) is False

    def test_explicit_false_wins(self):
        assert _resolve_param(False, True, True) is False

    def test_string_values(self):
        assert _resolve_param(_UNSET, "db-model", "default") == "db-model"
        assert _resolve_param("explicit", "db-model", "default") == "explicit"
        assert _resolve_param(_UNSET, None, "default") == "default"

    def test_list_values(self):
        db_val = ["system", "messages"]
        default = ("system", "tools", "messages")
        assert _resolve_param(_UNSET, db_val, default) == db_val
        assert _resolve_param(["explicit"], db_val, default) == ["explicit"]
        assert _resolve_param(_UNSET, None, default) == default
