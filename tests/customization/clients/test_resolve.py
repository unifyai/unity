"""Tests for the code-first client customization resolve() function."""

from __future__ import annotations

from pathlib import Path

import pytest

from unity.customization.configs.types.actor_config import ActorConfig
from unity.customization.clients import (
    ResolvedCustomization,
    _ORG_CONFIGS,
    _ORG_ENVIRONMENTS,
    _ORG_FUNCTION_DIRS,
    _ORG_VENV_DIRS,
    _ORG_CONTACTS,
    _ORG_GUIDANCE,
    _ORG_KNOWLEDGE,
    _ORG_BLACKLIST,
    _TEAM_CONFIGS,
    _TEAM_ENVIRONMENTS,
    _TEAM_FUNCTION_DIRS,
    _TEAM_VENV_DIRS,
    _TEAM_CONTACTS,
    _TEAM_GUIDANCE,
    _TEAM_KNOWLEDGE,
    _TEAM_BLACKLIST,
    _USER_CONFIGS,
    _USER_ENVIRONMENTS,
    _USER_FUNCTION_DIRS,
    _USER_VENV_DIRS,
    _USER_CONTACTS,
    _USER_GUIDANCE,
    _USER_KNOWLEDGE,
    _USER_BLACKLIST,
    _ASSISTANT_CONFIGS,
    _ASSISTANT_ENVIRONMENTS,
    _ASSISTANT_FUNCTION_DIRS,
    _ASSISTANT_VENV_DIRS,
    _ASSISTANT_CONTACTS,
    _ASSISTANT_GUIDANCE,
    _ASSISTANT_KNOWLEDGE,
    _ASSISTANT_BLACKLIST,
    _merge_configs,
    resolve,
)
from unity.actor.code_act_actor import _resolve_param, _UNSET

# ---------------------------------------------------------------------------
# Fixtures: save/restore the global registry dicts around each test
# ---------------------------------------------------------------------------

_ALL_DICTS = [
    _ORG_CONFIGS,
    _ORG_ENVIRONMENTS,
    _ORG_FUNCTION_DIRS,
    _ORG_VENV_DIRS,
    _ORG_CONTACTS,
    _ORG_GUIDANCE,
    _ORG_KNOWLEDGE,
    _ORG_BLACKLIST,
    _TEAM_CONFIGS,
    _TEAM_ENVIRONMENTS,
    _TEAM_FUNCTION_DIRS,
    _TEAM_VENV_DIRS,
    _TEAM_CONTACTS,
    _TEAM_GUIDANCE,
    _TEAM_KNOWLEDGE,
    _TEAM_BLACKLIST,
    _USER_CONFIGS,
    _USER_ENVIRONMENTS,
    _USER_FUNCTION_DIRS,
    _USER_VENV_DIRS,
    _USER_CONTACTS,
    _USER_GUIDANCE,
    _USER_KNOWLEDGE,
    _USER_BLACKLIST,
    _ASSISTANT_CONFIGS,
    _ASSISTANT_ENVIRONMENTS,
    _ASSISTANT_FUNCTION_DIRS,
    _ASSISTANT_VENV_DIRS,
    _ASSISTANT_CONTACTS,
    _ASSISTANT_GUIDANCE,
    _ASSISTANT_KNOWLEDGE,
    _ASSISTANT_BLACKLIST,
]


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot and restore all registry dicts so tests don't leak."""
    saved = [dict(d) for d in _ALL_DICTS]
    for d in _ALL_DICTS:
        d.clear()
    yield
    for d, s in zip(_ALL_DICTS, saved):
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
        user = ActorConfig()
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
        r = resolve(org_id=999, user_id="nobody", assistant_id=999)
        assert isinstance(r, ResolvedCustomization)
        assert r.config == ActorConfig()
        assert r.environments == []
        assert r.function_dirs == []
        assert r.contacts == []
        assert r.guidance == []
        assert r.knowledge == {}
        assert r.blacklist == []
        assert r.secrets == []

    def test_none_ids_returns_defaults(self):
        r = resolve()
        assert r.config == ActorConfig()
        assert r.environments == []

    def test_org_config_resolved(self):
        _ORG_CONFIGS[1] = ActorConfig(model="org-model")
        r = resolve(org_id=1)
        assert r.config.model == "org-model"

    def test_user_overrides_org(self):
        _ORG_CONFIGS[1] = ActorConfig(model="org-model", timeout=60.0)
        _USER_CONFIGS["u1"] = ActorConfig(model="user-model")
        r = resolve(org_id=1, user_id="u1")
        assert r.config.model == "user-model"
        assert r.config.timeout == 60.0

    def test_assistant_overrides_user_and_org(self):
        _ORG_CONFIGS[1] = ActorConfig(model="org-model")
        _USER_CONFIGS["u1"] = ActorConfig(model="user-model")
        _ASSISTANT_CONFIGS[10] = ActorConfig(model="assistant-model")
        r = resolve(org_id=1, user_id="u1", assistant_id=10)
        assert r.config.model == "assistant-model"

    def test_guidelines_concatenated_across_levels(self):
        _ORG_CONFIGS[1] = ActorConfig(guidelines="Org.")
        _USER_CONFIGS["u1"] = ActorConfig(guidelines="User.")
        _ASSISTANT_CONFIGS[10] = ActorConfig(guidelines="Assistant.")
        r = resolve(org_id=1, user_id="u1", assistant_id=10)
        assert r.config.guidelines == "Org.\nUser.\nAssistant."

    def test_function_dirs_cascade(self):
        _ORG_FUNCTION_DIRS[1] = [Path("/org/functions")]
        _USER_FUNCTION_DIRS["u1"] = [Path("/user/functions")]
        _ASSISTANT_FUNCTION_DIRS[10] = [Path("/asst/functions")]
        r = resolve(org_id=1, user_id="u1", assistant_id=10)
        assert r.function_dirs == [
            Path("/org/functions"),
            Path("/user/functions"),
            Path("/asst/functions"),
        ]

    def test_venv_dirs_cascade(self):
        _ORG_VENV_DIRS[1] = [Path("/org/venvs")]
        r = resolve(org_id=1)
        assert r.venv_dirs == [Path("/org/venvs")]

    def test_contacts_cascade_dedup_by_name(self):
        _ORG_CONTACTS[1] = [
            {"first_name": "Alice", "surname": "Smith", "email_address": "a@x.com"},
        ]
        _USER_CONTACTS["u1"] = [
            {
                "first_name": "Alice",
                "surname": "Smith",
                "email_address": "alice@new.com",
            },
            {"first_name": "Bob", "surname": "Jones"},
        ]
        r = resolve(org_id=1, user_id="u1")
        names = {c["first_name"] for c in r.contacts}
        assert "Alice" in names
        assert "Bob" in names
        alice = next(c for c in r.contacts if c["first_name"] == "Alice")
        assert alice["email_address"] == "alice@new.com"

    def test_guidance_cascade_dedup_by_title(self):
        _ORG_GUIDANCE[1] = [{"title": "Workflow", "content": "Org version"}]
        _USER_GUIDANCE["u1"] = [{"title": "Workflow", "content": "User version"}]
        r = resolve(org_id=1, user_id="u1")
        assert len(r.guidance) == 1
        assert r.guidance[0]["content"] == "User version"

    def test_knowledge_cascade_merges_tables(self):
        _ORG_KNOWLEDGE[1] = {
            "Companies": {
                "columns": {"name": "str"},
                "seed_key": "name",
                "rows": [{"name": "OrgCo"}],
            },
        }
        _USER_KNOWLEDGE["u1"] = {
            "Products": {
                "columns": {"product": "str"},
                "seed_key": "product",
                "rows": [{"product": "Widget"}],
            },
        }
        r = resolve(org_id=1, user_id="u1")
        assert "Companies" in r.knowledge
        assert "Products" in r.knowledge

    def test_blacklist_cascade(self):
        _ORG_BLACKLIST[1] = [
            {"medium": "email", "contact_detail": "s@x.com", "reason": "Spam"},
        ]
        r = resolve(org_id=1)
        assert len(r.blacklist) == 1


# ---------------------------------------------------------------------------
# resolve() with team_ids
# ---------------------------------------------------------------------------


class TestResolveTeam:
    def test_team_config_resolved(self):
        _TEAM_CONFIGS[100] = ActorConfig(model="team-model")
        r = resolve(team_ids=[100])
        assert r.config.model == "team-model"

    def test_team_overrides_org(self):
        _ORG_CONFIGS[1] = ActorConfig(model="org-model", timeout=60.0)
        _TEAM_CONFIGS[100] = ActorConfig(model="team-model")
        r = resolve(org_id=1, team_ids=[100])
        assert r.config.model == "team-model"
        assert r.config.timeout == 60.0

    def test_user_overrides_team(self):
        _TEAM_CONFIGS[100] = ActorConfig(model="team-model")
        _USER_CONFIGS["u1"] = ActorConfig(model="user-model")
        r = resolve(team_ids=[100], user_id="u1")
        assert r.config.model == "user-model"

    def test_full_cascade_org_team_user_assistant(self):
        _ORG_CONFIGS[1] = ActorConfig(guidelines="Org.", timeout=120.0)
        _TEAM_CONFIGS[100] = ActorConfig(guidelines="Team.", model="team-model")
        _USER_CONFIGS["u1"] = ActorConfig(guidelines="User.")
        _ASSISTANT_CONFIGS[10] = ActorConfig(
            guidelines="Assistant.",
            model="asst-model",
        )
        r = resolve(org_id=1, team_ids=[100], user_id="u1", assistant_id=10)
        assert r.config.guidelines == "Org.\nTeam.\nUser.\nAssistant."
        assert r.config.model == "asst-model"
        assert r.config.timeout == 120.0

    def test_multi_team_merge_ordered_by_id(self):
        _TEAM_CONFIGS[200] = ActorConfig(model="team-200")
        _TEAM_CONFIGS[100] = ActorConfig(model="team-100", timeout=30.0)
        r = resolve(team_ids=[200, 100])
        assert r.config.model == "team-200"
        assert r.config.timeout == 30.0

    def test_multi_team_guidelines_concatenated_in_order(self):
        _TEAM_CONFIGS[100] = ActorConfig(guidelines="Team100.")
        _TEAM_CONFIGS[200] = ActorConfig(guidelines="Team200.")
        r = resolve(team_ids=[200, 100])
        assert r.config.guidelines == "Team100.\nTeam200."

    def test_team_function_dirs_cascade(self):
        _ORG_FUNCTION_DIRS[1] = [Path("/org/fn")]
        _TEAM_FUNCTION_DIRS[100] = [Path("/team100/fn")]
        _TEAM_FUNCTION_DIRS[200] = [Path("/team200/fn")]
        _USER_FUNCTION_DIRS["u1"] = [Path("/user/fn")]
        r = resolve(org_id=1, team_ids=[200, 100], user_id="u1")
        assert r.function_dirs == [
            Path("/org/fn"),
            Path("/team100/fn"),
            Path("/team200/fn"),
            Path("/user/fn"),
        ]

    def test_team_venv_dirs_cascade(self):
        _TEAM_VENV_DIRS[100] = [Path("/team/venvs")]
        r = resolve(team_ids=[100])
        assert r.venv_dirs == [Path("/team/venvs")]

    def test_team_contacts_cascade_dedup(self):
        _ORG_CONTACTS[1] = [
            {"first_name": "Alice", "surname": "Smith", "email_address": "org@x.com"},
        ]
        _TEAM_CONTACTS[100] = [
            {"first_name": "Alice", "surname": "Smith", "email_address": "team@x.com"},
            {"first_name": "TeamBob", "surname": "Jones"},
        ]
        _USER_CONTACTS["u1"] = [
            {"first_name": "UserCarol", "surname": "Lee"},
        ]
        r = resolve(org_id=1, team_ids=[100], user_id="u1")
        names = {c["first_name"] for c in r.contacts}
        assert names == {"Alice", "TeamBob", "UserCarol"}
        alice = next(c for c in r.contacts if c["first_name"] == "Alice")
        assert alice["email_address"] == "team@x.com"

    def test_team_guidance_cascade_dedup(self):
        _ORG_GUIDANCE[1] = [{"title": "Workflow", "content": "Org version"}]
        _TEAM_GUIDANCE[100] = [{"title": "Workflow", "content": "Team version"}]
        r = resolve(org_id=1, team_ids=[100])
        assert len(r.guidance) == 1
        assert r.guidance[0]["content"] == "Team version"

    def test_team_knowledge_cascade(self):
        _ORG_KNOWLEDGE[1] = {
            "Companies": {
                "columns": {"name": "str"},
                "seed_key": "name",
                "rows": [{"name": "OrgCo"}],
            },
        }
        _TEAM_KNOWLEDGE[100] = {
            "Companies": {
                "columns": {"name": "str"},
                "seed_key": "name",
                "rows": [{"name": "TeamCo"}],
            },
            "TeamTable": {
                "columns": {"x": "str"},
                "seed_key": "x",
                "rows": [{"x": "val"}],
            },
        }
        r = resolve(org_id=1, team_ids=[100])
        assert "Companies" in r.knowledge
        assert "TeamTable" in r.knowledge
        co_names = {row["name"] for row in r.knowledge["Companies"]["rows"]}
        assert co_names == {"OrgCo", "TeamCo"}

    def test_multi_team_knowledge_merge(self):
        _TEAM_KNOWLEDGE[100] = {
            "Tbl": {
                "columns": {"k": "str"},
                "seed_key": "k",
                "rows": [{"k": "from100", "v": "a"}],
            },
        }
        _TEAM_KNOWLEDGE[200] = {
            "Tbl": {
                "columns": {"k": "str"},
                "seed_key": "k",
                "rows": [{"k": "from100", "v": "overridden"}, {"k": "from200"}],
            },
        }
        r = resolve(team_ids=[200, 100])
        rows = r.knowledge["Tbl"]["rows"]
        by_k = {row["k"]: row for row in rows}
        assert by_k["from100"]["v"] == "overridden"
        assert "from200" in by_k

    def test_team_blacklist_cascade(self):
        _TEAM_BLACKLIST[100] = [
            {"medium": "email", "contact_detail": "t@x.com", "reason": "Team block"},
        ]
        r = resolve(team_ids=[100])
        assert len(r.blacklist) == 1
        assert r.blacklist[0]["reason"] == "Team block"

    def test_empty_team_ids_same_as_no_teams(self):
        _ORG_CONFIGS[1] = ActorConfig(model="org-model")
        r1 = resolve(org_id=1, team_ids=[])
        r2 = resolve(org_id=1, team_ids=None)
        r3 = resolve(org_id=1)
        assert r1.config == r2.config == r3.config

    def test_team_ids_with_no_registrations_is_noop(self):
        _ORG_CONFIGS[1] = ActorConfig(model="org-model")
        r = resolve(org_id=1, team_ids=[999, 888])
        assert r.config.model == "org-model"


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
