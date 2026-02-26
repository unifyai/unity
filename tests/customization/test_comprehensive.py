"""Comprehensive tests for the code-first client customization system.

Covers registration helpers, environment reconstruction, the Colliers
client scaffold, knowledge row-level dedup, ActorConfig model behavior,
and the sync_all_seed_data orchestrator.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unity.customization.configs.types.actor_config import ActorConfig
from unity.customization.environments.reconstruct import (
    parse_env_path,
    write_files_to_package,
    import_and_resolve,
)
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
    register_org,
    register_user,
    register_assistant,
    resolve,
)
from unity.customization.seed_sync import (
    _aggregate_hash,
    _record_hash,
    sync_all_seed_data,
)
from unity.customization.secrets_file import load_secrets

_ALL_DICTS = [
    _ORG_CONFIGS,
    _ORG_ENVIRONMENTS,
    _ORG_FUNCTION_DIRS,
    _ORG_VENV_DIRS,
    _ORG_CONTACTS,
    _ORG_GUIDANCE,
    _ORG_KNOWLEDGE,
    _ORG_BLACKLIST,
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
    saved = [dict(d) for d in _ALL_DICTS]
    for d in _ALL_DICTS:
        d.clear()
    yield
    for d, s in zip(_ALL_DICTS, saved):
        d.clear()
        d.update(s)


# ---------------------------------------------------------------------------
# 1. Registration helpers
# ---------------------------------------------------------------------------


class TestRegisterOrg:
    def test_registers_config(self):
        register_org(1, config=ActorConfig(model="m"))
        assert _ORG_CONFIGS[1].model == "m"

    def test_registers_contacts(self):
        register_org(1, contacts=[{"first_name": "A"}])
        assert len(_ORG_CONTACTS[1]) == 1

    def test_registers_guidance(self):
        register_org(1, guidance=[{"title": "T", "content": "C"}])
        assert _ORG_GUIDANCE[1][0]["title"] == "T"

    def test_registers_knowledge(self):
        register_org(
            1,
            knowledge={"Tbl": {"columns": {"x": "str"}, "seed_key": "x", "rows": []}},
        )
        assert "Tbl" in _ORG_KNOWLEDGE[1]

    def test_registers_blacklist(self):
        register_org(
            1,
            blacklist=[{"medium": "email", "contact_detail": "x", "reason": "y"}],
        )
        assert len(_ORG_BLACKLIST[1]) == 1

    def test_registers_function_dir(self):
        register_org(1, function_dir=Path("/fn"))
        assert _ORG_FUNCTION_DIRS[1] == [Path("/fn")]

    def test_registers_venv_dir(self):
        register_org(1, venv_dir=Path("/v"))
        assert _ORG_VENV_DIRS[1] == [Path("/v")]

    def test_multiple_calls_append(self):
        register_org(1, contacts=[{"first_name": "A"}])
        register_org(1, contacts=[{"first_name": "B"}])
        assert len(_ORG_CONTACTS[1]) == 2


class TestRegisterUser:
    def test_registers_config(self):
        register_user("u1", config=ActorConfig(timeout=30.0))
        assert _USER_CONFIGS["u1"].timeout == 30.0

    def test_registers_contacts(self):
        register_user("u1", contacts=[{"first_name": "Z"}])
        assert len(_USER_CONTACTS["u1"]) == 1


class TestRegisterAssistant:
    def test_registers_config(self):
        register_assistant(10, config=ActorConfig(can_compose=False))
        assert _ASSISTANT_CONFIGS[10].can_compose is False

    def test_registers_guidance(self):
        register_assistant(10, guidance=[{"title": "G", "content": "C"}])
        assert len(_ASSISTANT_GUIDANCE[10]) == 1


# ---------------------------------------------------------------------------
# 2. ActorConfig model
# ---------------------------------------------------------------------------


class TestActorConfig:
    def test_default_all_none(self):
        cfg = ActorConfig()
        assert cfg.can_compose is None
        assert cfg.model is None
        assert cfg.guidelines is None

    def test_to_post_json_excludes_none(self):
        cfg = ActorConfig(model="m")
        d = cfg.to_post_json()
        assert d == {"model": "m"}

    def test_to_post_json_empty_for_default(self):
        assert ActorConfig().to_post_json() == {}

    def test_all_fields_round_trip(self):
        cfg = ActorConfig(
            can_compose=True,
            can_store=False,
            timeout=60.0,
            model="m",
            prompt_caching=["system"],
            guidelines="G",
        )
        d = cfg.to_post_json()
        assert ActorConfig(**d) == cfg


# ---------------------------------------------------------------------------
# 3. Environment reconstruction helpers
# ---------------------------------------------------------------------------


class TestEnvironmentReconstruct:
    def test_parse_env_path_valid(self):
        mod, attr = parse_env_path("my_module:my_attr")
        assert mod == "my_module"
        assert attr == "my_attr"

    def test_parse_env_path_invalid_no_colon(self):
        with pytest.raises(ValueError, match="must be"):
            parse_env_path("no_colon_here")

    def test_parse_env_path_empty_parts(self):
        with pytest.raises(ValueError, match="non-empty"):
            parse_env_path(":attr")

    def test_write_files_and_import(self, tmp_path):
        pkg_dir = write_files_to_package(
            environment_id=999,
            files={"hello.py": "GREETING = 'world'"},
            root=tmp_path,
        )
        assert (pkg_dir / "hello.py").exists()

        result = import_and_resolve(
            pkg_dir=pkg_dir,
            module_name="hello",
            attr_name="GREETING",
        )
        assert result == "world"

    def test_write_files_only_rewrites_on_change(self, tmp_path):
        files = {"test.py": "X = 1"}
        pkg_dir = write_files_to_package(
            environment_id=1,
            files=files,
            root=tmp_path,
        )
        mtime1 = (pkg_dir / "test.py").stat().st_mtime_ns
        write_files_to_package(
            environment_id=1,
            files=files,
            root=tmp_path,
        )
        mtime2 = (pkg_dir / "test.py").stat().st_mtime_ns
        assert mtime1 == mtime2


# ---------------------------------------------------------------------------
# 4. Colliers client scaffold
# ---------------------------------------------------------------------------


class TestColliersScaffold:
    def test_colliers_environment_importable(self):
        from unity.customization.clients.colliers.colliers_env import (
            ColliersEnvironment,
        )

        assert ColliersEnvironment.NAMESPACE == "colliers"

    def test_colliers_environment_has_tools(self):
        from unity.customization.clients.colliers.colliers_env import (
            ColliersEnvironment,
        )

        env = ColliersEnvironment()
        tools = env.get_tools()
        assert "colliers.create_financial_data_excel" in tools
        assert "colliers.create_web_search_excel" in tools

    def test_colliers_environment_has_prompt_context(self):
        from unity.customization.clients.colliers.colliers_env import (
            ColliersEnvironment,
        )

        env = ColliersEnvironment()
        ctx = env.get_prompt_context()
        assert "FiscalYearData" in ctx
        assert "DealRow" in ctx

    def test_colliers_guidelines_nonempty(self):
        from unity.customization.clients.colliers.guidelines import (
            COLLIERS_GUIDELINES,
        )

        assert len(COLLIERS_GUIDELINES) > 100
        assert "Financial Data Extraction" in COLLIERS_GUIDELINES
        assert "Web Deal Research" in COLLIERS_GUIDELINES

    def test_colliers_schemas_importable(self):
        from unity.customization.clients.colliers.colliers_schemas import (
            FiscalYearData,
            DealRow,
        )

        assert FiscalYearData.model_fields["property_name"] is not None
        assert DealRow.model_fields["name"] is not None

    def test_colliers_not_registered_with_dummy_id(self):
        r = resolve(org_id=-1)
        assert r.config == ActorConfig()
        assert r.environments == []


# ---------------------------------------------------------------------------
# 5. Knowledge cascade: row-level dedup within same table
# ---------------------------------------------------------------------------


class TestKnowledgeCascade:
    def test_same_table_rows_merged_by_seed_key(self):
        _ORG_KNOWLEDGE[1] = {
            "Companies": {
                "columns": {"name": "str", "hq": "str"},
                "seed_key": "name",
                "rows": [
                    {"name": "Acme", "hq": "London"},
                    {"name": "Beta", "hq": "NYC"},
                ],
            },
        }
        _USER_KNOWLEDGE["u1"] = {
            "Companies": {
                "columns": {"name": "str", "hq": "str"},
                "seed_key": "name",
                "rows": [
                    {"name": "Acme", "hq": "Paris"},
                    {"name": "Gamma", "hq": "Berlin"},
                ],
            },
        }
        r = resolve(org_id=1, user_id="u1")
        rows = r.knowledge["Companies"]["rows"]
        names = {row["name"] for row in rows}
        assert names == {"Acme", "Beta", "Gamma"}
        acme = next(row for row in rows if row["name"] == "Acme")
        assert acme["hq"] == "Paris"

    def test_different_tables_from_different_levels(self):
        _ORG_KNOWLEDGE[1] = {
            "T1": {
                "columns": {"a": "str"},
                "seed_key": "a",
                "rows": [{"a": "x"}],
            },
        }
        _USER_KNOWLEDGE["u1"] = {
            "T2": {
                "columns": {"b": "str"},
                "seed_key": "b",
                "rows": [{"b": "y"}],
            },
        }
        r = resolve(org_id=1, user_id="u1")
        assert "T1" in r.knowledge
        assert "T2" in r.knowledge

    def test_empty_knowledge_returns_empty(self):
        r = resolve(org_id=999)
        assert r.knowledge == {}


# ---------------------------------------------------------------------------
# 6. Blacklist cascade dedup
# ---------------------------------------------------------------------------


class TestBlacklistCascade:
    def test_dedup_by_medium_and_contact_detail(self):
        _ORG_BLACKLIST[1] = [
            {"medium": "email", "contact_detail": "x@x.com", "reason": "Org reason"},
        ]
        _USER_BLACKLIST["u1"] = [
            {"medium": "email", "contact_detail": "x@x.com", "reason": "User reason"},
            {"medium": "sms_message", "contact_detail": "+123", "reason": "Spam"},
        ]
        r = resolve(org_id=1, user_id="u1")
        assert len(r.blacklist) == 2
        email_entry = next(e for e in r.blacklist if e["contact_detail"] == "x@x.com")
        assert email_entry["reason"] == "User reason"


# ---------------------------------------------------------------------------
# 7. sync_all_seed_data with empty data
# ---------------------------------------------------------------------------


class TestSyncAllSeedData:
    def test_noop_with_empty_resolved(self):
        empty = ResolvedCustomization(
            config=ActorConfig(),
            environments=[],
            function_dirs=[],
            venv_dirs=[],
            contacts=[],
            guidance=[],
            knowledge={},
            blacklist=[],
            secrets=[],
        )
        result = sync_all_seed_data(empty)
        assert result is False


# ---------------------------------------------------------------------------
# 8. Secrets file: assistant-level cascade
# ---------------------------------------------------------------------------


class TestSecretsAssistantLevel:
    def test_assistant_overrides_user_and_org(self, tmp_path):
        f = tmp_path / ".secrets.json"
        f.write_text(
            json.dumps(
                {
                    "org": {"1": {"K": {"value": "org", "description": "d"}}},
                    "user": {"u1": {"K": {"value": "user", "description": "d"}}},
                    "assistant": {"10": {"K": {"value": "asst", "description": "d"}}},
                },
            ),
        )
        result = load_secrets(org_id=1, user_id="u1", assistant_id=10, path=f)
        assert len(result) == 1
        assert result[0]["value"] == "asst"


# ---------------------------------------------------------------------------
# 9. Custom function collection from directories
# ---------------------------------------------------------------------------


class TestCustomFunctionCollection:
    def test_collect_from_empty_dir(self, tmp_path):
        from unity.function_manager.custom_functions import (
            collect_custom_functions,
        )

        fn_dir = tmp_path / "functions"
        fn_dir.mkdir()
        result = collect_custom_functions(directory=fn_dir)
        assert result == {}

    def test_collect_ignores_underscore_prefixed_files(self, tmp_path):
        from unity.function_manager.custom_functions import (
            collect_custom_functions,
        )

        fn_dir = tmp_path / "functions"
        fn_dir.mkdir()
        (fn_dir / "_private.py").write_text(
            "from unity.function_manager.custom import custom_function\n"
            "@custom_function()\n"
            "async def hidden() -> int:\n"
            "    return 1\n",
        )
        result = collect_custom_functions(directory=fn_dir)
        assert "hidden" not in result


# ---------------------------------------------------------------------------
# 10. Hash behavior edge cases
# ---------------------------------------------------------------------------


class TestHashEdgeCases:
    def test_hash_with_nested_dicts(self):
        r = {"name": "x", "meta": {"a": 1, "b": [2, 3]}}
        h = _record_hash(r, set())
        assert len(h) == 16

    def test_aggregate_hash_single_record(self):
        h = _aggregate_hash(
            [{"k": "only"}],
            lambda r: r["k"],
            set(),
        )
        assert len(h) == 16

    def test_hash_changes_when_value_changes(self):
        h1 = _record_hash({"name": "A", "v": 1}, set())
        h2 = _record_hash({"name": "A", "v": 2}, set())
        assert h1 != h2
