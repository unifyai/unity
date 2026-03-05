"""Colliers client scaffold tests — verify registration data and structure."""

from __future__ import annotations

import pytest

from unity.customization.clients import (
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
    register_org,
    resolve,
)

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
    saved = [dict(d) for d in _ALL_DICTS]
    for d in _ALL_DICTS:
        d.clear()
    yield
    for d, s in zip(_ALL_DICTS, saved):
        d.clear()
        d.update(s)


class TestColliersScaffold:
    def test_colliers_functions_importable(self):
        from unity.customization.clients.colliers.functions.create_financial_data_excel import (
            create_financial_data_excel,
        )
        from unity.customization.clients.colliers.functions.create_web_search_excel import (
            create_web_search_excel,
        )

        assert callable(create_financial_data_excel)
        assert callable(create_web_search_excel)

    def test_colliers_guidance_entries(self):
        from unity.customization.clients.colliers import _COLLIERS_GUIDANCE

        assert len(_COLLIERS_GUIDANCE) == 4
        titles = {g["title"] for g in _COLLIERS_GUIDANCE}
        assert "Financial Data Extraction from Excel and PDF Documents" in titles
        assert "CoStar Web Research for UK Care Home Deals" in titles
        assert "Healthcare Valuation Financial Data Schema" in titles
        assert "Deal Tracker Data Schema" in titles
        for g in _COLLIERS_GUIDANCE:
            assert len(g["content"]) > 100

    def test_colliers_config_is_thin(self):
        from unity.customization.clients.colliers import _COLLIERS_CONFIG

        assert _COLLIERS_CONFIG.guidelines is not None
        assert len(_COLLIERS_CONFIG.guidelines) < 300

    def test_colliers_org_registration(self):
        from unity.customization.clients.colliers import (
            _COLLIERS_CONFIG,
            _COLLIERS_FUNCTION_DIR,
            _COLLIERS_GUIDANCE,
            _COLLIERS_ORG_ID,
            _COLLIERS_SECRETS,
        )

        assert _COLLIERS_ORG_ID == 2
        register_org(
            _COLLIERS_ORG_ID,
            config=_COLLIERS_CONFIG,
            function_dir=_COLLIERS_FUNCTION_DIR,
            guidance=_COLLIERS_GUIDANCE,
            secrets=_COLLIERS_SECRETS,
        )
        r = resolve(org_id=_COLLIERS_ORG_ID)
        assert r.config.guidelines is not None
        assert r.environments == []
        assert len(r.guidance) == 4
        assert len(r.function_dirs) == 1
