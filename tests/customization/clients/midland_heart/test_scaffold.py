"""Midland Heart client scaffold tests — verify registration data and structure."""

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


class TestMidlandHeartScaffold:
    """Structural tests following the Colliers pattern."""

    def test_mh_helper_functions_importable(self):
        from unity.customization.clients.midland_heart.functions.helpers import (
            discover_repairs_table,
            discover_telematics_tables,
            build_metric_result,
            extract_count,
            compute_percentage,
        )

        assert callable(discover_repairs_table)
        assert callable(discover_telematics_tables)
        assert callable(build_metric_result)
        assert callable(extract_count)
        assert callable(compute_percentage)

    def test_mh_metric_functions_importable(self):
        from unity.customization.clients.midland_heart.functions.metrics import (
            repairs_kpi_jobs_completed,
            repairs_kpi_no_access_rate,
            repairs_kpi_total_distance_travelled,
        )

        assert callable(repairs_kpi_jobs_completed)
        assert callable(repairs_kpi_no_access_rate)
        assert callable(repairs_kpi_total_distance_travelled)

    def test_mh_guidance_entries(self):
        from unity.customization.clients.midland_heart import _MH_GUIDANCE

        assert len(_MH_GUIDANCE) == 4
        titles = {g["title"] for g in _MH_GUIDANCE}
        assert "Repairs KPI Analysis Workflow" in titles
        assert "Repairs Data Schema" in titles
        assert "Telematics Data Schema" in titles
        assert "Data Discovery Guide" in titles
        for g in _MH_GUIDANCE:
            assert len(g["content"]) > 100

    def test_mh_config_is_thin(self):
        from unity.customization.clients.midland_heart import _MH_CONFIG

        assert _MH_CONFIG.guidelines is not None
        assert len(_MH_CONFIG.guidelines) < 300

    def test_mh_org_registration(self):
        from unity.customization.clients.midland_heart import (
            _MH_CONFIG,
            _MH_FUNCTION_DIR,
            _MH_GUIDANCE,
            _MH_ORG_ID,
        )

        assert _MH_ORG_ID == 3
        register_org(
            _MH_ORG_ID,
            config=_MH_CONFIG,
            function_dir=_MH_FUNCTION_DIR,
            guidance=_MH_GUIDANCE,
        )
        r = resolve(org_id=_MH_ORG_ID)
        assert r.config.guidelines is not None
        assert r.environments == []
        assert len(r.guidance) == 4
        assert len(r.function_dirs) == 1

    def test_mh_function_dir_exists(self):
        from unity.customization.clients.midland_heart import _MH_FUNCTION_DIR

        assert _MH_FUNCTION_DIR.is_dir()
        py_files = [
            f for f in _MH_FUNCTION_DIR.glob("*.py") if not f.name.startswith("_")
        ]
        assert len(py_files) >= 2, f"Expected >=2 function files, got {len(py_files)}"

    def test_seed_data_importable(self):
        from unity.customization.clients.midland_heart.seed_data import (
            seed_all,
            seed_repairs,
            seed_telematics,
            REPAIRS_CONTEXT,
            TELEMATICS_CONTEXT_PREFIX,
        )

        assert callable(seed_all)
        assert callable(seed_repairs)
        assert callable(seed_telematics)
        assert REPAIRS_CONTEXT == "Data/MidlandHeart/Repairs"
        assert TELEMATICS_CONTEXT_PREFIX == "Data/MidlandHeart/Telematics"

    def test_build_metric_result_returns_standard_dict(self):
        from unity.customization.clients.midland_heart.functions.helpers import (
            build_metric_result,
        )

        result = build_metric_result(
            metric_name="test_metric",
            group_by=None,
            time_period="day",
            start_date=None,
            end_date=None,
            results=[{"group": "total", "count": 42}],
            total=42.0,
        )
        assert result["metric_name"] == "test_metric"
        assert result["total"] == 42.0
        assert result["group_by"] is None
        assert result["time_period"] == "day"
        assert result["plots"] == []

    def test_compute_percentage_edge_cases(self):
        from unity.customization.clients.midland_heart.functions.helpers import (
            compute_percentage,
        )

        assert compute_percentage(50, 100) == 50.0
        assert compute_percentage(0, 100) == 0.0
        assert compute_percentage(50, 0) == 0.0
        assert compute_percentage(1, 3) == 33.33

    def test_extract_count_handles_various_inputs(self):
        from unity.customization.clients.midland_heart.functions.helpers import (
            extract_count,
        )

        assert extract_count(5) == 5
        assert extract_count(3.14) == 3
        assert extract_count(None) == 0
        assert extract_count({"count": 10}) == 10
        assert extract_count({"shared_value": 7}) == 7
        assert extract_count("not a number") == 0
