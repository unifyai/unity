from __future__ import annotations

from unity.guidance_manager import guidance_manager as guidance_module
from unity.guidance_manager.guidance_manager import GuidanceManager


class _FakeLog:
    entries = {"guidance_id": 42}


def _guidance_manager_without_backend() -> GuidanceManager:
    gm = GuidanceManager.__new__(GuidanceManager)
    gm._ctx = "Guidance"
    gm.include_in_multi_assistant_table = True
    return gm


def test_add_guidance_details_include_near_duplicates_when_provided(monkeypatch):
    monkeypatch.setattr(guidance_module, "unity_log", lambda **_kwargs: _FakeLog())

    result = _guidance_manager_without_backend().add_guidance(
        title="Renewals",
        content="Before renewal calls, verify contract status.",
        _near_duplicates=[
            {
                "id": 9,
                "summary": "Check contract status before renewal outreach.",
                "similarity_score": 0.88,
            },
        ],
    )

    assert result["details"]["guidance_id"] == 42
    assert result["details"]["near_duplicates"] == [
        {
            "id": 9,
            "summary": "Check contract status before renewal outreach.",
            "similarity_score": 0.88,
        },
    ]


def test_add_guidance_details_omit_near_duplicates_when_absent(monkeypatch):
    monkeypatch.setattr(guidance_module, "unity_log", lambda **_kwargs: _FakeLog())

    result = _guidance_manager_without_backend().add_guidance(
        title="Renewals",
        content="Before renewal calls, verify contract status.",
    )

    assert result["details"] == {"guidance_id": 42}
    assert "near_duplicates" not in result["details"]
