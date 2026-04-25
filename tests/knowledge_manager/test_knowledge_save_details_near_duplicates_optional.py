from __future__ import annotations

from unity.knowledge_manager import knowledge_manager as knowledge_module
from unity.knowledge_manager.knowledge_manager import KnowledgeManager


def test_add_rows_details_include_near_duplicates_when_provided(monkeypatch):
    def fake_add_rows(_manager, *, table, rows):
        return [101, 102]

    monkeypatch.setattr(knowledge_module, "_op_add_rows", fake_add_rows)

    result = KnowledgeManager._add_rows(
        object(),
        table="Facts",
        rows=[{"summary": "Patch reports are due Mondays."}],
        _near_duplicates=[
            {
                "id": 7,
                "summary": "Weekly patch reports are sent every Monday.",
                "similarity_score": 0.91,
            },
        ],
    )

    assert result["details"]["length"] == 2
    assert result["details"]["near_duplicates"] == [
        {
            "id": 7,
            "summary": "Weekly patch reports are sent every Monday.",
            "similarity_score": 0.91,
        },
    ]


def test_add_rows_details_omit_near_duplicates_when_absent(monkeypatch):
    def fake_add_rows(_manager, *, table, rows):
        return [101]

    monkeypatch.setattr(knowledge_module, "_op_add_rows", fake_add_rows)

    result = KnowledgeManager._add_rows(
        object(),
        table="Facts",
        rows=[{"summary": "Patch reports are due Mondays."}],
    )

    assert result["details"] == {"length": 1}
    assert "near_duplicates" not in result["details"]
