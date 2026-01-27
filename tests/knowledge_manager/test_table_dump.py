from __future__ import annotations

import json
import pytest


@pytest.mark.asyncio
async def test_full_table_dump_seeded_when_fits(
    monkeypatch,
    knowledge_manager_scenario,
):
    km, _ = knowledge_manager_scenario

    # Configure KM to allow full dump
    km._full_table_dump = True
    km._per_table_dumps = False
    km._max_input_tokens = 64000

    # Two tables with tiny estimates
    tables_overview = {"A": {}, "B": {}}

    # Stub token estimation to small totals
    async def fake_estimate_tables_tokens_parallel(**kwargs):
        return {"A": 100, "B": 120}

    monkeypatch.setattr(
        "unity.knowledge_manager.knowledge_manager._tok.estimate_tables_tokens_parallel",
        fake_estimate_tables_tokens_parallel,
        raising=True,
    )

    # Stub payload builder to include both tables
    def fake_build_grouped_dump_payload(table_to_ctx, selected, **kwargs):
        payload = json.dumps({t: [{"x": 1}] for t in selected})
        per_tbl = {t: 50 for t in selected}
        return payload, per_tbl

    monkeypatch.setattr(
        "unity.knowledge_manager.knowledge_manager.build_grouped_dump_payload",
        fake_build_grouped_dump_payload,
        raising=True,
    )

    seeded = await km._maybe_build_show_all_seed("show all", tables_overview)
    assert seeded is not None
    # Assistant turn with show_all
    assistant = next(m for m in seeded if m["role"] == "assistant")
    assert any(
        tc.get("function", {}).get("name") == "show_all"
        for tc in assistant.get("tool_calls", [])
    )
    # Tool payload contains both tables
    tool_msg = next(
        m for m in seeded if m["role"] == "tool" and m.get("name") == "show_all"
    )
    data = json.loads(tool_msg["content"])
    assert set(data.keys()) == {"A", "B"}


@pytest.mark.asyncio
async def test_per_table_dump_selects_only_small(
    monkeypatch,
    knowledge_manager_scenario,
):
    km, _ = knowledge_manager_scenario
    km._full_table_dump = False
    km._per_table_dumps = True
    km._max_input_tokens = 8000

    tables_overview = {"Big": {}, "Small": {}}

    async def fake_estimate_tables_tokens_parallel(**kwargs):
        return {"Big": 100000, "Small": 200}

    monkeypatch.setattr(
        "unity.knowledge_manager.knowledge_manager._tok.estimate_tables_tokens_parallel",
        fake_estimate_tables_tokens_parallel,
        raising=True,
    )

    def fake_build_grouped_dump_payload(table_to_ctx, selected, **kwargs):
        payload = json.dumps({t: [] for t in selected})
        per_tbl = {t: 100 for t in selected}
        return payload, per_tbl

    monkeypatch.setattr(
        "unity.knowledge_manager.knowledge_manager.build_grouped_dump_payload",
        fake_build_grouped_dump_payload,
        raising=True,
    )

    seeded = await km._maybe_build_show_all_seed("overview", tables_overview)
    assert seeded is not None
    tool_msg = next(
        m for m in seeded if m["role"] == "tool" and m.get("name") == "show_all"
    )
    data = json.loads(tool_msg["content"])
    assert set(data.keys()) == {"Small"}


@pytest.mark.asyncio
async def test_trimming_applies_when_payload_exceeds_budget(
    monkeypatch,
    knowledge_manager_scenario,
):
    km, _ = knowledge_manager_scenario
    km._full_table_dump = True
    km._per_table_dumps = False
    km._max_input_tokens = 1000

    tables_overview = {"T1": {}, "T2": {}, "T3": {}}

    async def fake_estimate_tables_tokens_parallel(**kwargs):
        # Small total to pass full-dump selection
        return {"T1": 100, "T2": 100, "T3": 100}

    monkeypatch.setattr(
        "unity.knowledge_manager.knowledge_manager._tok.estimate_tables_tokens_parallel",
        fake_estimate_tables_tokens_parallel,
        raising=True,
    )

    # Return per-table payload sizes so that T3 is largest and gets trimmed first
    def fake_build_grouped_dump_payload(table_to_ctx, selected, **kwargs):
        per_tbl = {t: (500 if t == "T3" else 300) for t in selected}
        payload = json.dumps({t: "X" * per_tbl[t] for t in selected})
        return payload, per_tbl

    monkeypatch.setattr(
        "unity.knowledge_manager.knowledge_manager.build_grouped_dump_payload",
        fake_build_grouped_dump_payload,
        raising=True,
    )

    seeded = await km._maybe_build_show_all_seed("trim", tables_overview)
    assert seeded is not None
    tool_msg = next(
        m for m in seeded if m["role"] == "tool" and m.get("name") == "show_all"
    )
    data = json.loads(tool_msg["content"])
    # Expect fewer than all tables due to trimming
    assert 1 <= len(data.keys()) < 3


@pytest.mark.asyncio
async def test_no_dump_when_flags_off(knowledge_manager_scenario):
    km, _ = knowledge_manager_scenario
    km._full_table_dump = False
    km._per_table_dumps = False
    seeded = await km._maybe_build_show_all_seed("noop", {"A": {}, "B": {}})
    assert seeded is None
