"""Tests for GuidanceManager environment exclusion (guidance_id masking).

Mirrors tests/function_manager/test_fm_environment_exclusion.py for the
guidance side.  Verifies:
1. _build_id_exclusion produces correct filter clauses
2. _scoped_filter composes caller_filter / filter_scope / exclude_ids
3. _resolve_prompt_guidance returns (text, resolved_ids)
4. The wiring in ActorEnvironment.act() sets exclude_ids on the inner GM
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional


from unity.guidance_manager.guidance_manager import GuidanceManager
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# _build_id_exclusion (static helper)
# ────────────────────────────────────────────────────────────────────────────


def test_build_id_exclusion_none_when_empty():
    assert GuidanceManager._build_id_exclusion(None) is None
    assert GuidanceManager._build_id_exclusion(frozenset()) is None


def test_build_id_exclusion_single_id():
    result = GuidanceManager._build_id_exclusion(frozenset({7}))
    assert result == "guidance_id != 7"


def test_build_id_exclusion_multiple_ids_sorted():
    result = GuidanceManager._build_id_exclusion(frozenset({30, 10, 20}))
    assert result == "guidance_id != 10 and guidance_id != 20 and guidance_id != 30"


# ────────────────────────────────────────────────────────────────────────────
# _scoped_filter (composition of caller_filter / filter_scope / exclude_ids)
# ────────────────────────────────────────────────────────────────────────────


def _make_gm_stub(
    *,
    filter_scope: Optional[str] = None,
    exclude_ids: Optional[frozenset[int]] = None,
) -> SimpleNamespace:
    """Minimal stub with real GuidanceManager filter-composition methods."""
    ns = SimpleNamespace(
        _filter_scope=filter_scope,
        _exclude_ids=frozenset(exclude_ids) if exclude_ids else None,
    )
    ns._build_id_exclusion = GuidanceManager._build_id_exclusion
    ns._scoped_filter = lambda cf: GuidanceManager._scoped_filter(ns, cf)
    return ns


def test_scoped_filter_includes_exclusion():
    gm = _make_gm_stub(
        filter_scope="category == 'ops'",
        exclude_ids={5},
    )
    result = gm._scoped_filter("title == 'Deploy'")
    assert "title == 'Deploy'" in result
    assert "category == 'ops'" in result
    assert "guidance_id != 5" in result


def test_scoped_filter_exclusion_only():
    gm = _make_gm_stub(exclude_ids={99})
    result = gm._scoped_filter(None)
    assert result == "guidance_id != 99"


def test_scoped_filter_scope_only():
    gm = _make_gm_stub(filter_scope="category == 'ops'")
    result = gm._scoped_filter(None)
    assert result == "category == 'ops'"


def test_scoped_filter_all_none_returns_none():
    gm = _make_gm_stub()
    assert gm._scoped_filter(None) is None


# ────────────────────────────────────────────────────────────────────────────
# _resolve_prompt_guidance — empty / None fast path (no backend needed)
# ────────────────────────────────────────────────────────────────────────────


def test_resolve_prompt_guidance_none_input():
    from unity.actor.environments.actor import _resolve_prompt_guidance

    text, ids = _resolve_prompt_guidance(None)
    assert text is None
    assert ids == frozenset()


def test_resolve_prompt_guidance_empty_list():
    from unity.actor.environments.actor import _resolve_prompt_guidance

    text, ids = _resolve_prompt_guidance([])
    assert text is None
    assert ids == frozenset()


# ────────────────────────────────────────────────────────────────────────────
# _resolve_prompt_guidance — with real guidance entries (backend required)
# ────────────────────────────────────────────────────────────────────────────


def _seed(gm: GuidanceManager) -> dict[str, int]:
    """Create guidance entries and return {title: guidance_id}."""
    ids = {}
    for title, content in [
        ("Deploy Guide", "Step-by-step deployment procedure"),
        ("Review Checklist", "Code review checklist for PRs"),
    ]:
        out = gm.add_guidance(title=title, content=content)
        ids[title] = out["details"]["guidance_id"]
    return ids


@_handle_project
def test_resolve_prompt_guidance_by_title():
    from unity.actor.environments.actor import _resolve_prompt_guidance

    gm = GuidanceManager()
    ids = _seed(gm)

    text, resolved_ids = _resolve_prompt_guidance(["Deploy Guide"])
    assert text is not None
    assert "Deploy Guide" in text
    assert "Step-by-step deployment procedure" in text
    assert f"guidance_id: {ids['Deploy Guide']}" in text
    assert resolved_ids == frozenset({ids["Deploy Guide"]})


@_handle_project
def test_resolve_prompt_guidance_by_id():
    from unity.actor.environments.actor import _resolve_prompt_guidance

    gm = GuidanceManager()
    ids = _seed(gm)

    text, resolved_ids = _resolve_prompt_guidance([ids["Review Checklist"]])
    assert text is not None
    assert "Review Checklist" in text
    assert f"guidance_id: {ids['Review Checklist']}" in text
    assert resolved_ids == frozenset({ids["Review Checklist"]})


@_handle_project
def test_resolve_prompt_guidance_mixed():
    from unity.actor.environments.actor import _resolve_prompt_guidance

    gm = GuidanceManager()
    ids = _seed(gm)

    text, resolved_ids = _resolve_prompt_guidance(
        ["Deploy Guide", ids["Review Checklist"]],
    )
    assert text is not None
    assert "Deploy Guide" in text
    assert "Review Checklist" in text
    assert resolved_ids == frozenset({ids["Deploy Guide"], ids["Review Checklist"]})


@_handle_project
def test_resolve_prompt_guidance_renders_function_ids():
    """function_ids cross-references appear in the rendered guidance text."""
    from unity.actor.environments.actor import _resolve_prompt_guidance

    gm = GuidanceManager()
    gm.add_guidance(
        title="Linked Guide",
        content="Guide with linked functions",
        function_ids=[10, 20],
    )

    text, _ = _resolve_prompt_guidance(["Linked Guide"])
    assert text is not None
    assert "Related functions:" in text
    assert "10" in text
    assert "20" in text


@_handle_project
def test_resolve_prompt_guidance_unmatched_returns_empty():
    """Identifiers that don't match any guidance produce no text and no IDs."""
    from unity.actor.environments.actor import _resolve_prompt_guidance

    GuidanceManager()

    text, resolved_ids = _resolve_prompt_guidance(["Nonexistent Guide"])
    assert text is None
    assert resolved_ids == frozenset()


# ────────────────────────────────────────────────────────────────────────────
# Wiring: _build_scoped_gm receives exclude_ids from resolved prompt_guidance
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_build_scoped_gm_receives_exclude_ids():
    """Simulates the wiring in ActorEnvironment.act(): resolved guidance IDs
    are set on the inner GuidanceManager via exclude_ids, masking those
    entries from subsequent discovery queries."""
    from unity.actor.environments.actor import (
        _build_scoped_gm,
        _resolve_prompt_guidance,
    )

    gm = GuidanceManager()
    ids = _seed(gm)

    _, resolved_ids = _resolve_prompt_guidance(["Deploy Guide"])
    assert resolved_ids

    inner_gm = _build_scoped_gm(None)
    assert inner_gm.exclude_ids is None

    inner_gm.exclude_ids = resolved_ids
    assert inner_gm.exclude_ids == frozenset({ids["Deploy Guide"]})

    rows = inner_gm.filter()
    returned_ids = {r.guidance_id for r in rows}
    assert ids["Deploy Guide"] not in returned_ids
    assert ids["Review Checklist"] in returned_ids
