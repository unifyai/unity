"""
Trust is earned from the ledger and lost on any change.

Ramp to trust for each effect class with the exact counts from the policy
table; a policy raise blocks it; source, dependency, venv and guidance
changes invalidate; a repair invalidates and fixture replay re-trusts a pure
function immediately (bar its static review).

No LLM is involved anywhere in this file.
"""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unify.function_manager.function_manager import FunctionManager
from unify.function_manager.types.verification import (
    SideEffectClass,
    StaticReviewRecord,
    VerdictKind,
    VerificationRow,
)
from unify.guidance_manager.guidance_manager import GuidanceManager
from unify.manager_registry import ManagerRegistry


@pytest.fixture(autouse=True)
def _verification_enabled(monkeypatch):
    """This module exercises the trust machinery itself; the master switch
    defaults off, so hold it on for every test here."""
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS.function.verification, "enabled", True)


_PURE = "def add(a: int, b: int) -> int:\n    return a + b\n"
_READ = (
    "async def lookup(q: str) -> str:\n    return await primitives.contacts.ask(q)\n"
)
_UPSERT = (
    "async def save(rows: list) -> None:\n"
    "    await primitives.data.update_rows('T', rows)\n"
)
_SEND = (
    "async def notify(to: str, body: str) -> None:\n"
    "    await primitives.comms.send_email(to=to, subject='hi', body=body)\n"
)


def _fid(fm: FunctionManager, name: str) -> int:
    return int(fm._get_function_data_by_name(name=name)["function_id"])


def _current_hash(fm: FunctionManager, name: str) -> str:
    return fm.function_trust_hash(fm._get_function_data_by_name(name=name))


def _pass_static(fm: FunctionManager, name: str) -> None:
    fid = _fid(fm, name)
    fm._persist_verification_fields(
        function_id=fid,
        fields={
            "static_review": StaticReviewRecord(
                verdict="PASS",
                reason="ok",
                function_hash=_current_hash(fm, name),
            ).model_dump(mode="json"),
        },
    )


def _record(
    fm: FunctionManager,
    name: str,
    kind: VerdictKind,
    verdict: str,
    sig: str,
    *,
    fault=None,
) -> bool:
    fid = _fid(fm, name)
    fm.record_verification(
        VerificationRow(
            function_id=fid,
            function_hash=_current_hash(fm, name),
            kind=kind,
            verdict=verdict,
            fault=fault,
            args_signature=sig,
        ),
    )
    return bool(fm._get_function_data_by_name(name=name)["verify"])


def _ramp(fm: FunctionManager, name: str, *, passes: int, inputs: int) -> None:
    """Record ``passes`` args+post PASS pairs across ``inputs`` distinct signatures."""
    for i in range(passes):
        sig = f"sig{i % inputs}"
        _record(fm, name, VerdictKind.args, "PASS", sig)
        _record(fm, name, VerdictKind.post, "PASS", sig)


@pytest.mark.parametrize(
    "source, name, klass, passes, inputs",
    [
        (_READ, "lookup", SideEffectClass.read_only, 3, 2),
        (_UPSERT, "save", SideEffectClass.idempotent_effectful, 3, 2),
        (_SEND, "notify", SideEffectClass.unsafe_effectful, 5, 3),
    ],
)
@_handle_project
def test_ramp_to_trust_flips_at_exact_counts(source, name, klass, passes, inputs):
    fm = FunctionManager()
    fm.add_functions(implementations=source)
    assert fm._get_function_data_by_name(name=name)["side_effect_class"] == klass.value
    _pass_static(fm, name)

    # One pass short — across enough inputs — is still untrusted.
    _ramp(fm, name, passes=passes - 1, inputs=inputs)
    row = fm._get_function_data_by_name(name=name)
    assert row["verify"] is True
    assert row["verified_hash"] == fm.function_trust_hash(row)
    assert row["ledger"]["passes"] == {"args": passes - 1, "post": passes - 1}

    # The last pair flips it.
    sig = f"sig{(passes - 1) % inputs}"
    _record(fm, name, VerdictKind.args, "PASS", sig)
    assert _record(fm, name, VerdictKind.post, "PASS", sig) is False
    row = fm._get_function_data_by_name(name=name)
    assert row["ledger"]["passes"] == {"args": passes, "post": passes}
    assert len(row["ledger"]["distinct_args_signatures"]) == inputs


@_handle_project
def test_ramp_needs_distinct_inputs_and_static_pass():
    fm = FunctionManager()
    fm.add_functions(implementations=_READ)
    _pass_static(fm, "lookup")
    _ramp(fm, "lookup", passes=3, inputs=1)  # enough passes, one input
    assert fm._get_function_data_by_name(name="lookup")["verify"] is True
    _record(fm, "lookup", VerdictKind.args, "PASS", "other")
    assert _record(fm, "lookup", VerdictKind.post, "PASS", "other") is False

    fm.add_functions(implementations=_UPSERT)
    _ramp(fm, "save", passes=3, inputs=2)  # no static review yet
    assert fm._get_function_data_by_name(name="save")["verify"] is True
    _pass_static(fm, "save")
    fm.refresh_trust(_fid(fm, "save"))
    assert fm._get_function_data_by_name(name="save")["verify"] is False


@_handle_project
def test_policy_raise_blocks_trust_and_fail_keeps_it_blocked():
    fm = FunctionManager()
    fm.add_functions(implementations=_READ)
    fm._persist_verification_fields(
        function_id=_fid(fm, "lookup"),
        fields={"verification_policy": {"required_passes": 4}},
    )
    _pass_static(fm, "lookup")
    _ramp(fm, "lookup", passes=3, inputs=2)
    assert fm._get_function_data_by_name(name="lookup")["verify"] is True
    _record(fm, "lookup", VerdictKind.args, "PASS", "sig0")
    assert _record(fm, "lookup", VerdictKind.post, "PASS", "sig0") is False

    # Any FAIL for this hash withdraws trust until the content changes.
    assert (
        _record(fm, "lookup", VerdictKind.spot_check, "FAIL", "sig9", fault="leaf")
        is True
    )
    row = fm._get_function_data_by_name(name="lookup")
    assert row["ledger"]["fails"] == 1


@_handle_project
def test_safe_noop_trusts_on_tier0_alone():
    fm = FunctionManager()
    fm.add_functions(implementations=_PURE)
    _pass_static(fm, "add")
    assert _record(fm, "add", VerdictKind.tier0, "PASS", "sig0") is False


@_handle_project
def test_source_dependency_and_venv_changes_invalidate():
    fm = FunctionManager()
    fm.add_functions(
        implementations=[
            "def inner(x: int) -> int:\n    return x\n",
            "def outer(x: int) -> int:\n    return inner(x) + 1\n",
        ],
    )
    for name in ("inner", "outer"):
        _pass_static(fm, name)
        assert _record(fm, name, VerdictKind.tier0, "PASS", "s") is False

    # Dependency content change: the dependent loses trust before its next call.
    fm.add_functions(
        implementations="def inner(x: int) -> int:\n    return x + 0\n",
        overwrite=True,
    )
    assert fm._get_function_data_by_name(name="inner")["verify"] is True
    assert fm._get_function_data_by_name(name="outer")["verify"] is True

    # Re-earn, then a venv change on inner invalidates inner and outer again.
    for name in ("inner", "outer"):
        _pass_static(fm, name)
        assert _record(fm, name, VerdictKind.tier0, "PASS", "s") is False
    venv_id = fm.add_venv(venv="[project]\nname='x'\nversion='0'\n")
    fm.set_function_venv(function_id=_fid(fm, "inner"), venv_id=venv_id)
    assert fm._get_function_data_by_name(name="inner")["verify"] is True
    assert fm._get_function_data_by_name(name="outer")["verify"] is True

    for name in ("inner", "outer"):
        _pass_static(fm, name)
        assert _record(fm, name, VerdictKind.tier0, "PASS", "s") is False
    fm.update_venv(venv_id=venv_id, venv="[project]\nname='y'\nversion='0'\n")
    assert fm._get_function_data_by_name(name="inner")["verify"] is True
    assert fm._get_function_data_by_name(name="outer")["verify"] is True


@_handle_project
def test_guidance_edit_and_delete_invalidate_linked_functions():
    ManagerRegistry.clear()
    fm = FunctionManager()
    gm = GuidanceManager()
    fm.add_functions(implementations=_PURE)
    fid = _fid(fm, "add")
    outcome = gm.add_guidance(
        title="Adding rule",
        content="Add numbers carefully.",
        function_ids=[fid],
    )
    guidance_id = int(outcome["details"]["guidance_id"])
    assert guidance_id in fm._get_guidance_ids_for_function(function_id=fid)
    _pass_static(fm, "add")
    assert _record(fm, "add", VerdictKind.tier0, "PASS", "s") is False

    gm.update_guidance(guidance_id=guidance_id, content="Add numbers very carefully.")
    row = fm._get_function_data_by_name(name="add")
    assert row["verify"] is True
    assert row["verified_hash"] is None
    assert [r["kind"] for r in row["stale_reasons"]] == ["guidance_changed"]

    _pass_static(fm, "add")
    assert _record(fm, "add", VerdictKind.tier0, "PASS", "s") is False
    gm.delete_guidance(guidance_id=guidance_id)
    row = fm._get_function_data_by_name(name="add")
    assert row["verify"] is True
    ManagerRegistry.clear()


@_handle_project
def test_repair_invalidates_and_fixture_replay_retrusts_pure_leaf():
    fm = FunctionManager()
    fm.add_functions(
        implementations=_PURE,
        fixtures={"add": [{"args": {"a": 1, "b": 2}, "result": 3}]},
    )
    _pass_static(fm, "add")
    assert _record(fm, "add", VerdictKind.tier0, "PASS", "s") is False

    # A repair (overwrite) invalidates; replay against the fixture reseeds the
    # ledger under the new hash, so only the static review is outstanding.
    fm.add_functions(
        implementations="def add(a: int, b: int) -> int:\n    return b + a\n",
        overwrite=True,
    )
    row = fm._get_function_data_by_name(name="add")
    assert row["verify"] is True
    assert row["static_review"] is None
    assert row["ledger"]["passes"] == {"tier0": 1}
    assert row["verified_hash"] == fm.function_trust_hash(row)
    _pass_static(fm, "add")
    fm.refresh_trust(_fid(fm, "add"))
    assert fm._get_function_data_by_name(name="add")["verify"] is False


@_handle_project
def test_stale_hash_rows_are_history_not_evidence():
    fm = FunctionManager()
    fm.add_functions(implementations=_READ)
    _pass_static(fm, "lookup")
    _ramp(fm, "lookup", passes=3, inputs=2)
    assert fm._get_function_data_by_name(name="lookup")["verify"] is False
    fm.add_functions(
        implementations=_READ.replace("ask(q)", "ask(q.strip())"),
        overwrite=True,
    )
    row = fm._get_function_data_by_name(name="lookup")
    assert row["verify"] is True
    fid = _fid(fm, "lookup")
    assert len(fm.list_verifications(function_id=fid)) == 6  # history kept
    fm.refresh_trust(fid)
    row = fm._get_function_data_by_name(name="lookup")
    assert row["ledger"]["passes"] == {}
    assert row["verify"] is True
