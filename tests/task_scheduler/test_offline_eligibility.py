"""
Offline promotion is derived from the ledger: a task is eligible when its
entrypoint's whole compositional closure is trusted; promotion happens
automatically when the last member flips, and loss of trust never demotes.

The verifier passes are stubbed with recording PASS verdicts (no LLM).
"""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unify.actor.code_act_actor import CodeActActor
from unify.actor.verification_runtime import PassUsage, VerifierPasses
from unify.function_manager.function_manager import FunctionManager
from unify.function_manager.types.verification import (
    StaticReviewRecord,
    Verdict,
    VerdictKind,
    VerificationRow,
)
from unify.manager_registry import ManagerRegistry
from unify.task_scheduler.task_scheduler import TaskScheduler

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _verification_enabled(monkeypatch):
    """This module derives eligibility from the trust ledger; the master
    switch defaults off, so hold it on for every test here."""
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS.function.verification, "enabled", True)


_HELPER = "def helper(x: int) -> int:\n    return x * 2\n"
_ROOT = (
    "def offline_root(x: int = 1) -> int:\n"
    '    """Double x."""\n'
    "    return helper(x)\n"
)


def _fid(fm, name):
    return int(fm._get_function_data_by_name(name=name)["function_id"])


def _classify(fm, name, klass):
    fm._persist_verification_fields(
        function_id=_fid(fm, name),
        fields={
            "side_effect_class": klass,
            "class_source": "librarian",
            "class_rationale": "test",
        },
    )


def _pass_static(fm, name):
    row = fm._get_function_data_by_name(name=name)
    fm._persist_verification_fields(
        function_id=int(row["function_id"]),
        fields={
            "static_review": StaticReviewRecord(
                verdict="PASS",
                reason="ok",
                function_hash=fm.function_trust_hash(row),
            ).model_dump(mode="json"),
        },
    )


def _record(fm, name, kind, sig):
    row = fm._get_function_data_by_name(name=name)
    fm.record_verification(
        VerificationRow(
            function_id=int(row["function_id"]),
            function_hash=fm.function_trust_hash(row),
            kind=kind,
            verdict="PASS",
            args_signature=sig,
        ),
    )


def _install_recording_stubs(monkeypatch):
    async def _judge(self, kind, row, kwargs, call_site):
        verdict = Verdict(verdict="PASS", reason="stub")
        self._record(
            row,
            kind=kind,
            verdict=verdict,
            call_site=call_site,
            kwargs=kwargs,
            usage=PassUsage(),
            wall_ms=0,
        )
        return verdict

    async def static_review(self, row):
        return await _judge(self, VerdictKind.static, row, None, "root")

    async def args_review(self, row, *, kwargs, stable_block, call_site="root", **_):
        return await _judge(self, VerdictKind.args, row, kwargs, call_site)

    async def precondition_probe(
        self,
        row,
        *,
        kwargs,
        stable_block,
        call_site="root",
        **_,
    ):
        return await _judge(self, VerdictKind.precondition, row, kwargs, call_site)

    async def post_probe(
        self,
        row,
        *,
        kwargs,
        result,
        stable_block,
        call_site="root",
        kind=VerdictKind.post,
        **_,
    ):
        return await _judge(self, kind, row, kwargs, call_site)

    monkeypatch.setattr(VerifierPasses, "static_review", static_review)
    monkeypatch.setattr(VerifierPasses, "args_review", args_review)
    monkeypatch.setattr(VerifierPasses, "precondition_probe", precondition_probe)
    monkeypatch.setattr(VerifierPasses, "post_probe", post_probe)


@_handle_project
async def test_offline_eligibility_truth_table():
    ManagerRegistry.clear()
    fm = FunctionManager()
    ts = TaskScheduler()
    fm.add_functions(implementations=[_HELPER, _ROOT])
    root_id = _fid(fm, "offline_root")
    task_id = ts._create_task(
        name="Doubler",
        description="Double a number.",
        entrypoint=root_id,
    )["details"]["task_id"]
    task = ts._get_task_or_raise(task_id)

    # Untrusted closure: not eligible, both ids named.
    ok, reasons = ts.offline_eligible(task)
    assert not ok and sorted(reasons) == sorted(
        [f"untrusted:{root_id}", f"untrusted:{_fid(fm, 'helper')}"],
    )
    assert ts.promote_task_offline(task_id=task_id)["outcome"] == "not_eligible"

    # No entrypoint: never eligible.
    plain_id = ts._create_task(name="Plain", description="Agentic.")["details"][
        "task_id"
    ]
    ok, reasons = ts.offline_eligible(ts._get_task_or_raise(plain_id))
    assert not ok and reasons == ["no_entrypoint"]

    # Trust the closure: eligible, promoted, idempotent.
    for name in ("helper", "offline_root"):
        _pass_static(fm, name)
        _record(fm, name, VerdictKind.tier0, "s")
    ok, reasons = ts.offline_eligible(ts._get_task_or_raise(task_id))
    assert ok and reasons == []
    assert ts.promote_task_offline(task_id=task_id)["outcome"] == "offline_promoted"
    assert ts._get_task_or_raise(task_id).offline is True
    assert ts.promote_task_offline(task_id=task_id)["outcome"] == "already_offline"

    # An unconfirmed unsafe class blocks eligibility even when trusted.
    fm._persist_verification_fields(
        function_id=_fid(fm, "helper"),
        fields={
            "side_effect_class": "unsafe_effectful",
            "class_source": "inferred_third_party",
        },
    )
    ok, reasons = ts.offline_eligible(ts._get_task_or_raise(task_id))
    assert not ok and any(r.startswith("unconfirmed_unsafe_class:") for r in reasons)

    # Loss of trust never demotes.
    fm.add_functions(implementations=_HELPER.replace("* 2", "* 2 + 0"), overwrite=True)
    ok, reasons = ts.offline_eligible(ts._get_task_or_raise(task_id))
    assert not ok
    assert ts._get_task_or_raise(task_id).offline is True
    ManagerRegistry.clear()


@_handle_project
async def test_auto_promotion_fires_when_last_leaf_flips(monkeypatch):
    ManagerRegistry.clear()
    fm = FunctionManager()
    ts = TaskScheduler()
    fm.add_functions(implementations=[_HELPER, _ROOT])
    _classify(fm, "helper", "read_only")
    root_id = _fid(fm, "offline_root")
    task_id = ts._create_task(
        name="Doubler",
        description="Double a number.",
        entrypoint=root_id,
    )["details"]["task_id"]

    # Root: trusted already. Helper (read_only): one pass short of the bar.
    _pass_static(fm, "offline_root")
    _record(fm, "offline_root", VerdictKind.tier0, "s")
    assert fm._get_function_data_by_name(name="offline_root")["verify"] is False
    _pass_static(fm, "helper")
    for i in range(2):
        _record(fm, "helper", VerdictKind.args, f"pre{i}")
        _record(fm, "helper", VerdictKind.post, f"pre{i}")
    assert fm._get_function_data_by_name(name="helper")["verify"] is True

    _install_recording_stubs(monkeypatch)
    actor = CodeActActor(function_manager=fm)
    try:
        handle = await actor.act(
            "Double it.",
            entrypoint=root_id,
            entrypoint_kwargs={"x": 4, "task_id": task_id, "run_key": "run-1"},
            clarification_enabled=False,
            persist=False,
        )
        assert await handle.result() == "8"
        assert handle.held_outcome is None
    finally:
        await actor.close()

    helper = fm._get_function_data_by_name(name="helper")
    assert helper["verify"] is False
    assert (
        helper["ledger"]["passes"]["args"] == 3
        and helper["ledger"]["passes"]["post"] == 3
    )
    assert ts._get_task_or_raise(task_id).offline is True
    ManagerRegistry.clear()


@_handle_project
async def test_auto_promotion_respects_setting_and_needs_every_member(monkeypatch):
    ManagerRegistry.clear()
    fm = FunctionManager()
    ts = TaskScheduler()
    fm.add_functions(implementations=[_HELPER, _ROOT])
    _classify(fm, "helper", "read_only")
    root_id = _fid(fm, "offline_root")
    task_id = ts._create_task(
        name="Doubler",
        description="Double a number.",
        entrypoint=root_id,
    )["details"]["task_id"]
    _pass_static(fm, "offline_root")
    _record(fm, "offline_root", VerdictKind.tier0, "s")
    _pass_static(fm, "helper")
    _install_recording_stubs(monkeypatch)

    # Only one run of evidence for the read_only helper: not yet trusted, so no promotion.
    actor = CodeActActor(function_manager=fm)
    try:
        handle = await actor.act(
            "Double it.",
            entrypoint=root_id,
            entrypoint_kwargs={"x": 1, "task_id": task_id},
            clarification_enabled=False,
            persist=False,
        )
        await handle.result()
    finally:
        await actor.close()
    assert fm._get_function_data_by_name(name="helper")["verify"] is True
    assert ts._get_task_or_raise(task_id).offline is False

    # Bring the helper to the bar but switch auto-promotion off: stays live.
    for i in range(2):
        _record(fm, "helper", VerdictKind.args, f"pre{i}")
        _record(fm, "helper", VerdictKind.post, f"pre{i}")
    monkeypatch.setattr(fm.verification_settings, "auto_promote_offline", False)
    actor = CodeActActor(function_manager=fm)
    try:
        handle = await actor.act(
            "Double it.",
            entrypoint=root_id,
            entrypoint_kwargs={"x": 2, "task_id": task_id},
            clarification_enabled=False,
            persist=False,
        )
        await handle.result()
    finally:
        await actor.close()
    assert fm._get_function_data_by_name(name="helper")["verify"] is False
    assert ts._get_task_or_raise(task_id).offline is False
    # The librarian tool path still promotes on request.
    assert ts.promote_task_offline(task_id=task_id)["outcome"] == "offline_promoted"
    ManagerRegistry.clear()
