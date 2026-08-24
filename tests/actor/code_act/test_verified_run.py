"""
Symbolic tests for the run supervisor: pending verdicts, the total-order
barrier, the memo across rewinds, repair targeting and escalation, holds,
cancellation, and the zero-overhead path for a trusted closure.

The verifier passes are stubbed with controllable verdicts (no LLM). Stored
functions append to a trace file so the tests can assert exactly what ran
and in what order.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

from tests.helpers import _handle_project
from unify.actor.code_act_actor import CodeActActor
from unify.actor import verification_runtime as vr
from unify.actor.verification_runtime import (
    RewindRequested,
    VerifierPasses,
)
from unify.function_manager.function_manager import FunctionManager
from unify.function_manager.types.verification import Verdict
from unify.task_scheduler.types.execution import ExecutionState

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _verification_enabled(monkeypatch):
    """This module exercises the verification machinery itself; the master
    switch defaults off, so hold it on for every test here."""
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS.function.verification, "enabled", True)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


class Gate:
    """A thread-safe trigger the stubbed passes can wait on from any loop."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def open(self) -> None:
        self._event.set()

    async def wait(self, timeout: float = 300.0) -> None:
        deadline = time.monotonic() + timeout
        while not self._event.is_set():
            if time.monotonic() > deadline:
                raise AssertionError("gate never opened")
            await asyncio.sleep(0.01)


def _trace(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _wait_until(predicate, *, timeout: float = 300.0):
    async def _go():
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            await asyncio.sleep(0.02)
        raise AssertionError("condition not met in time")

    return _go()


def _src(
    name: str,
    body: str,
    *,
    params: str = "trace_path: str",
    ret: str = "dict",
) -> str:
    return (
        f"def {name}({params}) -> {ret}:\n"
        f'    """{name} for the verified-run tests."""\n'
        "    import json, pathlib\n"
        f"    pathlib.Path(trace_path).open('a').write(json.dumps({{'fn': '{name}'}}) + '\\n')\n"
        f"{body}"
    )


_FETCH = _src("fetch_rows", "    return {'rows': [1, 2, 3]}\n")
_SUMMARISE = (
    "def summarise(rows: dict, trace_path: str) -> dict:\n"
    '    """Summarise the rows."""\n'
    "    import json, pathlib\n"
    "    pathlib.Path(trace_path).open('a').write(json.dumps({'fn': 'summarise', 'n': len(rows['rows'])}) + '\\n')\n"
    "    return {'total': sum(rows['rows'])}\n"
)
_TRUSTED_LOOKUP = _src("trusted_lookup", "    return {'owner': 'finance'}\n")
_SEND = (
    "def send_report(summary: dict, owner: dict, trace_path: str) -> dict:\n"
    '    """Send the report."""\n'
    "    import json, pathlib\n"
    "    pathlib.Path(trace_path).open('a').write(json.dumps({'fn': 'send_report', 'total': summary['total']}) + '\\n')\n"
    "    return {'sent': True}\n"
)
_ROOT = (
    "def weekly_root(trace_path: str) -> dict:\n"
    '    """Fetch, summarise, look up the owner and send."""\n'
    "    rows = fetch_rows(trace_path)\n"
    "    summary = summarise(rows, trace_path)\n"
    "    owner = trusted_lookup(trace_path)\n"
    "    return send_report(summary, owner, trace_path)\n"
)


def _classify(fm: FunctionManager, name: str, klass: str) -> None:
    fid = fm._get_function_data_by_name(name=name)["function_id"]
    fm._persist_verification_fields(
        function_id=fid,
        fields={
            "side_effect_class": klass,
            "class_source": "librarian",
            "class_rationale": "test",
        },
    )


def _trust(fm: FunctionManager, name: str) -> None:
    """Earn trust for ``name`` the only way it can be earned: with ledger evidence."""
    from unify.function_manager.types.verification import (
        StaticReviewRecord,
        VerdictKind,
        VerificationRow,
    )
    from unify.function_manager.verification.policy import (
        min_distinct_inputs,
        required_passes,
    )

    row = fm._get_function_data_by_name(name=name)
    fid = int(row["function_id"])
    current = fm.function_trust_hash(row)
    fm._persist_verification_fields(
        function_id=fid,
        fields={
            "static_review": StaticReviewRecord(
                verdict="PASS",
                reason="test",
                function_hash=current,
            ).model_dump(mode="json"),
        },
    )
    settings = fm.verification_settings
    needed = required_passes(row, settings)
    inputs = max(1, min_distinct_inputs(row, settings))
    kinds = [VerdictKind.args, VerdictKind.post] if needed > 0 else [VerdictKind.tier0]
    for i in range(max(needed, 1)):
        for kind in kinds:
            fm.record_verification(
                VerificationRow(
                    function_id=fid,
                    function_hash=current,
                    kind=kind,
                    verdict="PASS",
                    args_signature=f"seed{i % inputs}",
                ),
            )
    assert fm._get_function_data_by_name(name=name)["verify"] is False


class PassStubs:
    """Controllable verdicts per (pass, function name); everything else passes."""

    def __init__(self, *, record: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.plan: dict[tuple[str, str], list] = {}
        self.gates: dict[tuple[str, str], Gate] = {}
        # When set, verdicts are written to the ledger like a real pass would.
        self.record = record

    def install(self, monkeypatch) -> None:
        stubs = self

        async def _run(
            passes,
            kind: str,
            row: dict,
            kwargs=None,
            verdict_kind=None,
        ) -> Verdict:
            key = (kind, str(row.get("name")))
            stubs.calls.append(key)
            gate = stubs.gates.get(key)
            if gate is not None:
                await gate.wait()
            queue = stubs.plan.get(key)
            verdict = Verdict(verdict="PASS", reason="stub")
            if queue:
                verdict = queue.pop(0)
                if isinstance(verdict, BaseException):
                    raise verdict
            if stubs.record:
                from unify.actor.verification_runtime import PassUsage
                from unify.function_manager.types.verification import VerdictKind

                passes._record(
                    row,
                    kind=verdict_kind or VerdictKind(kind),
                    verdict=verdict,
                    call_site="root",
                    kwargs=kwargs,
                    usage=PassUsage(),
                    wall_ms=0,
                )
            return verdict

        async def static_review(self, row):
            return await _run(self, "static", row)

        async def args_review(self, row, *, kwargs=None, **_):
            return await _run(self, "args", row, kwargs)

        async def precondition_probe(self, row, *, kwargs=None, **_):
            return await _run(self, "precondition", row, kwargs)

        async def post_probe(self, row, *, kwargs=None, kind=None, **_):
            return await _run(self, "post", row, kwargs, verdict_kind=kind)

        monkeypatch.setattr(VerifierPasses, "static_review", static_review)
        monkeypatch.setattr(VerifierPasses, "args_review", args_review)
        monkeypatch.setattr(VerifierPasses, "precondition_probe", precondition_probe)
        monkeypatch.setattr(VerifierPasses, "post_probe", post_probe)

    def fail(
        self,
        kind: str,
        name: str,
        *,
        fault: str = "leaf",
        reason: str = "stub fail",
    ) -> None:
        self.plan.setdefault((kind, name), []).append(
            Verdict(verdict="FAIL", reason=reason, fault=fault),
        )

    def unsure(self, kind: str, name: str, *, reason: str = "cannot tell") -> None:
        self.plan.setdefault((kind, name), []).append(
            Verdict(verdict="UNSURE", reason=reason),
        )

    def gate(self, kind: str, name: str) -> Gate:
        return self.gates.setdefault((kind, name), Gate())


async def _setup(
    tmp_path: Path,
    monkeypatch,
    *,
    trusted: tuple[str, ...] = ("trusted_lookup",),
):
    fm = FunctionManager()
    fm.add_functions(
        implementations=[_FETCH, _SUMMARISE, _TRUSTED_LOOKUP, _SEND, _ROOT],
    )
    _classify(fm, "fetch_rows", "read_only")
    _classify(fm, "trusted_lookup", "read_only")
    _classify(fm, "send_report", "unsafe_effectful")
    _classify(fm, "weekly_root", "unsafe_effectful")
    for name in trusted:
        _trust(fm, name)
    stubs = PassStubs()
    stubs.install(monkeypatch)
    actor = CodeActActor(function_manager=fm)
    root_id = fm._get_function_data_by_name(name="weekly_root")["function_id"]
    trace = tmp_path / "trace.jsonl"
    return fm, actor, stubs, root_id, trace


async def _run(actor, root_id, trace):
    handle = await actor.act(
        "Weekly report run.",
        entrypoint=root_id,
        entrypoint_kwargs={"trace_path": str(trace), "task_id": 1, "run_key": "run-1"},
        entrypoint_repair_context={"task_name": "Weekly report"},
        clarification_enabled=False,
        persist=False,
    )
    result = await handle.result()
    return handle, result


# ────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
async def test_happy_path_runs_every_pass_and_delivers_after_drain(
    tmp_path,
    monkeypatch,
):
    fm, actor, stubs, root_id, trace = await _setup(tmp_path, monkeypatch)
    try:
        handle, result = await _run(actor, root_id, trace)
        assert result == "{'sent': True}"
        assert handle.held_outcome is None
        assert [t["fn"] for t in _trace(trace)] == [
            "fetch_rows",
            "summarise",
            "trusted_lookup",
            "send_report",
        ]
        kinds = {(k, n) for k, n in stubs.calls}
        assert ("static", "fetch_rows") in kinds and ("post", "fetch_rows") in kinds
        assert ("args", "send_report") in kinds and (
            "precondition",
            "send_report",
        ) in kinds
        assert ("post", "summarise") not in kinds  # safe_noop: tier-0 only
        assert not any(n == "trusted_lookup" for _, n in stubs.calls)
        assert handle.run_stats["verdicts"]["FAIL"] == 0
        assert handle.run_stats["verdicts"]["PASS"] >= 6
    finally:
        await actor.close()


@_handle_project
async def test_effectful_leaf_waits_for_every_earlier_verdict(tmp_path, monkeypatch):
    fm, actor, stubs, root_id, trace = await _setup(tmp_path, monkeypatch)
    gate = stubs.gate("post", "fetch_rows")
    try:
        run = asyncio.create_task(_run(actor, root_id, trace))
        # summarise and trusted_lookup run; send_report must not.
        await _wait_until(lambda: len(_trace(trace)) >= 3)
        await asyncio.sleep(0.5)
        assert [t["fn"] for t in _trace(trace)] == [
            "fetch_rows",
            "summarise",
            "trusted_lookup",
        ]
        gate.open()
        handle, result = await run
        assert [t["fn"] for t in _trace(trace)][-1] == "send_report"
        assert handle.held_outcome is None
    finally:
        await actor.close()


@_handle_project
async def test_fail_rewinds_repairs_leaf_and_memo_skips_trusted_and_passed_calls(
    tmp_path,
    monkeypatch,
):
    fm, actor, stubs, root_id, trace = await _setup(tmp_path, monkeypatch)
    gate = stubs.gate("post", "fetch_rows")
    stubs.fail("post", "fetch_rows", fault="leaf", reason="rows are stale")
    repairs: list[RewindRequested] = []

    async def _repair(self, *, function_id, **kwargs):
        repairs.append(kwargs)
        assert (
            function_id
            == fm._get_function_data_by_name(name="fetch_rows")["function_id"]
        )
        fm.add_functions(
            implementations=_FETCH.replace("[1, 2, 3]", "[10, 20]"),
            overwrite=True,
        )
        _classify(fm, "fetch_rows", "read_only")

    monkeypatch.setattr(CodeActActor, "_repair_function", _repair)
    try:
        run = asyncio.create_task(_run(actor, root_id, trace))
        await _wait_until(lambda: len(_trace(trace)) >= 3)
        gate.open()
        handle, result = await run
        names = [t["fn"] for t in _trace(trace)]
        # Attempt 1: fetch, summarise, trusted_lookup, then the FAIL landed and send never ran.
        # Attempt 2: repaired fetch re-executes, summarise re-executes with new args,
        # trusted_lookup is served from the memo, send runs.
        assert names == [
            "fetch_rows",
            "summarise",
            "trusted_lookup",
            "fetch_rows",
            "summarise",
            "send_report",
        ]
        assert _trace(trace)[-2]["n"] == 2
        assert _trace(trace)[-1]["total"] == 30
        assert len(repairs) == 1
        assert repairs[0]["verdict"].reason == "rows are stale"
        assert handle.run_stats["rewinds"] == 1
        assert handle.held_outcome is None
    finally:
        await actor.close()


@_handle_project
async def test_unsure_on_barrier_holds_and_effect_never_runs(tmp_path, monkeypatch):
    fm, actor, stubs, root_id, trace = await _setup(tmp_path, monkeypatch)
    stubs.unsure(
        "post",
        "fetch_rows",
        reason="cannot tell whether the rows are current",
    )
    try:
        handle, result = await _run(actor, root_id, trace)
        assert handle.held_outcome is not None
        assert handle.held_outcome.code == "unsure"
        assert handle.held_outcome.leaf_name == "fetch_rows"
        assert result.startswith("Holding Weekly report: could not verify fetch_rows")
        assert "cannot tell whether the rows are current" in result
        assert "Nothing was sent or changed" in result
        assert "send_report" not in [t["fn"] for t in _trace(trace)]
        assert handle.run_stats["held_reason"].startswith("unsure:")
    finally:
        await actor.close()


@_handle_project
async def test_exhausted_rewinds_hold(tmp_path, monkeypatch):
    fm, actor, stubs, root_id, trace = await _setup(tmp_path, monkeypatch)
    for _ in range(4):
        stubs.fail("args", "send_report", fault="caller", reason="wrong recipient")

    async def _repair(self, **kwargs):
        return "no change"

    monkeypatch.setattr(CodeActActor, "_repair_function", _repair)
    try:
        handle, result = await _run(actor, root_id, trace)
        assert handle.held_outcome is not None
        assert handle.held_outcome.code == "exhausted"
        assert (
            handle.run_stats["rewinds"] == fm.verification_settings.max_rewinds_per_run
        )
        assert "send_report" not in [t["fn"] for t in _trace(trace)]
        assert "wrong recipient" in result
    finally:
        await actor.close()


@_handle_project
async def test_caller_fault_targets_parent_and_repeat_escalates(tmp_path, monkeypatch):
    fm, actor, stubs, root_id, trace = await _setup(tmp_path, monkeypatch)
    stubs.fail("args", "send_report", fault="caller", reason="wrong recipient")
    stubs.fail("args", "send_report", fault="caller", reason="still wrong")
    targets: list[str] = []

    async def _repair(self, *, function_id, **kwargs):
        row = fm.filter_functions(filter=f"function_id == {function_id}")[0]
        targets.append(row["name"])

    monkeypatch.setattr(CodeActActor, "_repair_function", _repair)
    monkeypatch.setattr(fm.verification_settings, "max_rewinds_per_run", 3)
    try:
        handle, result = await _run(actor, root_id, trace)
        # fault=caller blames the parent (weekly_root); the second failure on
        # the same target escalates one frame up — the root has no parent, so
        # it stays the target.
        assert targets[:1] == ["weekly_root"]
        assert len(targets) >= 2
    finally:
        await actor.close()


@_handle_project
async def test_second_failure_on_same_leaf_escalates_to_parent(tmp_path, monkeypatch):
    fm, actor, stubs, root_id, trace = await _setup(tmp_path, monkeypatch)
    stubs.fail("post", "fetch_rows", fault="leaf", reason="bad rows")
    stubs.fail("post", "fetch_rows", fault="leaf", reason="bad rows again")
    targets: list[str] = []

    async def _repair(self, *, function_id, **kwargs):
        row = fm.filter_functions(filter=f"function_id == {function_id}")[0]
        targets.append(row["name"])

    monkeypatch.setattr(CodeActActor, "_repair_function", _repair)
    monkeypatch.setattr(fm.verification_settings, "max_rewinds_per_run", 3)
    try:
        await _run(actor, root_id, trace)
        assert targets[:2] == ["fetch_rows", "weekly_root"]
    finally:
        await actor.close()


@_handle_project
async def test_trusted_closure_creates_zero_verifier_tasks(tmp_path, monkeypatch):
    fm, actor, stubs, root_id, trace = await _setup(
        tmp_path,
        monkeypatch,
        trusted=(
            "fetch_rows",
            "summarise",
            "trusted_lookup",
            "send_report",
            "weekly_root",
        ),
    )
    created = []
    original_submit = vr.VerifierExecutor.submit

    def _submit(self, *args, **kwargs):
        created.append(kwargs.get("label"))
        return original_submit(self, *args, **kwargs)

    monkeypatch.setattr(vr.VerifierExecutor, "submit", _submit)
    try:
        handle, result = await _run(actor, root_id, trace)
        assert result == "{'sent': True}"
        assert created == []
        assert stubs.calls == []
        assert handle.run_stats["verifier_tasks"] == 0
        assert [t["fn"] for t in _trace(trace)] == [
            "fetch_rows",
            "summarise",
            "trusted_lookup",
            "send_report",
        ]
    finally:
        await actor.close()


@_handle_project
async def test_stop_cancels_pending_verdicts_and_records_unsure_cancelled(
    tmp_path,
    monkeypatch,
):
    fm, actor, stubs, root_id, trace = await _setup(tmp_path, monkeypatch)
    gate = stubs.gate("post", "fetch_rows")
    try:
        handle = await actor.act(
            "Weekly report run.",
            entrypoint=root_id,
            entrypoint_kwargs={
                "trace_path": str(trace),
                "task_id": 1,
                "run_key": "run-1",
            },
            clarification_enabled=False,
            persist=False,
        )
        await _wait_until(lambda: len(_trace(trace)) >= 3)
        await handle.stop(reason="operator stop")
        result = await handle.result()
        assert "stopped" in result
        gate.open()
        fetch_id = fm._get_function_data_by_name(name="fetch_rows")["function_id"]
        rows = await _wait_until(
            lambda: [
                r
                for r in fm.list_verifications(function_id=fetch_id)
                if r["kind"] == "post"
            ]
            or None,
        )
        assert rows[0]["verdict"] == "UNSURE" and rows[0]["reason"] == "cancelled"
        assert "send_report" not in [t["fn"] for t in _trace(trace)]
    finally:
        await actor.close()


@_handle_project
async def test_deployment_owned_function_failure_holds(tmp_path, monkeypatch):
    fm, actor, stubs, root_id, trace = await _setup(tmp_path, monkeypatch)
    fetch_id = fm._get_function_data_by_name(name="fetch_rows")["function_id"]
    fm._persist_verification_fields(
        function_id=fetch_id,
        fields={"custom_hash": "abc123", "custom_key": "fetch_rows"},
    )
    stubs.fail("post", "fetch_rows", fault="leaf", reason="rows are stale")
    try:
        handle, result = await _run(actor, root_id, trace)
        assert handle.held_outcome is not None
        assert handle.held_outcome.code == "deployment_owned_function_failed"
        assert "owned by the deployment" in result
        assert "send_report" not in [t["fn"] for t in _trace(trace)]
    finally:
        await actor.close()


@_handle_project
async def test_active_task_persists_held_state(monkeypatch):
    from unify.task_scheduler import active_task as at

    written: list[dict] = []
    monkeypatch.setattr(
        at,
        "update_task_run_record",
        lambda ref, updates: written.append(dict(updates)),
    )

    class _Handle:
        held_outcome = vr.HeldOutcome(
            code="unsure",
            leaf_name="send_report",
            reason="cannot tell",
            payload={"x": 1},
        )
        run_stats = {
            "verdicts": {"PASS": 2, "FAIL": 0, "UNSURE": 1},
            "rewinds": 0,
            "verifier_tasks": 3,
            "held_reason": "unsure: cannot tell",
        }

        async def result(self):
            return "Holding Weekly report: could not verify send_report (cannot tell)."

        def done(self):
            return True

    task = at.ActiveTask(
        _Handle(),
        task_id=None,
        scheduler=None,
        task_run_reference=object(),
    )
    out = await task.result()
    assert out.startswith("Holding")
    assert written and written[0]["state"] == ExecutionState.held.value
    assert written[0]["held_reason"] == "unsure: cannot tell"
    assert written[0]["held_payload"] == {"x": 1}
    assert written[0]["verdicts"] == {"PASS": 2, "FAIL": 0, "UNSURE": 1}
    assert ExecutionState.held.is_terminal and not ExecutionState.held.is_open


_UNTYPED_SEND = (
    "def untyped_send(trace_path: str):\n"
    '    """Send without a declared output."""\n'
    "    import json, pathlib\n"
    "    pathlib.Path(trace_path).open('a').write(json.dumps({'fn': 'untyped_send'}) + '\\n')\n"
    "    return {'sent': True}\n"
)
_SC_ROOT = (
    "def spot_root(trace_path: str) -> dict:\n"
    '    """Send via the untyped sender."""\n'
    "    return untyped_send(trace_path)\n"
)


@_handle_project
async def test_spot_check_fail_invalidates_trust_and_notifies_owner(
    tmp_path,
    monkeypatch,
):
    fm = FunctionManager()
    fm.add_functions(implementations=[_UNTYPED_SEND, _SC_ROOT])
    _classify(fm, "untyped_send", "unsafe_effectful")
    _classify(fm, "spot_root", "unsafe_effectful")
    _trust(fm, "untyped_send")
    _trust(fm, "spot_root")
    assert (
        fm._get_function_data_by_name(name="untyped_send")["contract"]["output_schema"]
        is None
    )
    monkeypatch.setattr(
        fm.verification_settings,
        "spot_check_rate",
        {"unsafe_effectful": 1.0, "idempotent_effectful": 1.0},
    )
    stubs = PassStubs(record=True)
    stubs.install(monkeypatch)
    stubs.fail(
        "post",
        "untyped_send",
        fault="leaf",
        reason="the message body was empty",
    )
    actor = CodeActActor(function_manager=fm)
    root_id = fm._get_function_data_by_name(name="spot_root")["function_id"]
    trace = tmp_path / "trace.jsonl"
    try:
        handle, result = await _run(actor, root_id, trace)
        # The effect ran once and the result was delivered; nothing was repeated.
        assert result == "{'sent': True}"
        assert handle.held_outcome is None
        assert [t["fn"] for t in _trace(trace)] == ["untyped_send"]
        notification = await asyncio.wait_for(handle.next_notification(), timeout=300)
        assert "Spot check of Weekly report" in notification["message"]
        assert "the message body was empty" in notification["message"]
        assert "was not repeated" in notification["message"]
        row = fm._get_function_data_by_name(name="untyped_send")
        assert row["verify"] is True
        assert row["ledger"]["fails"] >= 1
        rows = fm.list_verifications(function_id=int(row["function_id"]))
        assert [
            (r["kind"], r["verdict"]) for r in rows if r["kind"] == "spot_check"
        ] == [("spot_check", "FAIL")]
        assert ("post", "untyped_send") in stubs.calls
    finally:
        await actor.close()
