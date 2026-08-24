"""
The librarian shapes verification but never grants trust.

Symbolic: ``confirm_side_effect_class`` raises freely and lowers only to the
detected bound; ``set_verification_policy`` only raises the bar; both tools
are wired into the storage and repair loops.

Eval (cached): given a trajectory that computes and then sends, the librarian
stores the computation and the effect as separate functions plus a root, and
any class it confirms stays within bounds.
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import MagicMock

import pytest

from tests.helpers import _handle_project
from unify.actor.code_act_actor import (
    CodeActActor,
    _verification_librarian_tools,
)
from unify.function_manager.function_manager import FunctionManager
from unify.manager_registry import ManagerRegistry


@pytest.fixture(autouse=True)
def _verification_enabled(monkeypatch):
    """This module exercises the verification machinery itself; the master
    switch defaults off, so hold it on for every test here."""
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS.function.verification, "enabled", True)


_PURE = "def add(a: int, b: int) -> int:\n    return a + b\n"
_READ = (
    "async def lookup(q: str) -> str:\n    return await primitives.contacts.ask(q)\n"
)
_THIRD_PARTY = "def s3(x: str) -> str:\n    import boto3\n    return x\n"


def _row(fm, name):
    return fm._get_function_data_by_name(name=name)


@_handle_project
def test_confirm_side_effect_class_is_bounded_below_by_detection():
    fm = FunctionManager()
    fm.add_functions(implementations=[_PURE, _READ])
    venv_id = fm.add_venv(
        venv="[project]\nname='s3'\nversion='0'\ndependencies=['boto3']\n",
    )
    fm.add_functions(implementations=_THIRD_PARTY, venv_id=venv_id)
    add_id = _row(fm, "add")["function_id"]
    lookup_id = _row(fm, "lookup")["function_id"]
    s3_id = _row(fm, "s3")["function_id"]

    # Raising is free.
    out = fm.confirm_side_effect_class(
        function_id=add_id,
        side_effect_class="unsafe_effectful",
        rationale="it drives a desktop",
    )
    assert out["outcome"] == "confirmed" and out["effective"] == "unsafe_effectful"
    row = _row(fm, "add")
    assert row["side_effect_class"] == "unsafe_effectful"
    assert (
        row["class_source"] == "librarian"
        and row["class_rationale"] == "it drives a desktop"
    )
    assert row["side_effect_class_detected"] == "safe_noop"

    # Lowering stops at the detected bound.
    out = fm.confirm_side_effect_class(
        function_id=lookup_id,
        side_effect_class="safe_noop",
        rationale="looks pure",
    )
    assert out["outcome"] == "rejected" and out["detected"] == "read_only"
    assert _row(fm, "lookup")["side_effect_class"] == "read_only"
    out = fm.confirm_side_effect_class(
        function_id=lookup_id,
        side_effect_class="read_only",
        rationale="reads contacts",
    )
    assert out["outcome"] == "confirmed"

    # An unconfirmed third-party inference is treated as unsafe; confirming
    # at the detected bound replaces that safe default.
    s3 = _row(fm, "s3")
    assert (
        s3["side_effect_class_detected"] == "read_only"
        and s3["class_source"] == "inferred_third_party"
    )
    out = fm.confirm_side_effect_class(
        function_id=s3_id,
        side_effect_class="read_only",
        rationale="boto3 read of a public bucket",
    )
    assert out["outcome"] == "confirmed"
    assert _row(fm, "s3")["class_source"] == "librarian"

    # Bad input and empty rationale are rejected without writes.
    assert (
        fm.confirm_side_effect_class(
            function_id=add_id,
            side_effect_class="mystery",
            rationale="x",
        )["outcome"]
        == "rejected"
    )
    assert (
        fm.confirm_side_effect_class(
            function_id=add_id,
            side_effect_class="read_only",
            rationale="  ",
        )["outcome"]
        == "rejected"
    )


@_handle_project
def test_set_verification_policy_only_raises():
    fm = FunctionManager()
    fm.add_functions(implementations=[_PURE, _READ])
    add_id = _row(fm, "add")["function_id"]
    lookup_id = _row(fm, "lookup")["function_id"]

    # read_only class default is 3 passes / 2 inputs; equal or lower is rejected.
    out = fm.set_verification_policy(function_id=lookup_id, required_passes=3)
    assert out["outcome"] == "rejected"
    out = fm.set_verification_policy(
        function_id=lookup_id,
        required_passes=5,
        min_distinct_inputs=4,
    )
    assert out["outcome"] == "raised"
    assert _row(fm, "lookup")["verification_policy"]["required_passes"] == 5
    assert (
        fm.set_verification_policy(function_id=lookup_id, required_passes=4)["outcome"]
        == "rejected"
    )
    assert (
        fm.set_verification_policy(function_id=lookup_id, required_passes=6)["outcome"]
        == "raised"
    )

    # always_verify can be set but never unset.
    assert (
        fm.set_verification_policy(function_id=lookup_id, always_verify=True)["outcome"]
        == "raised"
    )
    assert (
        fm.set_verification_policy(function_id=lookup_id, always_verify=False)[
            "outcome"
        ]
        == "rejected"
    )

    # spot checks: read_only has none; the rate must exceed the current one.
    assert (
        fm.set_verification_policy(function_id=lookup_id, spot_check_rate=1.5)[
            "outcome"
        ]
        == "rejected"
    )
    assert (
        fm.set_verification_policy(function_id=lookup_id, spot_check_rate=0.5)[
            "outcome"
        ]
        == "raised"
    )
    assert (
        fm.set_verification_policy(function_id=lookup_id, spot_check_rate=0.5)[
            "outcome"
        ]
        == "rejected"
    )

    # fixture_only is for pure functions only.
    assert (
        fm.set_verification_policy(function_id=lookup_id, fixture_only=True)["outcome"]
        == "rejected"
    )
    assert (
        fm.set_verification_policy(function_id=add_id, fixture_only=True)["outcome"]
        == "raised"
    )
    assert fm.set_verification_policy(function_id=add_id)["outcome"] == "unchanged"


def test_librarian_tools_are_wired_into_storage_and_repair(monkeypatch):
    fm = MagicMock()
    fm.confirm_side_effect_class.__doc__ = "confirm"
    fm.set_verification_policy.__doc__ = "policy"
    tools = _verification_librarian_tools(fm)
    assert set(tools) == {"confirm_side_effect_class", "set_verification_policy"}
    assert list(inspect.signature(tools["confirm_side_effect_class"]).parameters) == [
        "function_id",
        "side_effect_class",
        "rationale",
    ]
    assert (
        "spot_check_rate"
        in inspect.signature(tools["set_verification_policy"]).parameters
    )
    assert _verification_librarian_tools(None) == {}


@pytest.mark.asyncio
@_handle_project
async def test_repair_loop_receives_verification_tools(monkeypatch):
    ManagerRegistry.clear()
    fm = FunctionManager()
    actor = CodeActActor(function_manager=fm)
    try:
        fm.add_functions(implementations=_PURE)
        function_id = _row(fm, "add")["function_id"]
        seen: dict = {}

        class _FakeHandle:
            async def result(self):
                return "repaired"

        def _fake_loop(**kwargs):
            seen.update(kwargs)
            return _FakeHandle()

        monkeypatch.setattr(
            "unify.actor.code_act_actor.start_async_tool_loop",
            _fake_loop,
        )
        monkeypatch.setattr(
            "unify.actor.code_act_actor.new_llm_client",
            lambda model, **kwargs: MagicMock(set_system_message=MagicMock()),
        )
        await actor._repair_function(
            function_id=function_id,
            request="x",
            entrypoint_kwargs={},
            failure=RuntimeError("boom"),
            repair_context=None,
        )
        assert {
            "confirm_side_effect_class",
            "set_verification_policy",
            "run_diagnostic_probe",
        } <= set(seen["tools"])
        assert "FunctionManager_add_functions" in seen["tools"]
    finally:
        await actor.close()
        ManagerRegistry.clear()


class _TrackingGuidanceManager:
    def __init__(self) -> None:
        self.add_calls: list[dict] = []

    def search(self, references=None, k=10):
        """Search for guidance entries by semantic similarity to reference content."""
        return []

    def filter(self, filter=None, offset=0, limit=100):
        """Filter guidance entries using a Python filter expression."""
        return []

    def get_guidance(self, *, guidance_id):
        """Fetch one guidance entry by id with its complete content."""
        raise ValueError(f"No guidance found with guidance_id {guidance_id}.")

    def add_guidance(self, *, title, content, function_ids=None):
        """Add a guidance entry describing a compositional procedure or playbook."""
        self.add_calls.append(
            {"title": title, "content": content, "function_ids": function_ids},
        )
        return {"details": {"guidance_id": len(self.add_calls)}}

    def update_guidance(
        self,
        *,
        guidance_id,
        title=None,
        content=None,
        function_ids=None,
    ):
        """Update an existing guidance entry."""
        return {"details": {"guidance_id": guidance_id}}

    def delete_guidance(self, *, guidance_id):
        """Delete a guidance entry by ID."""
        return {"deleted": True}

    def reconcile_dependencies(self, *, guidance_ids=None, destination=None):
        """Refresh structured link debt for related functions."""
        return {"outcome": "checked", "details": {"guidance_ids": guidance_ids or []}}


@pytest.mark.eval
@pytest.mark.llm_call
@pytest.mark.asyncio
@pytest.mark.timeout(900)
@_handle_project
async def test_librarian_stores_thin_effects_and_confirms_within_bounds(monkeypatch):
    """A compute-then-send trajectory becomes a pure computation, an effect, and a root."""
    fm = FunctionManager(include_primitives=False)
    # The effect the task must use: a pre-stored sender whose source reveals no
    # effect (it returns a receipt), exactly the case a librarian should
    # confirm upward.
    fm.add_functions(
        implementations=(
            "def send_slack_report(channel: str, text: str) -> dict:\n"
            '    """Send the report text to the named Slack channel and return the delivery receipt."""\n'
            "    return {'sent': True, 'channel': channel, 'chars': len(text)}\n"
        ),
    )
    confirmations: list[dict] = []
    original_confirm = fm.confirm_side_effect_class

    def _tracking_confirm(**kwargs):
        out = original_confirm(**kwargs)
        confirmations.append({**kwargs, "outcome": out.get("outcome")})
        return out

    monkeypatch.setattr(fm, "confirm_side_effect_class", _tracking_confirm)
    gm = _TrackingGuidanceManager()
    actor = CodeActActor(function_manager=fm, guidance_manager=gm, timeout=180)
    try:
        handle = await actor.act(
            "Every Monday I need last week's order summary posted to Slack. Do it now for this "
            "week's data, in two clearly separate steps of Python: first compute the summary "
            "from the orders below (number of paid orders and their total amount in pence, "
            "ignoring refunded orders), then post the summary text to the '#finance' channel "
            "by calling the stored function send_slack_report(channel, text) — discover it "
            "with FunctionManager_search_functions and call it from execute_code. Do not "
            "post anywhere else and do not send twice.\n\n"
            "Orders:\n"
            "[{'id': 1, 'status': 'paid', 'amount': 1250},\n"
            " {'id': 2, 'status': 'refunded', 'amount': 700},\n"
            " {'id': 3, 'status': 'paid', 'amount': 750},\n"
            " {'id': 4, 'status': 'paid', 'amount': 2000}]",
            can_store=True,
            persist=False,
            clarification_enabled=False,
        )
        await asyncio.wait_for(handle.result(), timeout=300)
        deadline = asyncio.get_event_loop().time() + 400
        while not handle.done():
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError("Storage loop did not complete in time")
            await asyncio.sleep(0.5)

        rows = {r["name"]: r for r in fm.filter_functions(include_implementations=True)}
        new_rows = {n: r for n, r in rows.items() if n != "send_slack_report"}
        assert new_rows, f"expected the computation to be stored, got {sorted(rows)}"
        # Thin effects: the computation is stored pure, with the send kept apart.
        pure = [
            r
            for r in new_rows.values()
            if "send_slack_report" not in (r["implementation"] or "")
            and r["side_effect_class"] == "safe_noop"
        ]
        assert (
            pure
        ), f"no pure computation stored: { {n: r['side_effect_class'] for n, r in new_rows.items()}}"
        bundled = [
            n
            for n, r in new_rows.items()
            if "send_slack_report" in (r["implementation"] or "")
            and any(tok in (r["implementation"] or "") for tok in ("sum(", "amount"))
        ]
        assert (
            not bundled
        ), f"computation and effect were bundled into one function: {bundled}"
        # Every parameter and return of what was stored is hinted (a contract exists).
        assert all(r["contract"]["source"] != "none" for r in new_rows.values()), {
            n: r["contract"]["source"] for n, r in new_rows.items()
        }
        # Every confirmation stayed within bounds.
        assert all(c["outcome"] == "confirmed" for c in confirmations), confirmations
    finally:
        await actor.close()
