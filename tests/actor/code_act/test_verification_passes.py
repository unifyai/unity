"""
Verifier passes: prompt assembly is stable per call site and hash-pure for
static review; verdict parsing enforces fault on FAIL; the intent-chain
context and call-site location work from compiled stored-function frames.

The eval tests at the bottom run each pass against a real (cached) model.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from pydantic import ValidationError

from tests.helpers import _handle_project
from unify.actor.prompt_builders import (
    build_args_review_prompt,
    build_call_stable_block,
    build_post_probe_prompt,
    build_static_review_prompt,
)
from unify.actor.verification_runtime import (
    Frame,
    VerifierPasses,
    _parse_verdict,
    current_verification_frames,
    locate_call_site,
    pushed_frame,
)
from unify.function_manager.function_manager import FunctionManager
from unify.function_manager.types.verification import Verdict
from unify.function_manager.verification.source_labels import (
    compile_function_source,
    function_source_filename,
)

# ────────────────────────────────────────────────────────────────────────────
# Prompt assembly
# ────────────────────────────────────────────────────────────────────────────

_LEAF = {
    "function_id": 7,
    "name": "post_summary",
    "docstring": "Post the weekly summary text to a Slack channel.",
    "implementation": (
        "async def post_summary(channel: str, text: str) -> dict:\n"
        '    """Post the weekly summary text to a Slack channel."""\n'
        "    return await primitives.comms.send_slack_channel_message(channel=channel, text=text)\n"
    ),
    "side_effect_class": "unsafe_effectful",
    "side_effect_class_detected": "unsafe_effectful",
    "contract": {
        "input_schema": {
            "type": "object",
            "properties": {"channel": {"type": "string"}, "text": {"type": "string"}},
            "required": ["channel", "text"],
        },
        "output_schema": {"type": "object"},
        "postconditions": [],
        "source": "type_hints",
    },
    "depends_on": ["primitives.comms.send_slack_channel_message"],
}
_ROOT_FRAME = Frame(
    function_id=1,
    name="weekly_report",
    docstring="Compute and post the weekly finance summary.",
    effect_class="unsafe_effectful",
    call_site_line="",
    args_repr="{}",
)
_LEAF_FRAME = Frame(
    function_id=7,
    name="post_summary",
    docstring=_LEAF["docstring"],
    effect_class="unsafe_effectful",
    call_site_line="    await post_summary(channel=channel, text=summary)",
    args_repr="",
)


def _stable() -> str:
    return build_call_stable_block(
        goal="Weekly finance report: compute last week's totals and post them to the #finance Slack channel.",
        guidance=[
            {"title": "Finance reporting", "content": "Always post to #finance."},
        ],
        frames=[_ROOT_FRAME.as_dict(), _LEAF_FRAME.as_dict()],
        leaf=_LEAF,
        parent_source="async def weekly_report():\n    summary = build()\n    await post_summary(channel=channel, text=summary)\n",
        call_line=3,
        children=[],
    )


def test_stable_block_is_byte_identical_across_calls_with_different_args():
    stable = _stable()
    p1 = build_args_review_prompt(
        stable_block=stable,
        kwargs={"channel": "#finance", "text": "a"},
        tier0="PASS",
    )
    p2 = build_args_review_prompt(
        stable_block=stable,
        kwargs={"channel": "#general", "text": "b"},
        tier0="PASS",
    )
    assert p1[0] == p2[0]  # static prefix constant
    assert p1[1] == p2[1]  # stable block identical for the call site
    assert p1[2] != p2[2]  # volatile differs
    assert "#general" not in p1[1] and "#general" in p2[2]
    post = build_post_probe_prompt(
        stable_block=stable,
        kwargs={"channel": "#finance", "text": "a"},
        result={"ok": True},
        tier0="PASS",
    )
    assert post[1] == p1[1]


def test_static_review_prompt_has_no_call_context():
    prefix, stable, volatile = build_static_review_prompt(
        _LEAF,
        dependencies=[{"name": "helper", "docstring": "Helps."}],
    )
    assert volatile == ""
    for forbidden in (
        "Goal of the run",
        "Call chain",
        "Arguments",
        "Returned value",
        "Linked guidance",
    ):
        assert forbidden not in stable
    assert "post_summary" in stable and "helper" in stable and "Contract" in stable
    assert "transcript" not in prefix.lower() and "trajectory" not in prefix.lower()


def test_static_prefixes_never_mention_transcript_or_trajectory():
    from unify.actor import prompt_builders as pb

    for prefix in (
        pb.STATIC_REVIEW_STATIC_PREFIX,
        pb.ARGS_REVIEW_STATIC_PREFIX,
        pb.PRECONDITION_PROBE_STATIC_PREFIX,
        pb.POST_PROBE_STATIC_PREFIX,
    ):
        low = prefix.lower()
        assert "transcript" not in low and "trajectory" not in low
        assert '"verdict"' in prefix and "UNSURE" in prefix


# ────────────────────────────────────────────────────────────────────────────
# Verdict parsing
# ────────────────────────────────────────────────────────────────────────────


def test_verdict_fail_requires_fault():
    with pytest.raises(ValidationError):
        Verdict(verdict="FAIL", reason="x")
    assert Verdict(verdict="FAIL", reason="x", fault="caller").fault == "caller"
    assert Verdict(verdict="PASS", reason="ok").fault is None


def test_parse_verdict_tolerates_fences_and_rejects_garbage():
    assert (
        _parse_verdict('{"verdict": "PASS", "reason": "fine", "fault": null}').verdict
        == "PASS"
    )
    assert (
        _parse_verdict('```json\n{"verdict": "UNSURE", "reason": "?"}\n```').verdict
        == "UNSURE"
    )
    assert (
        _parse_verdict(
            'Sure! {"verdict": "FAIL", "reason": "bad", "fault": "leaf"} done',
        ).fault
        == "leaf"
    )
    assert _parse_verdict('{"verdict": "FAIL", "reason": "no fault"}') is None
    assert _parse_verdict("not json at all") is None
    assert _parse_verdict(None) is None


# ────────────────────────────────────────────────────────────────────────────
# Intent chain + call sites
# ────────────────────────────────────────────────────────────────────────────


def test_pushed_frame_maintains_chain():
    assert current_verification_frames.get() == ()
    with pushed_frame(_ROOT_FRAME) as chain1:
        assert chain1 == (_ROOT_FRAME,)
        with pushed_frame(_LEAF_FRAME) as chain2:
            assert chain2 == (_ROOT_FRAME, _LEAF_FRAME)
            assert current_verification_frames.get() == chain2
        assert current_verification_frames.get() == (_ROOT_FRAME,)
    assert current_verification_frames.get() == ()


def test_locate_call_site_reads_the_compiled_stored_function_frame():
    captured = {}

    def leaf():
        captured["site"] = locate_call_site()
        return 1

    source = "def parent(x):\n    y = x + 1\n    return leaf() + y\n"
    namespace = {"leaf": leaf}
    exec(compile_function_source("parent", source), namespace)
    assert namespace["parent"](1) == 3
    site = captured["site"]
    assert site.parent_name == "parent"
    assert site.line_number == 3
    assert site.line_text == "    return leaf() + y"
    assert site.label == "parent"
    # Outside any stored function there is no call site.
    assert locate_call_site().parent_name is None
    assert function_source_filename("parent") == "<function:parent>"


# ────────────────────────────────────────────────────────────────────────────
# Passes against a real (cached) model
# ────────────────────────────────────────────────────────────────────────────

_ROOT_SRC = (
    "async def weekly_report() -> dict:\n"
    '    """Compute last week\'s totals and post them to the finance Slack channel."""\n'
    "    text = 'Weekly totals: 5 orders, 2000 pence'\n"
    "    return await post_summary(channel='#finance', text=text)\n"
)
_POST_SRC = _LEAF["implementation"]
_PAID_ONLY_SRC = (
    "def paid_orders(rows: list[dict]) -> list[dict]:\n"
    '    """Return only the orders whose status is exactly \'paid\'."""\n'
    "    return [r for r in rows if r.get('status') == 'paid']\n"
)
_PAID_ONLY_BROKEN_SRC = (
    "def paid_orders(rows: list[dict]) -> list[dict]:\n"
    '    """Return only the orders whose status is exactly \'paid\'."""\n'
    "    return list(rows)\n"
)
_TOTAL_MINOR_SRC = (
    "def total_minor(rows: list[dict]) -> int:\n"
    '    """Return the sum of `amount` over rows, in minor units (pence) as an int."""\n'
    "    return sum(r['amount'] for r in rows)\n"
)


def _wait_rows(fm, function_id, n, timeout=300.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = fm.list_verifications(function_id=function_id)
        if len(rows) >= n:
            return rows
        time.sleep(0.2)
    raise AssertionError("verification rows not written in time")


@pytest.mark.eval
@pytest.mark.llm_call
@_handle_project
def test_args_review_flags_wrong_recipient():
    fm = FunctionManager()
    fm.add_functions(implementations=[_POST_SRC, _ROOT_SRC])
    leaf = fm._get_function_data_by_name(name="post_summary")
    root = fm._get_function_data_by_name(name="weekly_report")
    passes = VerifierPasses(
        function_manager=fm,
        goal="Weekly finance report: compute last week's totals and post them to the #finance Slack channel.",
    )
    frames = [
        Frame(
            function_id=root["function_id"],
            name="weekly_report",
            docstring=root["docstring"],
            effect_class="unsafe_effectful",
            call_site_line="",
            args_repr="{}",
        ),
        Frame(
            function_id=leaf["function_id"],
            name="post_summary",
            docstring=leaf["docstring"],
            effect_class="unsafe_effectful",
            call_site_line="    return await post_summary(channel='#finance', text=text)",
            args_repr="",
        ),
    ]
    from unify.actor.verification_runtime import CallSite

    stable = passes.stable_block(
        leaf,
        frames=frames,
        call_site=CallSite(
            "weekly_report",
            4,
            "    return await post_summary(channel='#finance', text=text)",
        ),
        root_row=root,
    )

    async def _go():
        wrong = await passes.args_review(
            leaf,
            kwargs={
                "channel": "#general",
                "text": "Weekly totals: 5 orders, 2000 pence",
            },
            stable_block=stable,
            call_site="weekly_report",
            tier0="PASS: input contract satisfied",
        )
        right = await passes.args_review(
            leaf,
            kwargs={
                "channel": "#finance",
                "text": "Weekly totals: 5 orders, 2000 pence",
            },
            stable_block=stable,
            call_site="weekly_report",
            tier0="PASS: input contract satisfied",
        )
        return wrong, right

    wrong, right = asyncio.run(_go())
    assert wrong.verdict == "FAIL", wrong
    assert wrong.fault == "caller", wrong
    assert right.verdict == "PASS", right
    rows = _wait_rows(fm, leaf["function_id"], 2)
    assert sorted(r["kind"] for r in rows) == ["args", "args"]
    assert all(r["prompt_tokens"] > 0 for r in rows)


@pytest.mark.eval
@pytest.mark.llm_call
@_handle_project
def test_post_pass_flags_units_change():
    fm = FunctionManager()
    fm.add_functions(implementations=_TOTAL_MINOR_SRC)
    leaf = fm._get_function_data_by_name(name="total_minor")
    passes = VerifierPasses(
        function_manager=fm,
        goal="Report last week's order total in pence.",
    )
    from unify.actor.verification_runtime import CallSite

    frames = [
        Frame(
            function_id=leaf["function_id"],
            name="total_minor",
            docstring=leaf["docstring"],
            effect_class="safe_noop",
            call_site_line="",
            args_repr="",
        ),
    ]
    stable = passes.stable_block(
        leaf,
        frames=frames,
        call_site=CallSite(None, None, None),
        root_row=leaf,
    )
    rows = [{"amount": 1250}, {"amount": 750}]

    async def _go():
        drifted = await passes.post_probe(
            leaf,
            kwargs={"rows": rows},
            result=20.0,
            stable_block=stable,
            tier0="FAIL: output contract violated: 20.0 is not of type 'integer'",
        )
        fine = await passes.post_probe(
            leaf,
            kwargs={"rows": rows},
            result=2000,
            stable_block=stable,
            tier0="PASS: output contract satisfied",
        )
        return drifted, fine

    drifted, fine = asyncio.run(_go())
    assert drifted.verdict == "FAIL" and drifted.fault == "leaf", drifted
    assert fine.verdict == "PASS", fine


@pytest.mark.eval
@pytest.mark.llm_call
@_handle_project
def test_static_review_flags_contract_mismatch_and_caches_per_hash():
    fm = FunctionManager()
    fm.add_functions(implementations=_PAID_ONLY_BROKEN_SRC)
    row = fm._get_function_data_by_name(name="paid_orders")
    passes = VerifierPasses(function_manager=fm)

    # The persist runs off-loop in an executor; capture its completion
    # future so the test joins the write instead of polling wall-clock.
    persist_writes = []
    _persist = fm.persist_static_review_nowait

    def _capturing_persist(fn_row, record):
        future = _persist(fn_row, record)
        persist_writes.append(future)
        return future

    fm.persist_static_review_nowait = _capturing_persist

    broken = asyncio.run(passes.static_review(row))
    assert broken.verdict == "FAIL" and broken.fault == "leaf", broken

    # Persisted and cached: the second call under the same hash asks no model.
    (write,) = persist_writes
    write.result(timeout=300)  # a lost write raises its real failure here
    persisted = fm._get_function_data_by_name(name="paid_orders")
    assert persisted["static_review"]["verdict"] == "FAIL"
    assert persisted["static_review"]["function_hash"] == fm.function_trust_hash(
        persisted,
    )
    passes._judge = None  # any model call would now raise
    again = asyncio.run(passes.static_review(persisted))
    assert again.verdict == "FAIL"

    fm.add_functions(implementations=_PAID_ONLY_SRC, overwrite=True)
    fixed_row = fm._get_function_data_by_name(name="paid_orders")
    assert fixed_row["static_review"] is None
    fresh = VerifierPasses(function_manager=fm)
    good = asyncio.run(fresh.static_review(fixed_row))
    assert good.verdict == "PASS", good
