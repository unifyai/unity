"""
Tests for the verification ledger schema: effect-class detection, the trust
hash, the trust policy, and backfill of rows stored before the ledger existed.

No LLM is involved anywhere in this file.
"""

import logging

import pytest
import unisdk

from tests.helpers import _handle_project
from unify.function_manager.function_manager import FunctionManager
from unify.function_manager.primitives.registry import collect_primitives
from unify.function_manager.settings import VerificationSettings
from unify.function_manager.types.verification import (
    SideEffectClass,
    StaticReviewRecord,
    VerdictKind,
    VerificationRow,
    VerificationSummary,
)
from unify.function_manager.verification.classify import (
    PRIMITIVE_EFFECT_CLASSES,
    classify_source,
    effective_class,
)
from unify.function_manager.verification.ledger import (
    apply_verdict,
    args_signature,
    function_trust_hash,
)
from unify.function_manager.verification.policy import derive_verify, spot_check_rate


@pytest.fixture(autouse=True)
def _verification_enabled(monkeypatch):
    """This module exercises the trust machinery through the global settings
    too (``derive_verify_for_row``); the master switch defaults off, so hold
    it on for every test here. The off state has its own explicit test."""
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS.function.verification, "enabled", True)


# ────────────────────────────────────────────────────────────────────────────
# Primitive table completeness
# ────────────────────────────────────────────────────────────────────────────


def test_every_primitive_has_an_explicit_effect_class():
    """Every static primitive row must be classified explicitly."""
    names = set(collect_primitives())
    missing = sorted(names - set(PRIMITIVE_EFFECT_CLASSES))
    assert not missing, f"Unclassified primitives: {missing}"
    stale = sorted(set(PRIMITIVE_EFFECT_CLASSES) - names)
    assert not stale, f"Table entries for primitives that no longer exist: {stale}"


def test_every_comms_send_is_unsafe():
    for name, klass in PRIMITIVE_EFFECT_CLASSES.items():
        if name.startswith("primitives.comms.send_"):
            assert klass is SideEffectClass.unsafe_effectful, name


# ────────────────────────────────────────────────────────────────────────────
# Classification corpus
# ────────────────────────────────────────────────────────────────────────────

_PURE = "def add(a: int, b: int) -> int:\n    return a + b\n"
_READ = (
    "async def lookup(q: str) -> str:\n    return await primitives.contacts.ask(q)\n"
)
_SEND = (
    "async def notify(to: str, body: str) -> None:\n"
    "    await primitives.comms.send_email(to=to, subject='hi', body=body)\n"
)
_UPSERT = (
    "async def save(rows: list) -> None:\n"
    "    await primitives.data.update_rows('T', rows)\n"
)
_DESKTOP = "async def press():\n    await primitives.computer.click(10, 20)\n"
_THIRD_PARTY_HTTP = (
    "def fetch(url: str):\n    import requests\n    return requests.get(url)\n"
)
_THIRD_PARTY_PURE = (
    "def mean(xs):\n    import numpy as np\n    return float(np.mean(xs))\n"
)
_THIRD_PARTY_OTHER = "def s3(x):\n    import boto3\n    return x\n"
_OPEN_WRITE = "def dump(p):\n    with open(p, 'w') as fh:\n        fh.write('x')\n"
_UNKNOWN_PRIMITIVE = "async def odd():\n    await primitives.made_up.thing()\n"
_LEGACY_NAMESPACE = (
    "async def shot():\n    return await computer_primitives.get_screenshot()\n"
)
_CALLS_DEP = "def outer(x):\n    return inner(x)\n"


@pytest.mark.parametrize(
    "source, expected_class, expected_source",
    [
        (_PURE, SideEffectClass.safe_noop, "pure"),
        (_READ, SideEffectClass.read_only, "primitives"),
        (_SEND, SideEffectClass.unsafe_effectful, "primitives"),
        (_UPSERT, SideEffectClass.idempotent_effectful, "primitives"),
        (_DESKTOP, SideEffectClass.unsafe_effectful, "primitives"),
        (_THIRD_PARTY_HTTP, SideEffectClass.unsafe_effectful, "inferred_third_party"),
        (_THIRD_PARTY_PURE, SideEffectClass.safe_noop, "pure"),
        (_THIRD_PARTY_OTHER, SideEffectClass.read_only, "inferred_third_party"),
        (_OPEN_WRITE, SideEffectClass.unsafe_effectful, "inferred_third_party"),
        (_LEGACY_NAMESPACE, SideEffectClass.read_only, "primitives"),
    ],
)
def test_classification_corpus(source, expected_class, expected_source):
    result = classify_source(source)
    assert result.detected is expected_class
    assert result.source == expected_source


def test_unclassified_primitive_defaults_to_unsafe_and_logs_error(caplog):
    # unify loggers do not propagate to root, so attach the capture handler directly.
    classify_logger = logging.getLogger("unify.function_manager.verification.classify")
    classify_logger.addHandler(caplog.handler)
    caplog.set_level(logging.ERROR, logger=classify_logger.name)
    try:
        result = classify_source(_UNKNOWN_PRIMITIVE)
    finally:
        classify_logger.removeHandler(caplog.handler)
    assert result.detected is SideEffectClass.unsafe_effectful
    assert result.unclassified_primitives == ["primitives.made_up.thing"]
    errors = [rec for rec in caplog.records if rec.levelno == logging.ERROR]
    assert any("primitives.made_up.thing" in rec.getMessage() for rec in errors)


def test_dependency_class_raises_the_bound():
    def _dep(name):
        return SideEffectClass.unsafe_effectful if name == "inner" else None

    result = classify_source(
        _CALLS_DEP,
        known_function_names={"inner"},
        dependency_class=_dep,
    )
    assert result.dependencies == ["inner"]
    assert result.detected is SideEffectClass.unsafe_effectful
    assert result.source == "primitives"


def test_unresolved_dependency_is_unsafe():
    result = classify_source(
        _CALLS_DEP,
        known_function_names={"inner"},
        dependency_class=lambda name: None,
    )
    assert result.detected is SideEffectClass.unsafe_effectful


def test_integration_primitive_classified_from_action_class():
    rows = {
        "primitives.integrations.gmail.send_email": {
            "name": "primitives.integrations.gmail.send_email",
            "is_primitive": True,
            "metadata": {"action_class": "write"},
        },
    }
    src = "async def go():\n    await primitives.integrations.gmail.send_email(x=1)\n"
    assert (
        classify_source(src, primitive_rows=rows).detected
        is SideEffectClass.idempotent_effectful
    )
    assert classify_source(src).detected is SideEffectClass.unsafe_effectful


def test_effective_class_bounds():
    detected = SideEffectClass.read_only
    assert (
        effective_class(detected=detected, source="primitives", confirmed=None)
        is SideEffectClass.read_only
    )
    assert (
        effective_class(
            detected=detected,
            source="inferred_third_party",
            confirmed=None,
        )
        is SideEffectClass.unsafe_effectful
    )
    # Raising is free; lowering stops at the detected bound.
    assert (
        effective_class(
            detected=detected,
            source="primitives",
            confirmed=SideEffectClass.unsafe_effectful,
        )
        is SideEffectClass.unsafe_effectful
    )
    assert (
        effective_class(
            detected=detected,
            source="primitives",
            confirmed=SideEffectClass.safe_noop,
        )
        is SideEffectClass.read_only
    )


# ────────────────────────────────────────────────────────────────────────────
# Trust hash
# ────────────────────────────────────────────────────────────────────────────


def _hash(row, rows=None, venvs=None):
    rows = rows or {}
    venvs = venvs or {}
    return function_trust_hash(
        row,
        resolve_row=lambda name: rows.get(name),
        resolve_venv=lambda venv_id: venvs.get(venv_id),
    )


def test_trust_hash_changes_on_each_component():
    inner = {
        "name": "inner",
        "implementation": _PURE,
        "depends_on": [],
        "language": "python",
    }
    outer = {
        "name": "outer",
        "implementation": _CALLS_DEP,
        "depends_on": ["inner"],
        "language": "python",
        "venv_id": 7,
    }
    venvs = {7: {"venv_id": 7, "venv": "[project]\nname='a'\n"}}
    base = _hash(outer, {"inner": inner}, venvs)

    assert _hash(outer, {"inner": inner}, venvs) == base
    assert (
        _hash(dict(outer, implementation=_CALLS_DEP + "\n"), {"inner": inner}, venvs)
        == base
    )
    assert (
        _hash(
            dict(outer, implementation=_CALLS_DEP.replace("inner(x)", "inner(x + 1)")),
            {"inner": inner},
            venvs,
        )
        != base
    )
    changed_inner = dict(inner, implementation=_PURE.replace("a + b", "a - b"))
    assert _hash(outer, {"inner": changed_inner}, venvs) != base
    assert _hash(dict(outer, venv_id=8), {"inner": inner}, {8: venvs[7]}) != base
    assert (
        _hash(
            outer,
            {"inner": inner},
            {7: {"venv_id": 7, "venv": "[project]\nname='b'\n"}},
        )
        != base
    )
    assert _hash(dict(outer, language="bash"), {"inner": inner}, venvs) != base
    assert _hash(outer, {}, venvs) != base


def test_trust_hash_survives_dependency_cycles():
    a = {
        "name": "a",
        "implementation": "def a():\n    return b()\n",
        "depends_on": ["b"],
    }
    b = {
        "name": "b",
        "implementation": "def b():\n    return a()\n",
        "depends_on": ["a"],
    }
    rows = {"a": a, "b": b}
    assert _hash(a, rows) == _hash(a, rows)
    assert _hash(a, rows) != _hash(b, rows)


def test_args_signature_is_order_independent():
    assert args_signature({"a": 1, "b": [1, 2]}) == args_signature(
        {"b": [1, 2], "a": 1},
    )
    assert args_signature({"a": 1}) != args_signature({"a": 2})


# ────────────────────────────────────────────────────────────────────────────
# Trust policy
# ────────────────────────────────────────────────────────────────────────────

# These tests exercise the trust ladder itself, so the master switch is on;
# its off state has its own test below.
_SETTINGS = VerificationSettings(enabled=True)
_HASH = "deadbeef"


def _trusted_row(
    klass: SideEffectClass,
    *,
    passes: int,
    inputs: int,
    source="primitives",
):
    kinds = (
        {VerdictKind.tier0: passes}
        if klass is SideEffectClass.safe_noop
        else {VerdictKind.args: passes, VerdictKind.post: passes}
    )
    return {
        "side_effect_class": klass.value,
        "class_source": source,
        "verified_hash": _HASH,
        "static_review": StaticReviewRecord(
            verdict="PASS",
            function_hash=_HASH,
        ).model_dump(mode="json"),
        "ledger": VerificationSummary(
            passes={str(k): v for k, v in kinds.items()},
            distinct_args_signatures=[f"sig{i}" for i in range(inputs)],
        ).model_dump(mode="json"),
        "verification_policy": {},
    }


@pytest.mark.parametrize(
    "klass, needed_passes, needed_inputs",
    [
        (SideEffectClass.safe_noop, 1, 1),
        (SideEffectClass.read_only, 3, 2),
        (SideEffectClass.idempotent_effectful, 3, 2),
        (SideEffectClass.unsafe_effectful, 5, 3),
    ],
)
def test_derive_verify_truth_table(klass, needed_passes, needed_inputs):
    ok = _trusted_row(klass, passes=needed_passes, inputs=needed_inputs)
    assert derive_verify(ok, settings=_SETTINGS, current_hash=_HASH) is False
    short = _trusted_row(klass, passes=needed_passes - 1, inputs=needed_inputs)
    assert derive_verify(short, settings=_SETTINGS, current_hash=_HASH) is True
    few_inputs = _trusted_row(klass, passes=needed_passes, inputs=needed_inputs - 1)
    assert derive_verify(few_inputs, settings=_SETTINGS, current_hash=_HASH) is True


def test_derive_verify_requires_current_hash_static_pass_and_no_fails():
    row = _trusted_row(SideEffectClass.read_only, passes=3, inputs=2)
    assert derive_verify(row, settings=_SETTINGS, current_hash="other") is True
    stale_static = dict(
        row,
        static_review=StaticReviewRecord(
            verdict="PASS",
            function_hash="old",
        ).model_dump(mode="json"),
    )
    assert derive_verify(stale_static, settings=_SETTINGS, current_hash=_HASH) is True
    unsure_static = dict(
        row,
        static_review=StaticReviewRecord(
            verdict="UNSURE",
            function_hash=_HASH,
        ).model_dump(mode="json"),
    )
    assert derive_verify(unsure_static, settings=_SETTINGS, current_hash=_HASH) is True
    failed = dict(row, ledger=dict(row["ledger"], fails=1))
    assert derive_verify(failed, settings=_SETTINGS, current_hash=_HASH) is True


def test_master_switch_off_disables_the_whole_subsystem():
    """Disabled (the default), every row runs trusted: no pin, hash, ledger,
    or static-review state can demand verification, and spot checks never
    sample — the subsystem is absent, not lenient."""
    off = VerificationSettings()
    assert off.enabled is False
    virgin = {
        "side_effect_class": SideEffectClass.unsafe_effectful.value,
        "class_source": "primitives",
        "verification_policy": {},
    }
    assert derive_verify(virgin, settings=off, current_hash=_HASH) is False
    pinned = dict(virgin, verification_policy={"always_verify": True})
    assert derive_verify(pinned, settings=off, current_hash=_HASH) is False
    assert spot_check_rate(pinned, off) == 0.0
    # The same rows verify again the moment the switch turns back on.
    assert derive_verify(virgin, settings=_SETTINGS, current_hash=_HASH) is True
    assert derive_verify(pinned, settings=_SETTINGS, current_hash=_HASH) is True


def test_policy_can_only_raise_the_bar():
    row = _trusted_row(SideEffectClass.read_only, passes=3, inputs=2)
    assert (
        derive_verify(
            dict(row, verification_policy={"required_passes": 1}),
            settings=_SETTINGS,
            current_hash=_HASH,
        )
        is False
    )
    assert (
        derive_verify(
            dict(row, verification_policy={"required_passes": 4}),
            settings=_SETTINGS,
            current_hash=_HASH,
        )
        is True
    )
    assert (
        derive_verify(
            dict(row, verification_policy={"min_distinct_inputs": 3}),
            settings=_SETTINGS,
            current_hash=_HASH,
        )
        is True
    )
    assert (
        derive_verify(
            dict(row, verification_policy={"always_verify": True}),
            settings=_SETTINGS,
            current_hash=_HASH,
        )
        is True
    )


def test_unconfirmed_third_party_inference_is_held_to_unsafe_bar():
    row = _trusted_row(
        SideEffectClass.read_only,
        passes=3,
        inputs=2,
        source="inferred_third_party",
    )
    assert derive_verify(row, settings=_SETTINGS, current_hash=_HASH) is True
    row = _trusted_row(
        SideEffectClass.read_only,
        passes=5,
        inputs=3,
        source="inferred_third_party",
    )
    assert derive_verify(row, settings=_SETTINGS, current_hash=_HASH) is False


def test_apply_verdict_folds_rows():
    summary = VerificationSummary()
    summary = apply_verdict(
        summary,
        VerificationRow(
            function_id=1,
            kind=VerdictKind.args,
            verdict="PASS",
            args_signature="s1",
        ),
    )
    summary = apply_verdict(
        summary,
        VerificationRow(
            function_id=1,
            kind=VerdictKind.post,
            verdict="PASS",
            args_signature="s1",
        ),
    )
    summary = apply_verdict(
        summary,
        VerificationRow(
            function_id=1,
            kind=VerdictKind.post,
            verdict="UNSURE",
            args_signature="s2",
        ),
    )
    summary = apply_verdict(
        summary,
        VerificationRow(
            function_id=1,
            kind=VerdictKind.spot_check,
            verdict="FAIL",
            fault="leaf",
            args_signature="s3",
        ),
    )
    assert summary.pass_count(VerdictKind.args) == 1
    assert summary.pass_count(VerdictKind.post) == 1
    assert summary.unsure == 1
    assert summary.fails == 1
    assert summary.spot_checks == 1
    assert summary.distinct_args_signatures == ["s1", "s2", "s3"]
    assert summary.last_verdict_at is not None


def test_fail_verdict_requires_fault():
    with pytest.raises(ValueError):
        VerificationRow(function_id=1, kind=VerdictKind.post, verdict="FAIL")


# ────────────────────────────────────────────────────────────────────────────
# Storage: add_functions, overwrite, backfill, ledger rows
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_add_functions_persists_classification_and_starts_on_ramp():
    fm = FunctionManager()
    fm.add_functions(implementations=[_PURE, _SEND])
    rows = {
        row["name"]: row
        for row in fm.filter_functions(filter="name in ['add', 'notify']")
    }

    add = rows["add"]
    assert add["side_effect_class"] == "safe_noop"
    assert add["side_effect_class_detected"] == "safe_noop"
    assert add["class_source"] == "pure"
    assert add["verify"] is True
    assert add["contract"]["source"] == "type_hints"
    assert add["contract"]["input_schema"]["required"] == ["a", "b"]
    assert add["contract"]["output_schema"] == {"type": "integer"}
    # Ledger internals are runtime state and stay out of catalogue reads.
    for hidden in ("fixtures", "ledger", "static_review", "verified_hash"):
        assert hidden not in add

    notify = rows["notify"]
    assert notify["side_effect_class"] == "unsafe_effectful"
    assert notify["class_source"] == "primitives"
    assert notify["verify"] is True

    full = fm._get_function_data_by_name(name="add")
    assert full["verified_hash"] is None
    assert full["static_review"] is None
    assert full["ledger"] == VerificationSummary().model_dump(mode="json")
    assert full["fixtures"] == []


@_handle_project
def test_overwrite_reclassifies_and_keeps_policy():
    fm = FunctionManager()
    fm.add_functions(implementations=_PURE)
    fm._persist_verification_fields(
        function_id=fm._get_function_data_by_name(name="add")["function_id"],
        fields={"verification_policy": {"required_passes": 9}},
    )
    fm.add_functions(
        implementations=(
            "async def add(a: int, b: int) -> int:\n"
            "    await primitives.comms.send_sms(to='x', body='y')\n"
            "    return a + b\n"
        ),
        overwrite=True,
    )
    row = fm._get_function_data_by_name(name="add")
    assert row["side_effect_class"] == "unsafe_effectful"
    assert row["verification_policy"] == {"required_passes": 9}
    assert row["verify"] is True


@_handle_project
def test_librarian_confirmation_survives_overwrite_only_within_bounds():
    fm = FunctionManager()
    fm.add_functions(implementations=_READ)
    fid = fm._get_function_data_by_name(name="lookup")["function_id"]
    fm._persist_verification_fields(
        function_id=fid,
        fields={
            "side_effect_class": "idempotent_effectful",
            "class_source": "librarian",
            "class_rationale": "writes a cache",
        },
    )
    # Same detected bound: the confirmation stands.
    fm.add_functions(
        implementations=_READ.replace("ask(q)", "ask(q + '?')"),
        overwrite=True,
    )
    row = fm._get_function_data_by_name(name="lookup")
    assert row["side_effect_class"] == "idempotent_effectful"
    assert row["class_source"] == "librarian"
    assert row["class_rationale"] == "writes a cache"
    # Detected bound rises above the confirmation: detection wins.
    fm.add_functions(
        implementations=(
            "async def lookup(q: str) -> str:\n"
            "    await primitives.comms.send_sms(to=q, body='x')\n"
            "    return q\n"
        ),
        overwrite=True,
    )
    row = fm._get_function_data_by_name(name="lookup")
    assert row["side_effect_class"] == "unsafe_effectful"
    assert row["class_source"] == "primitives"
    assert row["class_rationale"] is None


@_handle_project
def test_backfill_classifies_rows_stored_before_the_ledger_once():
    fm = FunctionManager()
    unisdk.create_logs(
        context=fm._compositional_ctx,
        entries=[
            {
                "name": "legacy",
                "language": "python",
                "argspec": "(x: int) -> int",
                "docstring": "",
                "implementation": "def legacy(x: int) -> int:\n    return x\n",
                "depends_on": [],
                "embedding_text": "legacy",
                "guidance_ids": [],
            },
        ],
    )
    raw = unisdk.get_logs(context=fm._compositional_ctx, filter="name == 'legacy'")[
        0
    ].entries
    assert raw.get("side_effect_class") is None

    row = fm._get_function_data_by_name(name="legacy")
    assert row["side_effect_class"] == "safe_noop"
    assert row["class_source"] == "pure"
    assert row["verify"] is True
    assert row["contract"]["source"] == "type_hints"

    persisted = unisdk.get_logs(
        context=fm._compositional_ctx,
        filter="name == 'legacy'",
    )[0].entries
    assert persisted["side_effect_class"] == "safe_noop"

    writes = []
    original = fm._persist_verification_fields
    fm._persist_verification_fields = lambda **kw: writes.append(kw) or original(**kw)
    fm.list_functions()
    fm.filter_functions(filter="name == 'legacy'")
    assert writes == []


@_handle_project
def test_record_and_list_verifications():
    fm = FunctionManager()
    fm.add_functions(implementations=_PURE)
    fid = fm._get_function_data_by_name(name="add")["function_id"]
    fm.record_verification(
        VerificationRow(
            function_id=fid,
            function_hash="h1",
            kind=VerdictKind.tier0,
            verdict="PASS",
            args_signature="s1",
        ),
    )
    fm.record_verification(
        VerificationRow(
            function_id=fid,
            function_hash="h1",
            kind=VerdictKind.post,
            verdict="FAIL",
            fault="leaf",
            reason="x" * 5000,
        ),
    )
    fm.record_verification(
        VerificationRow(
            function_id=fid,
            function_hash="h2",
            kind=VerdictKind.args,
            verdict="UNSURE",
        ),
    )
    rows = fm.list_verifications(function_id=fid)
    assert [r["kind"] for r in rows] == ["tier0", "post", "args"]
    assert len(rows[1]["reason"]) == 2000
    assert all(r["created_at"] for r in rows)
    assert [
        r["kind"] for r in fm.list_verifications(function_id=fid, function_hash="h1")
    ] == ["tier0", "post"]


@_handle_project
def test_trust_hash_through_function_manager_tracks_dependency_content():
    fm = FunctionManager()
    fm.add_functions(
        implementations=[
            "def inner(x):\n    return x\n",
            "def outer(x):\n    return inner(x)\n",
        ],
    )
    outer = fm._get_function_data_by_name(name="outer")
    before = fm.function_trust_hash(outer)
    fm.add_functions(
        implementations="def inner(x):\n    return x + 1\n",
        overwrite=True,
    )
    outer = fm._get_function_data_by_name(name="outer")
    assert fm.function_trust_hash(outer) != before
    assert fm.derive_verify_for_row(outer) is True
