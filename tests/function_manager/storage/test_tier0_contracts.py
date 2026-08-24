"""
Tests for tier-0 verification: contracts from type hints, authored
postconditions, the deterministic boundary around stored functions, fixture
capture, and fixture replay on content change.

No LLM is involved anywhere in this file.
"""

import asyncio
import time

import pytest

from tests.helpers import _handle_project
from unify.function_manager.function_manager import FunctionManager
from unify.function_manager.types.verification import FunctionContract
from unify.function_manager.verification.contracts import (
    PostconditionError,
    check_input,
    check_output,
    compile_postcondition,
    contract_from_callable,
    merge_contract,
)
from unify.function_manager.verification.fixtures import (
    FixtureRegressionError,
    add_fixture,
    make_fixture,
    results_equal,
)
from unify.function_manager.verification.tier0 import (
    ContractViolation,
    bind_call_kwargs,
    signature_from_source,
)


@pytest.fixture(autouse=True)
def _verification_enabled(monkeypatch):
    """This module exercises tier-0 verification through the global settings;
    the master switch defaults off, so hold it on for every test here."""
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS.function.verification, "enabled", True)


# ────────────────────────────────────────────────────────────────────────────
# Contracts from hints
# ────────────────────────────────────────────────────────────────────────────


def test_contract_from_hints_covers_params_and_return():
    def total(rows: list[dict], scale: float = 1.0) -> float:
        return 0.0

    contract = contract_from_callable(total)
    assert contract.source == "type_hints"
    assert contract.input_schema["required"] == ["rows"]
    assert contract.input_schema["properties"]["rows"]["type"] == "array"
    assert contract.input_schema["properties"]["scale"]["type"] == "number"
    assert contract.output_schema == {"type": "number"}


def test_contract_from_no_hints_is_none():
    def f(a, b):
        return a

    contract = contract_from_callable(f)
    assert contract.source == "none"
    assert contract.input_schema is None and contract.output_schema is None


def test_contract_partial_hints_cover_only_annotated_parts():
    def f(a: int, b):
        return a

    contract = contract_from_callable(f)
    assert list(contract.input_schema["properties"]) == ["a"]
    assert contract.output_schema is None


# ────────────────────────────────────────────────────────────────────────────
# Postconditions
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "expression",
    [
        "isinstance(result, list)",
        "len(result) >= 1",
        "all(r['amount'] >= 0 for r in result)",
        "result['total'] == sum(kwargs['values'])",
        "sorted(result) == result",
        "result is not None and result.get('ok') is True",
        "max(result) <= 100 if result else True",
    ],
)
def test_postcondition_allowlist_accepts(expression):
    assert compile_postcondition(expression) == expression


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('x')",
        "result.__class__.__name__ == 'dict'",
        "open('/etc/passwd')",
        "eval('1')",
        "(lambda: True)()",
        "os.path.exists(result)",
        "kwargs.__globals__",
        "print(result)",
        "result := 3",
        "import os",
        "",
    ],
)
def test_postcondition_allowlist_rejects(expression):
    with pytest.raises(PostconditionError):
        compile_postcondition(expression)


def test_check_output_evaluates_postconditions_and_schema():
    contract = FunctionContract(
        output_schema={"type": "object"},
        postconditions=["result['n'] == len(kwargs['items'])"],
        source="librarian",
    )
    assert check_output(contract, result={"n": 2}, kwargs={"items": [1, 2]}) is None
    assert "postcondition" in check_output(
        contract,
        result={"n": 3},
        kwargs={"items": [1, 2]},
    )
    assert "output contract violated" in check_output(
        contract,
        result=[1],
        kwargs={"items": []},
    )
    assert "raised" in check_output(contract, result={}, kwargs={"items": []})


def test_check_input_reports_schema_errors():
    contract = FunctionContract(
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
        },
        source="type_hints",
    )
    assert check_input(contract, {"a": 1}) is None
    assert "required" in check_input(contract, {})
    assert "'x' is not of type 'integer'" in check_input(contract, {"a": "x"})
    assert check_input(None, {"anything": 1}) is None


def test_merge_contract_keeps_hint_schemas_and_adds_postconditions():
    hinted = FunctionContract(output_schema={"type": "integer"}, source="type_hints")
    merged = merge_contract(hinted, {"postconditions": ["result >= 0"]})
    assert merged.output_schema == {"type": "integer"}
    assert merged.postconditions == ["result >= 0"]
    assert merged.source == "librarian"
    with pytest.raises(PostconditionError):
        merge_contract(hinted, {"postconditions": ["__import__('os')"]})


# ────────────────────────────────────────────────────────────────────────────
# Fixtures (pure)
# ────────────────────────────────────────────────────────────────────────────


def test_make_fixture_respects_size_cap_and_serialisability():
    assert make_fixture(args={"a": 1}, result=[1, 2], max_bytes=8192) is not None
    assert make_fixture(args={"a": 1}, result="x" * 9000, max_bytes=8192) is None
    assert (
        make_fixture(args={"a": object()}, result=1, max_bytes=8192) is not None
    )  # default=str
    fixture = make_fixture(args={"b": 2, "a": 1}, result={"z": 1}, max_bytes=8192)
    assert (
        fixture.args_signature
        == make_fixture(
            args={"a": 1, "b": 2},
            result=None,
            max_bytes=8192,
        ).args_signature
    )


def test_add_fixture_dedupes_and_evicts_oldest():
    stored: list = []
    for i in range(7):
        stored = add_fixture(
            stored,
            make_fixture(args={"i": i}, result=i, max_bytes=8192),
            cap=5,
        )
    assert [item["args"]["i"] for item in stored] == [2, 3, 4, 5, 6]
    stored = add_fixture(
        stored,
        make_fixture(args={"i": 4}, result=44, max_bytes=8192),
        cap=5,
    )
    assert [item["args"]["i"] for item in stored] == [2, 3, 5, 6, 4]
    assert stored[-1]["result"] == 44


def test_results_equal_is_canonical():
    assert results_equal({"a": 1, "b": [1, 2]}, {"b": [1, 2], "a": 1})
    assert not results_equal({"a": 1}, {"a": 2})


def test_signature_from_source_and_binding():
    sig = signature_from_source("def f(a, b=2, *, c):\n    return a\n")
    assert list(sig.parameters) == ["a", "b", "c"]
    assert bind_call_kwargs(sig, (1,), {"c": 3}) == {"a": 1, "c": 3}
    assert bind_call_kwargs(None, (1,), {}) is None
    assert bind_call_kwargs(None, (), {"a": 1}) == {"a": 1}


# ────────────────────────────────────────────────────────────────────────────
# Storage + boundary
# ────────────────────────────────────────────────────────────────────────────

_ADD = "def add(a: int, b: int) -> int:\n    return a + b\n"
_LIES = "def lies(a: int) -> int:\n    return 'not an int'\n"
_TOTAL = (
    "def total(values: list[int]) -> dict:\n"
    "    return {'n': len(values), 'sum': sum(values)}\n"
)


def _wait_for(predicate, *, timeout: float = 300.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError("condition not met in time")


@_handle_project
def test_add_functions_persists_postconditions_and_rejects_bad_ones():
    fm = FunctionManager()
    fm.add_functions(
        implementations=_TOTAL,
        contracts={
            "total": {"postconditions": ["result['n'] == len(kwargs['values'])"]},
        },
    )
    row = fm._get_function_data_by_name(name="total")
    assert row["contract"]["postconditions"] == ["result['n'] == len(kwargs['values'])"]
    assert row["contract"]["source"] == "librarian"
    assert row["contract"]["output_schema"]["type"] == "object"

    with pytest.raises(ValueError, match="__import__"):
        fm.add_functions(
            implementations=_ADD,
            contracts={"add": {"postconditions": ["__import__('os')"]}},
        )
    # Postconditions survive an overwrite that does not replace them.
    fm.add_functions(implementations=_TOTAL.replace("'sum'", "'total'"), overwrite=True)
    row = fm._get_function_data_by_name(name="total")
    assert row["contract"]["postconditions"] == ["result['n'] == len(kwargs['values'])"]


@_handle_project
def test_namespace_call_runs_tier0_and_captures_fixture():
    fm = FunctionManager()
    fm.add_functions(implementations=[_ADD, _LIES])
    namespace: dict = {}
    fm.filter_functions(
        filter="name in ['add', 'lies']",
        _return_callable=True,
        _namespace=namespace,
    )
    add = namespace["add"]
    assert add(2, 3) == 5
    assert add(a=4, b=5) == 9

    with pytest.raises(ContractViolation) as caller_fault:
        add(a="x", b=1)
    assert caller_fault.value.verdict.fault == "caller"

    with pytest.raises(ContractViolation) as leaf_fault:
        namespace["lies"](a=1)
    assert leaf_fault.value.verdict.fault == "leaf"

    add_id = fm._get_function_data_by_name(name="add")["function_id"]
    rows = _wait_for(lambda: fm.list_verifications(function_id=add_id) or None)
    verdicts = sorted((r["kind"], r["verdict"]) for r in rows)
    assert verdicts == [("tier0", "FAIL"), ("tier0", "PASS"), ("tier0", "PASS")]
    assert [r["fault"] for r in rows if r["verdict"] == "FAIL"] == ["caller"]

    row = _wait_for(
        lambda: (lambda r: r if len(r.get("fixtures") or []) == 2 else None)(
            fm._get_function_data_by_name(name="add"),
        ),
    )
    captured = {tuple(sorted(f["args"].items())): f["result"] for f in row["fixtures"]}
    assert captured == {(("a", 2), ("b", 3)): 5, (("a", 4), ("b", 5)): 9}

    lies_id = fm._get_function_data_by_name(name="lies")["function_id"]
    lies_rows = _wait_for(lambda: fm.list_verifications(function_id=lies_id) or None)
    assert [(r["verdict"], r["fault"]) for r in lies_rows] == [("FAIL", "leaf")]


@_handle_project
def test_fixture_capture_caps_distinct_signatures():
    fm = FunctionManager()
    fm.add_functions(implementations=_ADD)
    namespace: dict = {}
    fm.filter_functions(
        filter="name == 'add'",
        _return_callable=True,
        _namespace=namespace,
    )
    for i in range(8):
        namespace["add"](a=i, b=1)
    row = _wait_for(
        lambda: (lambda r: r if len(r.get("fixtures") or []) >= 5 else None)(
            fm._get_function_data_by_name(name="add"),
        ),
    )
    time.sleep(0.5)
    row = fm._get_function_data_by_name(name="add")
    assert len(row["fixtures"]) == fm.verification_settings.max_fixtures_per_function


@_handle_project
def test_execute_function_runs_tier0():
    fm = FunctionManager()
    fm.add_functions(implementations=[_ADD, _LIES])

    async def _go():
        out = await fm.execute_function(
            function_name="add",
            call_kwargs={"a": 1, "b": 2},
        )
        assert out["result"] == 3
        with pytest.raises(ContractViolation):
            await fm.execute_function(
                function_name="add",
                call_kwargs={"a": "x", "b": 2},
            )
        with pytest.raises(ContractViolation):
            await fm.execute_function(function_name="lies", call_kwargs={"a": 1})

    asyncio.run(_go())
    add_id = fm._get_function_data_by_name(name="add")["function_id"]
    rows = _wait_for(
        lambda: (lambda r: r if len(r) >= 2 else None)(
            fm.list_verifications(function_id=add_id),
        ),
    )
    assert sorted(r["verdict"] for r in rows) == ["FAIL", "PASS"]


@_handle_project
def test_authored_fixtures_only_for_pure_functions():
    fm = FunctionManager()
    fm.add_functions(
        implementations=_ADD,
        fixtures={"add": [{"args": {"a": 1, "b": 2}, "result": 3}]},
    )
    row = fm._get_function_data_by_name(name="add")
    assert [f["result"] for f in row["fixtures"]] == [3]
    with pytest.raises(ValueError, match="safe_noop"):
        fm.add_functions(
            implementations="async def ping(q: str) -> str:\n    return await primitives.contacts.ask(q)\n",
            fixtures={"ping": [{"args": {"q": "x"}, "result": "y"}]},
        )


@_handle_project
def test_fixture_replay_on_overwrite_seeds_or_rejects():
    fm = FunctionManager()
    fm.add_functions(
        implementations=_ADD,
        fixtures={
            "add": [
                {"args": {"a": 1, "b": 2}, "result": 3},
                {"args": {"a": 5, "b": 5}, "result": 10},
            ],
        },
    )
    # A behaviour-preserving rewrite passes replay and seeds the ledger.
    fm.add_functions(
        implementations="def add(a: int, b: int) -> int:\n    return b + a\n",
        overwrite=True,
    )
    row = fm._get_function_data_by_name(name="add")
    assert row["ledger"]["passes"] == {"tier0": 2}
    assert len(row["ledger"]["distinct_args_signatures"]) == 2
    assert row["verified_hash"] == fm.function_trust_hash(row)
    assert row["static_review"] is None
    assert row["verify"] is True  # static review still outstanding

    # A behaviour-changing rewrite is rejected with the failing fixture.
    with pytest.raises(FixtureRegressionError) as info:
        fm.add_functions(
            implementations="def add(a: int, b: int) -> int:\n    return a - b\n",
            overwrite=True,
        )
    assert info.value.fixture.args == {"a": 1, "b": 2}
    assert info.value.actual == -1
    row = fm._get_function_data_by_name(name="add")
    assert "return b + a" in row["implementation"]
