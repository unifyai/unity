"""Midland Heart KPI function integration tests.

Verifies that the pre-registered KPI metric functions can be loaded into
a FunctionManager, that inter-function dependency edges are correctly
recorded, and that a representative metric can be executed against stub
data_primitives producing the standardised result dict.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pytest
import unify

from tests.helpers import _handle_project
from unity.function_manager.execution_env import create_base_globals
from unity.function_manager.function_manager import FunctionManager

_MH_FUNCTIONS_DIR = (
    Path(__file__).resolve().parents[4]
    / "unity"
    / "customization"
    / "clients"
    / "midland_heart"
    / "functions"
)


def _extract_standalone_sources(filepath: Path) -> List[str]:
    """Read a ``@custom_function()`` file and return bare function sources.

    Strips the module docstring, imports, and ``@custom_function()``
    decorators so each result is a clean function definition suitable for
    ``FunctionManager.add_functions(implementations=...)``.
    """
    source = filepath.read_text()
    tree = ast.parse(source)

    func_nodes = [
        n
        for n in ast.iter_child_nodes(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not func_nodes:
        return []

    sources: List[str] = []
    for func_node in func_nodes:
        func_node.decorator_list = []
        sources.append(ast.unparse(func_node))
    return sources


def _build_mh_implementations() -> List[str]:
    """Build standalone implementations from the MH functions directory."""
    implementations: List[str] = []
    for py_file in sorted(_MH_FUNCTIONS_DIR.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        implementations.extend(_extract_standalone_sources(py_file))
    return implementations


def _deps_for(fm: FunctionManager, *, function_name: str) -> set[str]:
    logs = unify.get_logs(
        context=fm._compositional_ctx,
        filter=f"name == '{function_name}'",
        limit=1,
    )
    assert logs, f"Missing FunctionManager log row for '{function_name}'"
    deps = logs[0].entries.get("depends_on") or []
    assert isinstance(deps, list)
    return set(d for d in deps if isinstance(d, str))


@_handle_project
def test_add_functions_records_dependency_edges():
    """add_functions should record inter-function dependency edges for KPIs."""
    fm = FunctionManager()

    implementations = _build_mh_implementations()
    results = fm.add_functions(implementations=implementations, overwrite=True)

    assert not any(str(v).startswith("error") for v in results.values())

    # Spot-check dependency edges for a repairs-table KPI.
    deps = _deps_for(fm, function_name="repairs_kpi_jobs_completed")
    assert "discover_repairs_table" in deps or "resolve_group_by" in deps
    assert "build_filter" in deps
    assert "build_metric_result" in deps

    # Spot-check dependency edges for a telematics KPI.
    deps2 = _deps_for(fm, function_name="repairs_kpi_total_distance_travelled")
    assert "discover_telematics_tables" in deps2
    assert "extract_sum" in deps2
    assert "build_metric_result" in deps2


# ────────────────────────────────────────────────────────────────────────────
# Runtime callable execution against stub data_primitives
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeTableDescription:
    columns: List[Any] = field(default_factory=list)


@dataclass
class _FakeColumn:
    name: str
    description: str = ""


class _FakeDataPrimitives:
    """Stub data_primitives with DM method signatures."""

    async def describe_table(
        self,
        context: str,
        **kwargs: Any,
    ) -> _FakeTableDescription:
        return _FakeTableDescription(
            columns=[
                _FakeColumn(name="JobTicketReference"),
                _FakeColumn(name="WorksOrderStatusDescription"),
                _FakeColumn(name="NoAccess"),
                _FakeColumn(name="FollowOn"),
                _FakeColumn(name="FirstTimeFix"),
            ],
        )

    async def list_tables(self, prefix: str = "", **kwargs: Any) -> List[str]:
        if "Telematics" in prefix:
            return [
                "Data/MidlandHeart/Telematics/July",
                "Data/MidlandHeart/Telematics/August",
            ]
        return ["Data/MidlandHeart/Repairs"]

    async def filter(self, context: str, **kwargs: Any) -> List[Dict[str, Any]]:
        return [{"JobTicketReference": "123", "WorksOrderStatusDescription": "Closed"}]

    async def reduce(self, *args: Any, **kwargs: Any) -> Any:
        return 3

    async def plot(self, **kwargs: Any) -> Dict[str, Any]:
        return {"url": "https://fake-plot.png", "error": None}

    async def get_columns(self, context: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "JobTicketReference": {"description": "Ticket ID"},
            "WorksOrderStatusDescription": {"description": "Status"},
        }


@_handle_project
@pytest.mark.asyncio
async def test_kpi_callable_executes_with_stub_data_primitives():
    """A loaded KPI function should execute against stub data_primitives."""
    fm = FunctionManager()
    fm.add_functions(
        implementations=_build_mh_implementations(),
        overwrite=True,
    )

    ns = create_base_globals()
    callables = fm.filter_functions(
        filter="name == 'repairs_kpi_jobs_completed'",
        limit=1,
        _return_callable=True,
        _namespace=ns,
    )
    assert len(callables) == 1

    assert "repairs_kpi_jobs_completed" in ns and callable(
        ns["repairs_kpi_jobs_completed"],
    )
    for dep in [
        "resolve_group_by",
        "build_filter",
        "build_metric_result",
    ]:
        assert dep in ns and callable(ns[dep])

    out = await ns["repairs_kpi_jobs_completed"](
        data_primitives=_FakeDataPrimitives(),
        group_by=None,
        include_plots=False,
    )

    assert isinstance(out, dict)
    assert out.get("metric_name") == "jobs_completed"
    assert out.get("total") == 3.0
