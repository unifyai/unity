"""Colliers Healthcare — financial data extraction eval test.

Runs the full CodeActActor with pre-seeded Colliers functions and guidance
against four real PDF annual accounts, then compares the extracted financial
data against a human-validated ground truth Excel workbook.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.helpers import _handle_project
from tests.actor.state_managers.utils import make_code_act_actor
from tests.customization.colliers_eval_helpers import (
    BASE_LENIENT_FIELDS,
    PDF_FILES,
    GROUND_TRUTH_XLSX,
    compare_results,
    find_output,
    parse_ground_truth,
    print_comparison,
)
from unity.manager_registry import ManagerRegistry

pytestmark = pytest.mark.eval


def _seed_colliers_guidance() -> None:
    from unity.customization.clients.colliers import _COLLIERS_GUIDANCE

    gm = ManagerRegistry.get_guidance_manager()
    for entry in _COLLIERS_GUIDANCE:
        gm.add_guidance(title=entry["title"], content=entry["content"])


def _seed_colliers_functions() -> "FunctionManager":
    from unity.function_manager.function_manager import FunctionManager
    from unity.function_manager.custom_functions import (
        collect_functions_from_directories,
    )

    colliers_fn_dir = (
        Path(__file__).resolve().parents[2]
        / "unity"
        / "customization"
        / "clients"
        / "colliers"
        / "functions"
    )
    fm = FunctionManager()
    source_fns = collect_functions_from_directories([colliers_fn_dir])
    if source_fns:
        fm.sync_custom(source_functions=source_fns, source_venvs={})
    return fm


@pytest.mark.asyncio
@pytest.mark.timeout(900)
@_handle_project
async def test_colliers_ash_financial_extraction(tmp_path: Path):
    """Extract financial data from 4 ASH annual accounts PDFs with
    Colliers pre-seeded functions and guidance, and compare against
    human-validated ground truth."""
    assert len(PDF_FILES) == 4
    assert GROUND_TRUTH_XLSX.exists()

    workspace_dir = tmp_path / "accounts"
    workspace_dir.mkdir()
    for pdf in PDF_FILES:
        shutil.copy(pdf, workspace_dir / pdf.name)

    output_path = tmp_path / "output" / "ASH_Historic_Accounts.xlsx"
    output_path.parent.mkdir()

    _seed_colliers_guidance()
    fm = _seed_colliers_functions()

    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
        function_manager=fm,
    ) as (actor, _primitives, calls):
        handle = await actor.act(
            f"Please extract the financial account data from the PDF files in "
            f"{workspace_dir} and produce the standard HISTORIC ACCOUNTS Excel "
            f"spreadsheet. Save the output to {output_path}. "
            f"There are 4 annual accounts PDFs covering fiscal years ending "
            f"30 April 2022 through 30 April 2025. The property name is "
            f"'Ablegrange Severn Heights'.",
            guidelines=(
                "You are the Colliers Healthcare Valuation Assistant. "
                "Search your guidance and functions for the standard financial "
                "extraction workflow and Excel formatting function."
            ),
            clarification_enabled=False,
        )
        result = await handle.result()

    actual = find_output(tmp_path, output_path)
    assert actual, (
        f"No output found. Actor result:\n{result[:500]}\n"
        f"Files: {list(tmp_path.rglob('*'))}"
    )

    expected = parse_ground_truth()
    matches, mismatches, acceptable = compare_results(
        expected,
        actual,
        lenient_fields=BASE_LENIENT_FIELDS,
    )
    print_comparison("COLLIERS-SEEDED RESULTS", matches, mismatches, acceptable)

    assert len(mismatches) == 0, f"{len(mismatches)} strict failures:\n" + "\n".join(
        f"  {m}" for m in mismatches
    )
    assert (
        len(matches) + len(mismatches) >= 60
    ), f"Only {len(matches)} values found (expected >= 60)."
