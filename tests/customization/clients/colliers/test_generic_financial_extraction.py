"""Generic financial extraction eval test — NO Colliers customization.

Runs the same PDF extraction task but with zero domain-specific
pre-seeding.  The actor has only the base CodeActActor prompt, general
FileManager primitives, and the discovery-first policy (with empty
FunctionManager and GuidanceManager stores).

Establishes the general-purpose baseline that domain seeding builds on.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.helpers import _handle_project
from tests.actor.state_managers.utils import make_code_act_actor
from tests.customization.clients.colliers.eval_helpers import (
    BASE_LENIENT_FIELDS,
    EXTRA_LENIENT_FIELDS,
    PDF_FILES,
    TEMPLATE_XLSX,
    compare_results,
    find_output,
    merge_lenient_fields,
    parse_ground_truth,
    print_comparison,
    stable_workspace,
)

pytestmark = pytest.mark.eval

LENIENT_FIELDS = merge_lenient_fields(BASE_LENIENT_FIELDS, EXTRA_LENIENT_FIELDS)


@pytest.mark.asyncio
@pytest.mark.timeout(900)
@_handle_project
async def test_generic_ash_financial_extraction(tmp_path: Path):
    """Extract financial data with NO domain-specific customization.

    Full default CodeActActor with discovery-first policy, empty stores,
    and a detailed human request with an empty Excel template.
    """
    assert len(PDF_FILES) == 4
    assert TEMPLATE_XLSX.exists()

    workspace_dir, output_path = stable_workspace("generic_ash_extraction")
    for pdf in PDF_FILES:
        shutil.copy(pdf, workspace_dir / pdf.name)

    template_dest = workspace_dir / "HISTORIC_ACCOUNTS_TEMPLATE.xlsx"
    shutil.copy(TEMPLATE_XLSX, template_dest)

    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
    ) as (actor, _primitives, calls):
        handle = await actor.act(
            f"I need your help extracting financial data from annual accounts.\n\n"
            f"In the folder {workspace_dir} there are 4 PDF files — these are "
            f"the signed annual accounts for a care home called 'Ablegrange "
            f"Severn Heights' for fiscal years ending 30 April 2022, 2023, "
            f"2024, and 2025 (one PDF per year).\n\n"
            f"Each PDF contains a Profit and Loss account and detailed Notes "
            f"to the P&L that break down the expenses line by line. The P&L "
            f"shows the high-level figures (Turnover, Cost of Sales, "
            f"Administration Expenses, Interest, Profit before Tax, Taxation). "
            f"The Notes page lists every individual expense (wages, heating, "
            f"repairs, insurance, etc.) with exact amounts.\n\n"
            f"I've attached an empty Excel template at {template_dest} that "
            f"shows the exact format I need. Please open this template first "
            f"to see the row labels and structure. Then go through each PDF "
            f"and fill in every value you can find. The PDFs are scans of "
            f"printed documents, not digital-native, so you'll need to view "
            f"them visually rather than trying to extract text.\n\n"
            f"For each value you extract, find the exact number from the PDF "
            f"page. Map each line item from the PDF's Notes to the closest "
            f"matching row in the template. Some PDF labels won't match the "
            f"template exactly — use your best judgment.\n\n"
            f"Save the completed spreadsheet to {output_path}. Each fiscal "
            f"year should be a column (Year 1 = FYE 2022, Year 4 = FYE 2025).",
            clarification_enabled=False,
        )
        result = await handle.result()

    search_dirs = [workspace_dir.parent, tmp_path]
    actual = find_output(search_dirs, output_path)

    expected = parse_ground_truth()
    matches, mismatches, acceptable = compare_results(
        expected,
        actual,
        lenient_fields=LENIENT_FIELDS,
    )
    print_comparison("GENERIC BASELINE RESULTS", matches, mismatches, acceptable)

    total_strict = len(matches) + len(mismatches)
    assert total_strict > 0 or actual, (
        f"Actor produced no parseable output.\n"
        f"Result: {result[:1000]}\n"
        f"Files: {list(workspace_dir.parent.rglob('*'))}"
    )
