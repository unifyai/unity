"""
Excel financial statement extraction using xlwings.

This function runs on Windows VM with COM automation support.
"""

from unity.function_manager.custom import custom_function


@custom_function(
    venv_name="spreadsheet_demo",
    windows_os_required=True,
    verify=False,
)
async def extract_financial_statements(dir_path: str) -> dict:
    """
    Extract financial statements from Excel files in a directory.

    This function runs on Windows with xlwings COM automation.
    All local dependencies and notebook guides are embedded.

    Args:
        dir_path: Directory containing .xlsx files to process.

    Returns:
        Dict with extracted financial data and total cost.
    """
    # ═══════════════════════════════════════════════════════════════════
    # EMBEDDED NOTEBOOK RESOURCE (from guides/excel_guide.ipynb)
    # ═══════════════════════════════════════════════════════════════════

    EXCEL_GUIDE_NOTEBOOK_JSON = r"""
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["# Excel Data Extraction Guide\n", "\n", "This guide demonstrates how to use xlwings for Excel automation."]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": ["import xlwings as xw\n", "\n", "# Open workbook\n", "wb = xw.Book('example.xlsx')\n", "sheet = wb.sheets[0]\n", "\n", "# Read data range\n", "data = sheet.range('A1:D10').value\n", "print(data)"]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": ["# Navigate sheets and find tables\n", "for sheet in wb.sheets:\n", "    print(f'Sheet: {sheet.name}')\n", "    used_range = sheet.used_range\n", "    print(f'Used range: {used_range.address}')"]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
"""

    # ═══════════════════════════════════════════════════════════════════
    # INLINE HELPER: Parse embedded notebook to LLM message parts
    # ═══════════════════════════════════════════════════════════════════

    import json as _json

    def notebook_to_llm_parts(notebook_json_str: str, skip_cells: int = 1) -> list:
        """Parse embedded notebook JSON string to LLM message parts."""
        nb = _json.loads(notebook_json_str)
        cells = nb["cells"][skip_cells:]
        parts = []
        for c in cells:
            if c["cell_type"] == "markdown":
                parts.append({"type": "text", "text": "".join(c["source"])})
            elif c["cell_type"] == "code":
                code = "".join(c["source"])
                if code:
                    parts.append({"type": "text", "text": f"```python\n{code}\n```"})
        return parts

    # ═══════════════════════════════════════════════════════════════════
    # IMPORTS (inside function for remote execution)
    # ═══════════════════════════════════════════════════════════════════

    from pathlib import Path as _Path
    from typing import Optional as _Optional

    import pandas as _pd
    import xlwings as _xw

    # ═══════════════════════════════════════════════════════════════════
    # SCHEMA DEFINITIONS (inlined from extraction_functions/schema.py)
    # ═══════════════════════════════════════════════════════════════════

    from pydantic import BaseModel as _BaseModel

    class Income(_BaseModel):
        fee_income_per_annum: _Optional[float] = None
        other_income: _Optional[float] = None

    class Payroll(_BaseModel):
        employee_wages: _Optional[float] = None
        agency: _Optional[float] = None
        national_insurance: _Optional[float] = None
        pension_contribution: _Optional[float] = None
        payroll: _Optional[float] = None

    class OperationalOverheads(_BaseModel):
        provisions: _Optional[float] = None
        heat_and_light: _Optional[float] = None
        accountancy: _Optional[float] = None
        bank_charges: _Optional[float] = None
        clinical_waste: _Optional[float] = None
        council_tax: _Optional[float] = None
        gardening_cost: _Optional[float] = None
        insurance: _Optional[float] = None
        laundry_and_cleaning: _Optional[float] = None
        advertising: _Optional[float] = None
        medical_costs: _Optional[float] = None
        motor_costs: _Optional[float] = None
        print_post_stationery_it: _Optional[float] = None
        professional_fees: _Optional[float] = None
        registration: _Optional[float] = None
        repairs_renewal_maintenance: _Optional[float] = None
        residents_activities: _Optional[float] = None
        staff_training_and_uniforms: _Optional[float] = None
        subscriptions: _Optional[float] = None
        telephone: _Optional[float] = None
        water_rates: _Optional[float] = None
        sundries: _Optional[float] = None

    class OperationalProfitability(_BaseModel):
        ebitda: _Optional[float] = None

    # ═══════════════════════════════════════════════════════════════════
    # MAIN EXTRACTION LOGIC
    # ═══════════════════════════════════════════════════════════════════

    DATA_SCHEMA = {
        "Income": list(Income.model_fields.keys()),
        "Payroll": list(Payroll.model_fields.keys()),
        "Operational Overheads": list(OperationalOverheads.model_fields.keys()),
        "Operational Profitability": list(OperationalProfitability.model_fields.keys()),
    }

    # Parse embedded notebook guide
    guide_parts = notebook_to_llm_parts(EXCEL_GUIDE_NOTEBOOK_JSON)

    # Resolve directory path
    target_dir = _Path(dir_path).expanduser().resolve()
    if not target_dir.exists():
        return {
            "results": [],
            "total_cost": 0.0,
            "error": f"Directory not found: {dir_path}",
        }

    # Find Excel files
    excel_files = list(target_dir.glob("*.xlsx"))
    if not excel_files:
        return {
            "results": [],
            "total_cost": 0.0,
            "error": f"No .xlsx files found in {dir_path}",
        }

    results = []
    total_cost = 0.0

    for excel_file in excel_files:
        try:
            # Open workbook with xlwings
            wb = _xw.Book(str(excel_file))

            file_result = {
                "filename": excel_file.name,
                "sheets": [],
                "data": {},
            }

            # Extract data from each sheet
            for sheet in wb.sheets:
                sheet_info = {
                    "name": sheet.name,
                    "used_range": (
                        sheet.used_range.address if sheet.used_range else None
                    ),
                }
                file_result["sheets"].append(sheet_info)

                # Read used range as dataframe
                if sheet.used_range:
                    try:
                        df = sheet.used_range.options(_pd.DataFrame, header=True).value
                        if df is not None and not df.empty:
                            file_result["data"][sheet.name] = {
                                "rows": len(df),
                                "columns": list(df.columns),
                            }
                    except Exception:
                        pass

            wb.close()
            results.append(file_result)

        except Exception as e:
            results.append(
                {
                    "filename": excel_file.name,
                    "error": str(e),
                },
            )

    return {
        "results": results,
        "total_cost": total_cost,
        "files_processed": len(excel_files),
        "guide_sections_loaded": len(guide_parts),
        "schema_categories": list(DATA_SCHEMA.keys()),
    }
