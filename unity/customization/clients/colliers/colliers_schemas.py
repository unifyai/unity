"""
Colliers Pydantic schemas, constants, and file-based result wrappers.

Shared between the Colliers environment tools and any future integrations.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from typing import Any, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    model_validator,
)

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  Financial Data Schemas                                                   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class FieldValue(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    value: float | int | str | None = None
    source: str | None = None


class IncomeData(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    notes: str | None = None
    no_of_days_management_accounts: FieldValue | None = Field(
        default=None,
        alias="No. of Days' Management Accounts",
    )
    fee_deflation_inflation_factor: FieldValue | None = Field(
        default=None,
        alias="Fee Deflation/Inflation Factor",
    )
    average_fee_per_week: FieldValue | None = Field(
        default=None,
        alias="Average fee per week",
    )
    registered_places: FieldValue | None = Field(
        default=None,
        alias="Registered places",
    )
    operational_places: FieldValue | None = Field(
        default=None,
        alias="Operational places",
    )
    occupancy_percent: FieldValue | None = Field(default=None, alias="Occupancy %")
    average_fill: FieldValue | None = Field(default=None, alias="Average fill")
    occupancy_assuming_100_singles: FieldValue | None = Field(
        default=None,
        alias="Occupancy assuming 100% singles",
    )
    fee_income_per_annum: FieldValue | None = Field(
        default=None,
        alias="Fee income per annum",
    )
    other_income: FieldValue | None = Field(default=None, alias="Other income")
    income_per_annum: FieldValue | None = Field(default=None, alias="Income per annum")


class ExpenditureData(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    notes: str | None = None
    employee_wages: FieldValue | None = Field(default=None, alias="Employee Wages")
    agency: FieldValue | None = Field(default=None, alias="Agency")
    national_insurance: FieldValue | None = Field(
        default=None,
        alias="National Insurance",
    )
    pension_contribution: FieldValue | None = Field(
        default=None,
        alias="Pension Contribution",
    )
    payroll: FieldValue | None = Field(default=None, alias="Payroll")
    payroll_percent: FieldValue | None = Field(default=None, alias="Payroll %")
    payroll_psupw: FieldValue | None = Field(default=None, alias="Payroll PSUPW")
    provisions: FieldValue | None = Field(default=None, alias="Provisions")
    provisions_psupw: FieldValue | None = Field(default=None, alias="Provisions PSUPW")
    heat_and_light: FieldValue | None = Field(default=None, alias="Heat & Light")
    heat_and_light_percent: FieldValue | None = Field(
        default=None,
        alias="Heat & Light %",
    )
    accountancy: FieldValue | None = Field(default=None, alias="Accountancy")
    bank_charges: FieldValue | None = Field(default=None, alias="Bank Charges")
    clinical_waste: FieldValue | None = Field(default=None, alias="Clinical Waste")
    council_tax: FieldValue | None = Field(default=None, alias="Council tax")
    gardening_costs: FieldValue | None = Field(default=None, alias="Gardening Costs")
    insurance: FieldValue | None = Field(default=None, alias="Insurance")
    laundry_and_cleaning: FieldValue | None = Field(
        default=None,
        alias="Laundry & Cleaning",
    )
    advertising: FieldValue | None = Field(default=None, alias="Advertising")
    medical_costs: FieldValue | None = Field(default=None, alias="Medical Costs")
    motor_costs: FieldValue | None = Field(default=None, alias="Motor Costs")
    print_post_stationery_it: FieldValue | None = Field(
        default=None,
        alias="Print,Post,Stationery, IT",
    )
    professional_fees: FieldValue | None = Field(
        default=None,
        alias="Professional Fees",
    )
    registration: FieldValue | None = Field(default=None, alias="Registration")
    repairs_renewal_maintenance: FieldValue | None = Field(
        default=None,
        alias="Repairs, Renewal, Maintenance",
    )
    residents_activities: FieldValue | None = Field(
        default=None,
        alias="Resident's Activities",
    )
    staff_training_and_uniforms: FieldValue | None = Field(
        default=None,
        alias="Staff Training and Uniforms",
    )
    subscriptions: FieldValue | None = Field(default=None, alias="Subscriptions")
    telephone: FieldValue | None = Field(default=None, alias="Telephone")
    water_rates: FieldValue | None = Field(default=None, alias="Water rates")
    sundries: FieldValue | None = Field(default=None, alias="Sundries")


class SummaryTotalsData(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    notes: str | None = None
    total_expenses: FieldValue | None = Field(default=None, alias="Total Expenses")
    total_expenses_psupw: FieldValue | None = Field(
        default=None,
        alias="Total expenses PSUPW",
    )
    non_payroll: FieldValue | None = Field(default=None, alias="Non Payroll")
    non_payroll_percent: FieldValue | None = Field(default=None, alias="Non Payroll %")
    non_payroll_psupw: FieldValue | None = Field(
        default=None,
        alias="Non Payroll PSUPW",
    )
    ebitda_fmop: FieldValue | None = Field(default=None, alias="EBITDA / FMOP")
    ebitda_fmop_percent: FieldValue | None = Field(
        default=None,
        alias="EBITDA / FMOP %",
    )
    ebitda_fmop_psupw: FieldValue | None = Field(
        default=None,
        alias="EBITDA / FMOP PSUPW",
    )


class AdditionalItemsData(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    notes: str | None = None
    directors_drawings: FieldValue | None = Field(
        default=None,
        alias="Director's Drawings",
    )
    directors_ni: FieldValue | None = Field(default=None, alias="Director's NI")
    rent: FieldValue | None = Field(default=None, alias="Rent")
    equipment_hire: FieldValue | None = Field(default=None, alias="Equipment Hire")
    depreciation: FieldValue | None = Field(default=None, alias="Depreciation")
    interest_payments: FieldValue | None = Field(
        default=None,
        alias="Interest Payments",
    )
    head_office_costs: FieldValue | None = Field(
        default=None,
        alias="Head Office Costs",
    )
    management_costs: FieldValue | None = Field(default=None, alias="Management Costs")
    motor_vehicles: FieldValue | None = Field(default=None, alias="Motor Vehicles")
    fixtures_and_fittings: FieldValue | None = Field(
        default=None,
        alias="Fixtures and Fittings",
    )
    other_one_off_costs: FieldValue | None = Field(
        default=None,
        alias="Other 'one off' costs",
    )
    amortisation_of_goodwill: FieldValue | None = Field(
        default=None,
        alias="Amortisation of Goodwill",
    )
    other: FieldValue | None = Field(default=None, alias="Other")
    total_expenses_in_accounts: FieldValue | None = Field(
        default=None,
        alias="Total Expenses in accounts",
    )
    total_net_profit_loss: FieldValue | None = Field(
        default=None,
        alias="TOTAL NET PROFIT/LOSS",
    )


class FiscalYearData(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    property_name: str
    fiscal_year: str
    days_in_period: int | None = None
    source_files: list[str] = []
    INCOME: IncomeData | None = None
    EXPENDITURE: ExpenditureData | None = None
    SUMMARY_TOTALS: SummaryTotalsData | None = None
    ADDITIONAL_ITEMS: AdditionalItemsData | None = None
    notes: str | None = None


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  Web Search / Deal Research Schema                                        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class DealRow(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    deal_date: Optional[date] = Field(default=None, alias="Date")
    name: Optional[str] = Field(default=None, alias="Name")
    status: Optional[str] = Field(default=None, alias="Status")
    address: Optional[str] = Field(default=None, alias="Address")
    description: Optional[str] = Field(default=None, alias="Description")
    deal_type: Optional[str] = Field(default=None, alias="Deal Type")
    tenant: Optional[str] = Field(default=None, alias="Tenant")
    lease_terms: Optional[str] = Field(default=None, alias="Lease Terms")
    rent: Optional[Decimal] = Field(default=None, alias="Rent")
    rent_pb: Optional[Decimal] = Field(default=None, alias="Rent PB")
    price: Optional[Decimal] = Field(default=None, alias="Price")
    cv_pb: Optional[Decimal] = Field(default=None, alias="CV PB")
    yield_pct: Optional[str] = Field(default=None, alias="Yield")
    vendor: Optional[str] = Field(default=None, alias="Vendor")
    vendor_agent: Optional[str] = Field(default=None, alias="Vendor Agent")
    purchaser: Optional[str] = Field(default=None, alias="Purchaser")
    comments: Optional[str] = Field(default=None, alias="Comments")
    inputter: Optional[str] = Field(default=None, alias="Inputter")
    date_launched_added: Optional[date] = Field(
        default=None,
        alias="Date Launched / Added",
    )
    postcode: Optional[str] = Field(default=None, alias="Postcode")
    region: Optional[str] = Field(default=None, alias="Region")
    tenure: Optional[str] = Field(default=None, alias="Tenure")
    single_asset_or_portfolio: Optional[str] = Field(
        default=None,
        alias="Single Asset or Portfolio",
    )
    homes: Optional[int] = Field(default=None, alias="Homes")
    beds: Optional[int] = Field(default=None, alias="Beds")
    build_type: Optional[str] = Field(default=None, alias="Build Type")
    age: Optional[str] = Field(default=None, alias="Age")
    quote_price: Optional[Decimal] = Field(default=None, alias="Quote Price")
    quote_cv_pb: Optional[Decimal] = Field(default=None, alias="Quote CV PB")
    quote_yield: Optional[str] = Field(default=None, alias="Quote Yield")
    yield_grouping_category: Optional[str] = Field(
        default=None,
        alias="Yield Grouping / Category",
    )
    purchaser_agent: Optional[str] = Field(default=None, alias="Purchaser Agent")
    epc_rating: Optional[str] = Field(default=None, alias="EPC Rating")
    vendor_type: Optional[str] = Field(default=None, alias="Vendor Type")
    vendor_nationality: Optional[str] = Field(default=None, alias="Vendor Nationality")
    vendors_global_territory: Optional[str] = Field(
        default=None,
        alias="Vendor's Global Territory",
    )
    purchaser_type: Optional[str] = Field(default=None, alias="Purchaser Type")
    purchaser_nationality: Optional[str] = Field(
        default=None,
        alias="Purchaser Nationality",
    )
    purchasers_global_territory: Optional[str] = Field(
        default=None,
        alias="Purchaser's Global Territory",
    )
    achieved_vs_quote_gbp: Optional[Decimal] = Field(
        default=None,
        alias="Achieved vs Quote (£)",
    )
    achieved_vs_quote_pct: Optional[Decimal] = Field(
        default=None,
        alias="Achieved vs Quote (%)",
    )
    achieved_vs_quote_cv_pb: Optional[Decimal] = Field(
        default=None,
        alias="Achieved vs Quote (CV PB)",
    )
    achieved_vs_quote_niy_basis_point: Optional[Decimal] = Field(
        default=None,
        alias="Achieved vs Quote (NIY Basis Point)",
    )
    transaction_quarter: Optional[str] = Field(
        default=None,
        alias="Transaction Quarter",
    )
    transaction_year: Optional[int] = Field(default=None, alias="Transaction Year")
    transaction_time_weeks: Optional[Decimal] = Field(
        default=None,
        alias="Transaction Time (Weeks)",
    )
    transaction_time_months: Optional[Decimal] = Field(
        default=None,
        alias="Transaction Time (Months)",
    )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  File-Based Result Wrappers (response_format for subagents)               ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


def _create_file_result_schema(data_schema: type) -> type:
    """Build a Pydantic model where ``file_path`` is the only field and a
    model validator reads + validates the JSON at that path against
    *data_schema*.  Used as ``response_format`` for subagents so they return
    a file path rather than massive inline JSON."""

    class FileBasedResult(BaseModel):
        file_path: str
        message: str | None = None
        _validated_data: Any = PrivateAttr(default=None)

        @model_validator(mode="after")
        def validate_file_contents(self) -> "FileBasedResult":
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                adapter = TypeAdapter(data_schema)
                self._validated_data = adapter.validate_python(raw_data)
            except FileNotFoundError:
                raise ValueError(f"Output file not found: {self.file_path}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in output file: {e}")
            except Exception as e:
                raise ValueError(f"Validation failed: {e}")
            return self

        @property
        def data(self):
            return self._validated_data

    return FileBasedResult


FinancialDataResult = _create_file_result_schema(list[FiscalYearData])
WebSearchResult = _create_file_result_schema(list[DealRow])


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  Constants for Excel Generation                                           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

FINANCIAL_DATA_SCHEMA = {
    "INCOME": [
        "No. of Days' Management Accounts",
        "Fee Deflation/Inflation Factor",
        "Average fee per week",
        "Registered places",
        "Operational places",
        "Occupancy %",
        "Average fill",
        "Occupancy assuming 100% singles",
        "Fee income per annum",
        "Other income",
        "Income per annum",
    ],
    "EXPENDITURE": [
        "Employee Wages",
        "Agency",
        "National Insurance",
        "Pension Contribution",
        "Payroll",
        "Payroll %",
        "Payroll PSUPW",
        "Provisions",
        "Provisions PSUPW",
        "Heat & Light",
        "Heat & Light %",
        "Accountancy",
        "Bank Charges",
        "Clinical Waste",
        "Council tax",
        "Gardening Costs",
        "Insurance",
        "Laundry & Cleaning",
        "Advertising",
        "Medical Costs",
        "Motor Costs",
        "Print,Post,Stationery, IT",
        "Professional Fees",
        "Registration",
        "Repairs, Renewal, Maintenance",
        "Resident's Activities",
        "Staff Training and Uniforms",
        "Subscriptions",
        "Telephone",
        "Water rates",
        "Sundries",
    ],
    "SUMMARY_TOTALS": [
        "Total Expenses",
        "Total expenses PSUPW",
        "Non Payroll",
        "Non Payroll %",
        "Non Payroll PSUPW",
        "EBITDA / FMOP",
        "EBITDA / FMOP %",
        "EBITDA / FMOP PSUPW",
    ],
    "ADDITIONAL_ITEMS": [
        "Director's Drawings",
        "Director's NI",
        "Rent",
        "Equipment Hire",
        "Depreciation",
        "Interest Payments",
        "Head Office Costs",
        "Management Costs",
        "Motor Vehicles",
        "Fixtures and Fittings",
        "Other 'one off' costs",
        "Amortisation of Goodwill",
        "Other",
        "Total Expenses in accounts",
        "TOTAL NET PROFIT/LOSS",
    ],
}

FIELD_NAME_MAPPING = {
    "No. of Days' Management Accounts": "no_of_days_management_accounts",
    "Fee Deflation/Inflation Factor": "fee_deflation_inflation_factor",
    "Average fee per week": "average_fee_per_week",
    "Registered places": "registered_places",
    "Operational places": "operational_places",
    "Occupancy %": "occupancy_percent",
    "Average fill": "average_fill",
    "Occupancy assuming 100% singles": "occupancy_assuming_100_singles",
    "Fee income per annum": "fee_income_per_annum",
    "Other income": "other_income",
    "Income per annum": "income_per_annum",
    "Employee Wages": "employee_wages",
    "Agency": "agency",
    "National Insurance": "national_insurance",
    "Pension Contribution": "pension_contribution",
    "Payroll": "payroll",
    "Payroll %": "payroll_percent",
    "Payroll PSUPW": "payroll_psupw",
    "Provisions": "provisions",
    "Provisions PSUPW": "provisions_psupw",
    "Heat & Light": "heat_and_light",
    "Heat & Light %": "heat_and_light_percent",
    "Accountancy": "accountancy",
    "Bank Charges": "bank_charges",
    "Clinical Waste": "clinical_waste",
    "Council tax": "council_tax",
    "Gardening Costs": "gardening_costs",
    "Insurance": "insurance",
    "Laundry & Cleaning": "laundry_and_cleaning",
    "Advertising": "advertising",
    "Medical Costs": "medical_costs",
    "Motor Costs": "motor_costs",
    "Print,Post,Stationery, IT": "print_post_stationery_it",
    "Professional Fees": "professional_fees",
    "Registration": "registration",
    "Repairs, Renewal, Maintenance": "repairs_renewal_maintenance",
    "Resident's Activities": "residents_activities",
    "Staff Training and Uniforms": "staff_training_and_uniforms",
    "Subscriptions": "subscriptions",
    "Telephone": "telephone",
    "Water rates": "water_rates",
    "Sundries": "sundries",
    "Total Expenses": "total_expenses",
    "Total expenses PSUPW": "total_expenses_psupw",
    "Non Payroll": "non_payroll",
    "Non Payroll %": "non_payroll_percent",
    "Non Payroll PSUPW": "non_payroll_psupw",
    "EBITDA / FMOP": "ebitda_fmop",
    "EBITDA / FMOP %": "ebitda_fmop_percent",
    "EBITDA / FMOP PSUPW": "ebitda_fmop_psupw",
    "Director's Drawings": "directors_drawings",
    "Director's NI": "directors_ni",
    "Rent": "rent",
    "Equipment Hire": "equipment_hire",
    "Depreciation": "depreciation",
    "Interest Payments": "interest_payments",
    "Head Office Costs": "head_office_costs",
    "Management Costs": "management_costs",
    "Motor Vehicles": "motor_vehicles",
    "Fixtures and Fittings": "fixtures_and_fittings",
    "Other 'one off' costs": "other_one_off_costs",
    "Amortisation of Goodwill": "amortisation_of_goodwill",
    "Other": "other",
    "Total Expenses in accounts": "total_expenses_in_accounts",
    "TOTAL NET PROFIT/LOSS": "total_net_profit_loss",
}

CATEGORY_TO_SECTION = {
    "INCOME": "INCOME",
    "EXPENDITURE": "EXPENDITURE",
    "SUMMARY_TOTALS": "SUMMARY_TOTALS",
    "ADDITIONAL_ITEMS": "ADDITIONAL_ITEMS",
}
