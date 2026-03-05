"""
Colliers Healthcare — client customization.

Pre-seeds the assistant with:
- A thin role identity (ActorConfig.guidelines)
- Two stored functions for producing formatted Excel deliverables
- Four focused guidance entries covering extraction workflows and schemas
- CoStar credentials via SecretManager

All workflow knowledge lives in the evolvable GuidanceManager and
FunctionManager stores.  The assistant can refine, extend, or replace
any of this through normal use and StorageCheck learning.
"""

from pathlib import Path

from unity.customization.configs.types.actor_config import ActorConfig
from unity.customization.clients import register_org, register_team, register_user

# ---------------------------------------------------------------------------
# Actor config — thin role identity only
# ---------------------------------------------------------------------------

_COLLIERS_CONFIG = ActorConfig(
    guidelines=(
        "You are the Colliers Healthcare Valuation Assistant, specialising "
        "in UK healthcare property valuations, financial data extraction "
        "from documents, and deal research."
    ),
)

# ---------------------------------------------------------------------------
# Pre-seeded functions (via function_dir)
# ---------------------------------------------------------------------------

_COLLIERS_FUNCTION_DIR = Path(__file__).parent / "functions"

# ---------------------------------------------------------------------------
# Guidance entries — focused, independently searchable recipes
# ---------------------------------------------------------------------------

_FINANCIAL_EXTRACTION_GUIDANCE = """\
Extract standardised financial data from PDF and Excel documents for
healthcare property valuations.

Workflow:
1. List files in the provided directory to find all PDF and Excel files.
2. For Excel files, open with openpyxl and iterate over visible sheets
   (skip sheets where sheet_state != 'visible'). For each sheet, use the
   render-first approach: render the full sheet to get a global view of
   the layout, then zoom into specific ranges and read exact cell values
   with openpyxl.
3. For PDFs, render 2-3 pages at a time to survey the document visually
   before extracting data from specific pages.
4. Identify fiscal year data — look for income statements, P&L,
   management accounts across multiple fiscal years.
5. Extract data into the Healthcare Valuation Financial Data Schema (see
   the separate guidance entry for field definitions). Each extracted
   value should be a FieldValue dict with "value" and "source" keys,
   where source records where the value was found (sheet name + cell
   reference, or page number).
6. Save the extracted data as a JSON array to a file, then call
   `create_financial_data_excel` to produce the final formatted Excel
   deliverable.

Key conventions:
- Percentages as decimals (0.9783 for 97.83%).
- Use null for fields not found in the source documents.
- Group data by fiscal year; each fiscal year is a separate object.
- Use stateful sessions (state_mode="stateful") to preserve loaded
  documents and extracted data across steps.\
"""

_WEB_RESEARCH_GUIDANCE = """\
Research UK care home deals and transactions by navigating CoStar and
extracting structured deal data.

Workflow:
1. Retrieve CoStar credentials from the Secret Manager (COSTAR_USERNAME,
   COSTAR_PASSWORD).
2. Create a web session with primitives.computer.web.new_session(). If
   the user's screen is being shared, pass visible=True.
3. Navigate to https://www.costar.com and log in. Use the ${SECRET_NAME}
   syntax with type_text() so credentials are never exposed.
4. Search for care home transactions and deals. By default, collect the
   last 20 deals unless the user specifies a different range.
5. Extract deal information matching the Deal Tracker Data Schema (see
   the separate guidance entry for field definitions). Use the
   session.observe() method with a structured response format.
6. Scroll and paginate to collect as many deals as possible.
7. Deduplicate results before saving — if two deals share the same name
   and address (case-insensitive), keep the entry with more non-null
   fields.
8. Save deduplicated deals as a JSON array, then call
   `create_web_search_excel` to produce the final Excel deliverable.
9. Stop the session when done.

Key conventions:
- Prefer low-level browser actions (click, type_text, scroll) over
  act() for faster, more deterministic execution.
- Take screenshots between actions to verify page state.
- Currency values are numeric (no symbols). Dates in ISO format.
- Yield fields may be strings with ranges (e.g. "6.5-7%").
- Include the source URL in the comments field.\
"""

_FINANCIAL_SCHEMA_GUIDANCE = """\
Schema for healthcare valuation financial data extraction. Each fiscal
year record is a JSON object with the structure below.

Top-level fields:
- property_name (str): Name of the property being valued.
- fiscal_year (str): The fiscal year period (e.g. "2023", "FY2022/23").
- days_in_period (int or null): Number of days in the accounting period.
- source_files (list of str): Filenames the data was extracted from.
- notes (str or null): Free-text notes about this fiscal year's data.

Each record contains four sections — INCOME, EXPENDITURE,
SUMMARY_TOTALS, ADDITIONAL_ITEMS — where every field is a FieldValue:
{"value": <number|string|null>, "source": "<where the value was found>"}.

INCOME fields:
  No. of Days' Management Accounts, Fee Deflation/Inflation Factor,
  Average fee per week, Registered places, Operational places,
  Occupancy %, Average fill, Occupancy assuming 100% singles,
  Fee income per annum, Other income, Income per annum

EXPENDITURE fields:
  Employee Wages, Agency, National Insurance, Pension Contribution,
  Payroll, Payroll %, Payroll PSUPW, Provisions, Provisions PSUPW,
  Heat & Light, Heat & Light %, Accountancy, Bank Charges,
  Clinical Waste, Council tax, Gardening Costs, Insurance,
  Laundry & Cleaning, Advertising, Medical Costs, Motor Costs,
  Print/Post/Stationery/IT, Professional Fees, Registration,
  Repairs/Renewal/Maintenance, Resident's Activities,
  Staff Training and Uniforms, Subscriptions, Telephone, Water rates,
  Sundries

SUMMARY TOTALS fields:
  Total Expenses, Total expenses PSUPW, Non Payroll, Non Payroll %,
  Non Payroll PSUPW, EBITDA / FMOP, EBITDA / FMOP %, EBITDA / FMOP PSUPW

ADDITIONAL ITEMS fields:
  Director's Drawings, Director's NI, Rent, Equipment Hire,
  Depreciation, Interest Payments, Head Office Costs, Management Costs,
  Motor Vehicles, Fixtures and Fittings, Other 'one off' costs,
  Amortisation of Goodwill, Other, Total Expenses in accounts,
  TOTAL NET PROFIT/LOSS

If the source documents contain additional categories not listed here,
include them in the notes field and flag for review.\
"""

_DEAL_TRACKER_SCHEMA_GUIDANCE = """\
Schema for web-research deal data extraction. Each deal is a JSON object
with the following fields (all optional — use null when not available):

Primary fields (core deal information):
  deal_date (ISO date), name (str), status (str), address (str),
  description (str), deal_type (str), tenant (str), lease_terms (str),
  rent (numeric), rent_pb (numeric, rent per bed), price (numeric),
  cv_pb (numeric, capital value per bed), yield_pct (str, e.g. "6.5%"),
  vendor (str), vendor_agent (str), purchaser (str), comments (str),
  inputter (str)

Secondary fields (additional deal metadata):
  date_launched_added (ISO date), postcode (str), region (str),
  tenure (str), single_asset_or_portfolio (str),
  homes (int), beds (int), build_type (str), age (str),
  quote_price (numeric), quote_cv_pb (numeric), quote_yield (str),
  yield_grouping_category (str), purchaser_agent (str),
  epc_rating (str), vendor_type (str), vendor_nationality (str),
  vendors_global_territory (str), purchaser_type (str),
  purchaser_nationality (str), purchasers_global_territory (str),
  achieved_vs_quote_gbp (numeric), achieved_vs_quote_pct (numeric),
  achieved_vs_quote_cv_pb (numeric),
  achieved_vs_quote_niy_basis_point (numeric),
  transaction_quarter (str), transaction_year (int),
  transaction_time_weeks (numeric), transaction_time_months (numeric)

Currency values should be plain numbers (no currency symbols).
Dates should be in ISO format (YYYY-MM-DD).
Yield fields may be strings containing ranges (e.g. "6.5-7%").\
"""

_COLLIERS_GUIDANCE = [
    {
        "title": "Financial Data Extraction from Excel and PDF Documents",
        "content": _FINANCIAL_EXTRACTION_GUIDANCE,
    },
    {
        "title": "CoStar Web Research for UK Care Home Deals",
        "content": _WEB_RESEARCH_GUIDANCE,
    },
    {
        "title": "Healthcare Valuation Financial Data Schema",
        "content": _FINANCIAL_SCHEMA_GUIDANCE,
    },
    {
        "title": "Deal Tracker Data Schema",
        "content": _DEAL_TRACKER_SCHEMA_GUIDANCE,
    },
]

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

_COLLIERS_SECRETS = [
    {
        "name": "COSTAR_USERNAME",
        "value": "adam.lenton@colliers.com",
        "description": "Costar.com login username / email",
    },
    {
        "name": "COSTAR_PASSWORD",
        "value": "Liverpool*2019",
        "description": "Costar.com login password",
    },
]

# ---------------------------------------------------------------------------
# Org-level registration (Colliers Healthcare, org ID 2)
# ---------------------------------------------------------------------------

_COLLIERS_ORG_ID = 2

register_org(
    _COLLIERS_ORG_ID,
    config=_COLLIERS_CONFIG,
    function_dir=_COLLIERS_FUNCTION_DIR,
    guidance=_COLLIERS_GUIDANCE,
    secrets=_COLLIERS_SECRETS,
)

# ---------------------------------------------------------------------------
# Team-level registration (Unify-internal testing)
# ---------------------------------------------------------------------------

_UNIFY_COLLIERS_TEAM_ID = 49

register_team(
    _UNIFY_COLLIERS_TEAM_ID,
    config=_COLLIERS_CONFIG,
    function_dir=_COLLIERS_FUNCTION_DIR,
    guidance=_COLLIERS_GUIDANCE,
    secrets=_COLLIERS_SECRETS,
)

# ---------------------------------------------------------------------------
# User-level registration (personal / org-less API keys)
# ---------------------------------------------------------------------------

_DAN_USER_ID = "cli3t38uc0000s60k5zmgj8ez"  # dan@unify.ai on staging
_YASSER_USER_ID = "40144b2a-722f-4f41-8d9e-384c316ee19f"  # yasser@unify.ai on staging

for _uid in (_DAN_USER_ID, _YASSER_USER_ID):
    register_user(
        _uid,
        config=_COLLIERS_CONFIG,
        function_dir=_COLLIERS_FUNCTION_DIR,
        guidance=_COLLIERS_GUIDANCE,
        secrets=_COLLIERS_SECRETS,
    )
