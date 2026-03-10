"""
Vantage — client customization.

Pre-seeds the assistant with:
- A role identity for the Vantage Carbon Club Assistant
- Guidance entries for M365 data access, portal navigation, and domain knowledge
- Microsoft 365 and Zoho portal credentials via SecretManager
- URL mapping to route Zoho Connect to a local demo replica

Vantage is a UK consultancy that runs best-practice benchmarking clubs for
~90 housing associations. The Carbon Club focuses on decarbonisation —
solar PV, insulation, heat pumps, EPC upgrades. The assistant helps with
knowledge retrieval, data analysis, member management, and post-session admin.
"""

from unity.customization.configs.types.actor_config import ActorConfig
from unity.customization.clients import register_user

# ---------------------------------------------------------------------------
# Actor config — role identity + URL mapping for demo portal
# ---------------------------------------------------------------------------

_VANTAGE_CONFIG = ActorConfig(
    guidelines=(
        "You are the Vantage Carbon Club Assistant. Vantage is a UK "
        "consultancy that runs best-practice benchmarking clubs for ~90 "
        "housing associations. The Carbon Club data — emails, files, "
        "OneNote notes — is managed by Steph Hosny (Club Lead) and "
        "accessible via her Microsoft 365 account "
        "(steph@unifyailtd123.onmicrosoft.com). When you need to search "
        "emails, read files, or check OneNote, access Steph's account "
        "using the stored credentials. The person chatting with you may "
        "be any member of the Vantage team or a client — do not assume "
        "the person talking to you is Steph."
    ),
    url_mappings={
        "https://connect.zoho.com": "http://localhost:4002",
    },
)

# ---------------------------------------------------------------------------
# Guidance entries
# ---------------------------------------------------------------------------

_M365_ACCESS_GUIDANCE = """\
Access Vantage's Carbon Club data via the Microsoft Graph API.

Steph Hosny's Microsoft 365 account contains all Carbon Club data.
Retrieve the access token from the Secret Manager (MS365_ACCESS_TOKEN).

Outlook (emails):
  GET https://graph.microsoft.com/v1.0/me/messages
  Search: GET .../me/messages?$search="Broadland Housing"
  Sent items: GET .../me/mailFolders/SentItems/messages
  Headers: Authorization: Bearer {token}

OneDrive (files):
  List folder: GET .../me/drive/root:/Vantage:/children
  Download file: GET .../me/drive/root:/Vantage/path/to/file.xlsx:/content
  Upload file: PUT .../me/drive/root:/Vantage/path/to/file.xlsx:/content

  Key folders:
  - Vantage/Carbon Club/1-to-1 Notes/ — 1:1 call notes spreadsheet (28 sheets)
  - Vantage/Carbon Club/Attendance/ — attendance log
  - Vantage/Carbon Club/Workshop Transcripts/ — meeting transcripts
  - Vantage/Membership/ — membership master (90 orgs, 253 memberships)
  - Vantage/Benchmarking Data/ — VfM financials, TSM satisfaction, operational KPIs
  - Vantage/Knowledge Hub/ — case study PDFs
  - Vantage/Email Log/ — email interaction log

OneNote (renewal notes):
  List notebooks: GET .../me/onenote/notebooks
  List pages: GET .../me/onenote/notebooks/{id}/sections/{id}/pages
  Read page: GET .../me/onenote/pages/{id}/content
  Notebook name: "Member Notes", section: "Renewal Calls"

For Excel files on OneDrive: download the file, work with it locally
using openpyxl or primitives.files.render_excel, then upload back if
modified.\
"""

_PORTAL_GUIDANCE = """\
Navigate the Vantage Carbon Club community portal (Zoho Connect) to
search and extract member discussion threads.

The portal is at https://connect.zoho.com (routed to local demo replica).
Login credentials are in the Secret Manager (PORTAL_USERNAME, PORTAL_PASSWORD).

Workflow:
1. Create a web session with primitives.computer.web.new_session().
2. Navigate to https://connect.zoho.com and log in.
3. Go to the Forums section to browse discussion threads.
4. Click into a thread to read the original post and all replies.
5. Extract relevant data — member names, organisations, specific numbers
   and practices mentioned in the discussion.
6. Synthesise into a structured summary when asked.

The portal contains 6 discussion threads:
1. Solar PV costs and procurement (Oct 2024)
2. Heat pump experiences (Sep 2024)
3. EPC data quality challenges (Nov 2024)
4. SHDF Wave 2 application tips (Dec 2024)
5. Tenant communications for retrofit works (Jan 2025)
6. Damp & mould management approaches (Feb 2025)\
"""

_DOMAIN_KNOWLEDGE_GUIDANCE = """\
Key domain terms for UK social housing and the Carbon Club:

Housing Association (HA): Non-profit organisation that builds, owns, and
manages affordable rental housing in the UK. Regulated by the Regulator
of Social Housing.

Club: A themed membership group of housing associations facilitated by
Vantage. The Carbon Club focuses on decarbonisation. There are 6 clubs
total (Carbon, Executive, Finance Directors, Operations, Customer
Experience, Performance Improvement).

PV / Solar PV: Photovoltaic solar panels installed on rooftops. Key data:
Greendale Homes did 1,200 installs at GBP4,200/unit. Northfield HA did
850 with battery storage at GBP6,200/unit. Beacon Dwellings did 340 at
GBP4,400/unit.

EPC: Energy Performance Certificate, rated A to G. The UK government
wants all social housing at EPC C or above. Many HAs are at 60-75%.

SHDF: Social Housing Decarbonisation Fund — a UK government grant for
retrofit measures on stock below EPC C. Wave 2.2 opens April 2025.

Void: A vacant property between tenants. Turnaround time is how many days
to get it re-let. Sector average is ~20 days.

TSM: Tenant Satisfaction Measures — standardised survey scores that every
large HA must report to the regulator annually.

VfM: Value for Money — financial benchmarking metrics (cost per unit,
operating margin, gearing, reinvestment rate).

DLO: Direct Labour Organisation — an in-house maintenance team, as
opposed to outsourcing to external contractors.

Retrofit: Upgrading existing properties with energy efficiency measures
(solar PV, insulation, heat pumps, new windows).

RAG: Red/Amber/Green scoring for member engagement — Red means
disengaging and at risk of non-renewal.\
"""

_VANTAGE_GUIDANCE = [
    {
        "title": "Microsoft 365 Data Access for Vantage Carbon Club",
        "content": _M365_ACCESS_GUIDANCE,
    },
    {
        "title": "Zoho Connect Portal Navigation",
        "content": _PORTAL_GUIDANCE,
    },
    {
        "title": "UK Social Housing and Carbon Club Domain Knowledge",
        "content": _DOMAIN_KNOWLEDGE_GUIDANCE,
    },
]

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

_VANTAGE_SECRETS = [
    {
        "name": "MS365_ACCESS_TOKEN",
        "value": "PLACEHOLDER_TOKEN",
        "description": "Microsoft Graph API access token for Steph's M365 account",
    },
    {
        "name": "PORTAL_USERNAME",
        "value": "steph",
        "description": "Zoho Connect portal login username",
    },
    {
        "name": "PORTAL_PASSWORD",
        "value": "demo123",
        "description": "Zoho Connect portal login password",
    },
]

# ---------------------------------------------------------------------------
# User-level registration (Yasser for testing)
# ---------------------------------------------------------------------------

_YASSER_USER_ID = "40144b2a-722f-4f41-8d9e-384c316ee19f"

register_user(
    _YASSER_USER_ID,
    config=_VANTAGE_CONFIG,
    guidance=_VANTAGE_GUIDANCE,
    secrets=_VANTAGE_SECRETS,
)
