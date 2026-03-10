"""
Mock Zoho Connect Portal for Vantage Carbon Club.

A lightweight FastAPI app that simulates the Zoho Connect community portal
with 6 discussion threads seeded with realistic Vantage member data.

Usage:
    uv run uvicorn app:app --reload --port 5000

Login credentials:
    Username: steph
    Password: demo123
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path

app = FastAPI(title="Vantage Connect Portal")
app.add_middleware(SessionMiddleware, secret_key="vantage-demo-secret-key-2025")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

VALID_USERS = {"steph": "demo123", "olivia": "demo123"}

AVATAR_COLORS = [
    "#e91e63",
    "#9c27b0",
    "#673ab7",
    "#3f51b5",
    "#2196f3",
    "#009688",
    "#4caf50",
    "#ff9800",
    "#f44336",
    "#795548",
    "#607d8b",
    "#00bcd4",
    "#8bc34a",
    "#ffc107",
    "#ff5722",
]

CAT_COLORS = {
    "Decarbonisation": "#4caf50",
    "Data & Reporting": "#2196f3",
    "Funding": "#ff9800",
    "Tenant Engagement": "#9c27b0",
    "Compliance": "#f44336",
}

CAT_BADGES = {
    "Decarbonisation": "badge-green",
    "Data & Reporting": "badge-blue",
    "Funding": "badge-orange",
    "Tenant Engagement": "badge-purple",
    "Compliance": "badge-red",
}


def _initials(name):
    parts = name.split()
    return (parts[0][0] + parts[-1][0]).upper() if len(parts) > 1 else name[0].upper()


def _color(name):
    return AVATAR_COLORS[hash(name) % len(AVATAR_COLORS)]


THREADS = [
    {
        "id": 1,
        "title": "Solar PV costs and procurement",
        "date": "8 October 2024",
        "category": "Decarbonisation",
        "op_name": "Lena Morris",
        "op_org": "Broadland Housing",
        "op_role": "Director of Assets",
        "op_text": (
            "Hi all,\n\n"
            "We're looking to start a solar PV programme next financial year and would "
            "appreciate any insights on procurement routes and typical costs per unit. "
            "We're looking at about 500 properties initially.\n\n"
            "Has anyone used a framework agreement? What kind of costs per property "
            "are you seeing for a standard 4kW system?\n\n"
            "Any advice would be really appreciated. Thanks!"
        ),
        "replies": [
            {
                "name": "David Osei",
                "org": "Beacon Dwellings",
                "date": "9 Oct 2024",
                "text": "Hi Lena, we've been using a mix of direct procurement and the Fusion21 framework. Costs have been around \u00a34,400 per property for a 4kW system. We've completed 340 installs so far as part of our wider EPC upgrade programme. The Fusion21 framework was straightforward to access and gave us good competitive tension on pricing.",
            },
            {
                "name": "Mark Jennings",
                "org": "Greendale Homes",
                "date": "10 Oct 2024",
                "text": "We went through the Solarfix Ltd framework \u2014 very smooth procurement process. Our costs have been about \u00a34,200 per property including scaffolding. We've now done 1,200 installs. Happy to share more details if useful, Lena.",
            },
            {
                "name": "Fiona Clarke",
                "org": "Northfield HA",
                "date": "11 Oct 2024",
                "text": "Just to add a different perspective \u2014 we went PV plus battery storage at \u00a36,200 per unit. More expensive but tenants save significantly more. Worth considering depending on your tenant demographic and usage patterns.",
            },
            {
                "name": "Karen Blackwell",
                "org": "Westmoor Housing Group",
                "date": "12 Oct 2024",
                "text": "Lena, if you haven't already, check whether your DNO has capacity constraints in your area. We found that grid connection was a bottleneck and we had to submit applications well in advance. Worth investigating early.",
            },
        ],
    },
    {
        "id": 2,
        "title": "Heat pump experiences",
        "date": "15 September 2024",
        "category": "Decarbonisation",
        "op_name": "Andrew Marsh",
        "op_org": "Severn Vale Homes",
        "op_role": "Operations Director",
        "op_text": (
            "We've just completed a pilot of 50 air source heat pumps on off-gas "
            "properties. Results are mixed \u2014 great energy savings but some tenant "
            "complaints about noise and running costs in very cold weather.\n\n"
            "Would be keen to hear others' experiences."
        ),
        "replies": [
            {
                "name": "Fiona Clarke",
                "org": "Northfield HA",
                "date": "16 Sep 2024",
                "text": "We've had similar findings with our 50-property pilot. The key issue for us was installation costs coming in at \u00a39,500 vs our \u00a38,000 estimate. Noise complaints from about 15% of tenants, mainly in semi-detached properties where the unit is near a neighbour's bedroom window.",
            },
            {
                "name": "Priya Sharma",
                "org": "Oaktree Living",
                "date": "17 Sep 2024",
                "text": "We haven't started heat pumps yet but we're watching closely. Our concern is the running costs for tenants on prepayment meters \u2014 the electricity tariff is higher than gas. Has anyone looked at whether tenants are actually better off financially?",
            },
            {
                "name": "Oliver Grant",
                "org": "Thameside Housing Trust",
                "date": "18 Sep 2024",
                "text": "We've done 120 heat pump installs across two estates. Overall satisfaction is 78% which is lower than we'd like. The main complaints are about the need for larger radiators and the fact that heat pumps don't provide instant heat like gas boilers. Education and expectation setting is crucial.",
            },
            {
                "name": "Claire Whitfield",
                "org": "Pennine Valleys Housing",
                "date": "19 Sep 2024",
                "text": "Just flagging that the government's Boiler Upgrade Scheme provides \u00a37,500 towards ASHPs. We've been using this alongside SHDF funding to bring our net costs down significantly.",
            },
        ],
    },
    {
        "id": 3,
        "title": "EPC data quality challenges",
        "date": "5 November 2024",
        "category": "Data & Reporting",
        "op_name": "James Thornton",
        "op_org": "Riverside Community Housing",
        "op_role": "Head of Property Services",
        "op_text": (
            "We've been cross-referencing our EPC data with recent stock condition "
            "surveys and finding significant discrepancies \u2014 about 20% of our EPCs "
            "don't match what's actually in the property.\n\n"
            "Is anyone else experiencing this? How are you managing it?"
        ),
        "replies": [
            {
                "name": "Mark Jennings",
                "org": "Greendale Homes",
                "date": "6 Nov 2024",
                "text": "Same problem here. We're working with Elmhurst Energy to resurvey about 2,000 properties. Initial estimate is 15% have inaccurate EPCs. The issue is mainly with pre-1940 solid wall stock where internal insulation has been done but the EPC wasn't updated.",
            },
            {
                "name": "Sarah Linehan",
                "org": "Meridian Housing Group",
                "date": "6 Nov 2024",
                "text": "We've taken a different approach \u2014 building our own asset database that tracks installed measures against each property and calculates what the EPC should be. Not official but gives us a much more accurate picture for investment planning.",
            },
            {
                "name": "David Osei",
                "org": "Beacon Dwellings",
                "date": "7 Nov 2024",
                "text": "We found about 25% inaccuracy in our solid wall stock EPCs. Real problem when you're trying to target retrofit programmes at properties below EPC C. We've budgeted for a full resurvey in 2025/26.",
            },
            {
                "name": "Samira Begum",
                "org": "Ironbridge Homes",
                "date": "8 Nov 2024",
                "text": "The regulatory implications worry me. If our reported EPC data is significantly wrong, that could be a compliance issue when the regulator comes asking. Are there any consequences for inaccurate data at the moment?",
            },
            {
                "name": "Oliver Grant",
                "org": "Thameside Housing Trust",
                "date": "9 Nov 2024",
                "text": "No direct penalty mechanism currently but the expectation is that data should be accurate. With the new consumer standards, if you're claiming 75% at EPC C but it's actually 65%, that's a governance concern.",
            },
        ],
    },
    {
        "id": 4,
        "title": "SHDF Wave 2 application tips",
        "date": "2 December 2024",
        "category": "Funding",
        "op_name": "Lena Morris",
        "op_org": "Broadland Housing",
        "op_role": "Director of Assets",
        "op_text": (
            "We're planning to apply for SHDF Wave 2.2 when the window opens in "
            "April. This will be our first application.\n\n"
            "Any tips from those who've been through the process?"
        ),
        "replies": [
            {
                "name": "Claire Whitfield",
                "org": "Pennine Valleys Housing",
                "date": "3 Dec 2024",
                "text": "Start early! The application is very detailed \u2014 you need property-level data, EPC certificates, contractor quotes, tenant engagement plans. It took us about six weeks. We got \u00a31.2m for 450 measures across 280 properties.",
            },
            {
                "name": "David Osei",
                "org": "Beacon Dwellings",
                "date": "4 Dec 2024",
                "text": "Agree with Claire \u2014 the data requirements are significant. Make sure your EPC data is accurate before you apply. We had to redo some of our submission because our EPC data didn't match the properties we were targeting.",
            },
            {
                "name": "Mark Jennings",
                "org": "Greendale Homes",
                "date": "5 Dec 2024",
                "text": "Our tip would be to have your procurement route sorted before you apply. Being able to show you have a framework agreement in place strengthens the bid significantly. We referenced our Solarfix framework and I think that helped.",
            },
            {
                "name": "Nadeem Hussain",
                "org": "Ashworth Housing Trust",
                "date": "6 Dec 2024",
                "text": "Does anyone know whether the 33% match funding can come from the Affordable Homes Programme, or does it have to be the organisation's own capital? We're trying to work out our budget.",
            },
        ],
    },
    {
        "id": 5,
        "title": "Tenant communications for retrofit works",
        "date": "15 January 2025",
        "category": "Tenant Engagement",
        "op_name": "Karen Blackwell",
        "op_org": "Westmoor Housing Group",
        "op_role": "Head of Sustainability",
        "op_text": (
            "We're about to start a major insulation programme and I'm looking for "
            "examples of good tenant communication.\n\n"
            "How are others managing expectations and getting buy-in?"
        ),
        "replies": [
            {
                "name": "James Thornton",
                "org": "Riverside Community Housing",
                "date": "16 Jan 2025",
                "text": "We've developed template letters for different retrofit types \u2014 PV, insulation, windows, heat pumps. The key is being specific about savings: 'you could save \u00a3150\u2013200/year' rather than just 'you'll save money'. Happy to share our templates.",
            },
            {
                "name": "Darren Walsh",
                "org": "Summit Housing Partnership",
                "date": "17 Jan 2025",
                "text": "We recruited a dedicated tenant liaison officer for our retrofit programme. Single point of contact throughout the process. Made a massive difference \u2014 tenants actually thank us now.",
            },
            {
                "name": "Karen Blackwell",
                "org": "Westmoor Housing Group",
                "date": "18 Jan 2025",
                "text": "Thanks both. We've found that face-to-face visits work much better than letters for our older tenants. Our refusal rate dropped from 15% to under 5% once we started doing home visits. More resource-intensive but worth it.",
            },
        ],
    },
    {
        "id": 6,
        "title": "Damp & mould management approaches",
        "date": "3 February 2025",
        "category": "Compliance",
        "op_name": "Helen Foster",
        "op_org": "Riverview Estates",
        "op_role": "Head of Housing",
        "op_text": (
            "Following the coroner's recommendations in the Awaab Ishak case, we've "
            "been reviewing our damp and mould procedures.\n\n"
            "What response times are others targeting and how are you prioritising cases?"
        ),
        "replies": [
            {
                "name": "Fiona Clarke",
                "org": "Northfield HA",
                "date": "4 Feb 2025",
                "text": "We implemented a 48-hour inspection target for all damp and mould reports. Triaged into three categories: emergency (structural/health risk \u2014 same day), urgent (visible mould \u2014 48 hours), routine (condensation advice \u2014 5 days). Reduced open cases from 180 to 45 in six months.",
            },
            {
                "name": "Chris Doyle",
                "org": "Lakeside Living",
                "date": "5 Feb 2025",
                "text": "We've invested in environmental monitoring sensors in our highest-risk properties \u2014 about 500 sensors across our stock. They alert us to high humidity before mould develops. Proactive rather than reactive. Cost about \u00a380 per sensor including installation.",
            },
            {
                "name": "Joanna Briggs",
                "org": "Harrowfield Homes",
                "date": "5 Feb 2025",
                "text": "We're training all housing officers to do basic damp and mould assessments during routine visits. The idea is early identification before tenants even report it. Also providing dehumidifiers and extractor fans to tenants in high-risk properties.",
            },
            {
                "name": "Lena Morris",
                "org": "Broadland Housing",
                "date": "6 Feb 2025",
                "text": "We've been struggling with this \u2014 currently at 35 cases per 1,000 homes which is above the sector average. We've just brought in a specialist surveyor to review our worst-affected properties and develop a remediation programme. Concerned about compliance with the new Awaab's Law requirements.",
            },
        ],
    },
]

for t in THREADS:
    t["initials"] = _initials(t["op_name"])
    t["color"] = _color(t["op_name"])
    t["badge_class"] = CAT_BADGES.get(t["category"], "badge-blue")
    t["reply_count"] = len(t["replies"])
    for r in t["replies"]:
        r["initials"] = _initials(r["name"])
        r["color"] = _color(r["name"])

CATEGORIES = list(dict.fromkeys(t["category"] for t in THREADS))


def _require_login(request: Request):
    user = request.session.get("user")
    if not user:
        return None
    return user


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if _require_login(request):
        return RedirectResponse("/forums", status_code=302)
    return RedirectResponse("/login", status_code=302)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    clean_user = username.strip().lower().split("@")[0]
    if clean_user in VALID_USERS and VALID_USERS[clean_user] == password:
        request.session["user"] = clean_user
        return RedirectResponse("/forums", status_code=303)
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid username or password"},
    )


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


@app.get("/forums", response_class=HTMLResponse)
async def forums_list(request: Request, category: str = None):
    user = _require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    threads = THREADS
    if category:
        threads = [t for t in THREADS if t["category"] == category]

    return templates.TemplateResponse(
        "forums.html",
        {
            "request": request,
            "threads": threads,
            "categories": CATEGORIES,
            "cat_colors": CAT_COLORS,
        },
    )


@app.get("/forums/{thread_id}", response_class=HTMLResponse)
async def thread_detail(request: Request, thread_id: int):
    user = _require_login(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    thread = next((t for t in THREADS if t["id"] == thread_id), None)
    if not thread:
        return RedirectResponse("/forums", status_code=302)

    return templates.TemplateResponse(
        "thread.html",
        {
            "request": request,
            "thread": thread,
            "categories": CATEGORIES,
            "cat_colors": CAT_COLORS,
        },
    )
