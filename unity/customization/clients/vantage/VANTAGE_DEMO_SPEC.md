# Vantage — Demo Specification

## Company Background

**Vantage** ([yourvantage.co.uk](https://yourvantage.co.uk)) is a UK-based consultancy that runs **best-practice benchmarking clubs** for roughly **90 housing associations** (non-profit organisations that build, manage, and maintain affordable rental housing in the UK). Vantage facilitates workshops, one-to-one advisory calls, and working groups under six themed "clubs" — including a Carbon Club focused on decarbonisation, an Executive Club, a Finance Directors' Performance Club, and others. Their team is small (roughly five people: Rob, Steph, Jane, John, Sophie). They track approximately **250 memberships** across the 90 organisations, with around **1,000 individual contacts**.

Their entire stack is Microsoft-heavy: OneDrive, Teams (with transcripts), OneNote, Excel spreadsheets, a WordPress knowledge hub, and a Zoho Connect portal for member Q&A.

---

## Workflow 1 — Cross-Source Knowledge Retrieval

### 1. Transcript Citations

> **Robert Bryan (08:00):** *"If someone says to Steph which three housing associations are operating the best with a particular system or they've got particular high level of performance in an area or they've got solar PV on the roofs, they're using some retrofit technologies — at the moment because we're building more knowledge, it's getting quite manual."*

> **Stephanie Hosny (06:56):** *"So somebody will come to me and say who's doing great stuff on electric vehicle fleet or who got great no access rates. And often we're relying on going through different data sources. So if I have a renewal call with them as well I might have like a note in OneNote where they might mention some stuff."*

> **Stephanie Hosny (11:19):** *"If it's something really specific like that… I will literally just be opening an Excel spreadsheet and doing a search on like where's PV or who's doing what and solar or just refreshing my memory."*

> **Stephanie Hosny (11:32):** *"It's because most of our members are in multiple clubs. It's then you might be going through five different spreadsheets or — I wasn't on the one to one. It was somebody else."*

> **Stephanie Hosny (22:13):** *"The other week for example people were asking about void standards and thresholds and things. So I took, you know, maybe there's like 10. I exported all that data. I just put it in a spreadsheet [or] Word document actually and shared it with a few people."*

### 2. Plain-English Explanation

Vantage staff accumulate knowledge about what each housing association is doing well (or poorly) across dozens of topics — solar panel installations, electric vehicle fleet rollout, void turnaround times (a "void" is a vacant property awaiting its next tenant; "turnaround time" is how many days it takes to get it re-let), retrofit programmes, complaint handling, and more.

This knowledge is scattered across:

- **Excel spreadsheets** with one-to-one call notes (one per club, going back ~4 years)
- **OneNote notebooks** with ad-hoc observations
- **Teams meeting transcripts** from workshops and group calls
- **Zoho Connect portal posts** where members ask each other questions and share responses
- **A WordPress knowledge hub** hosting shared documents, policies, and case studies

When a member asks "Who is doing great work on X?", a Vantage staff member currently has to manually search through 3–5 of these sources, recall which colleagues attended which calls, and piece together an answer from memory and keyword searches.

The demo should show the assistant **ingesting all of these disparate sources**, then **answering natural-language questions** that span multiple sources — retrieving the right organisations, citing the source document/call/post, and optionally producing a summary brief.

### 3. Inputs and Outputs

**Inputs (uploaded to the assistant beforehand):**

| # | Artifact | Format | Description |
|---|----------|--------|-------------|
| 1 | One-to-one call notes | `.xlsx` workbook, one sheet per housing association, columns for date, topic tags, self-assessment score (1–5), and free-text notes | ~4 years of records across ~30 organisations in the Carbon Club |
| 2 | Workshop transcripts | `.vtt` or `.txt` transcript files exported from MS Teams | 6–8 transcripts from Carbon Club sessions, each 60–90 min |
| 3 | Zoho Connect portal export | `.docx` or `.txt` file containing Q&A threads | Exported discussion threads on topics like void standards, EV fleet, retrofit technologies |
| 4 | WordPress knowledge hub articles | `.pdf` files or `.html` exports | 10–15 articles/case studies shared by member organisations |
| 5 | OneNote excerpts | `.txt` or `.docx` exports | Ad-hoc notes from renewal calls and member interactions |

**Outputs (produced on demand by the assistant):**

| # | Output | Format | Description |
|---|--------|--------|-------------|
| A | Ranked answer | Chat message | A natural-language answer identifying the top organisations for a given topic, with source citations (e.g., "Mentioned in 1:1 call with Orbit HA, 14 Mar 2025") |
| B | Summary brief | `.docx` or `.pdf` | A 1–2 page formatted brief that a staff member can forward to the requesting member |
| C | Cross-reference table | `.xlsx` | A quick-reference table showing which organisations have mentioned or demonstrated capability in the queried topic |

### 4. Demo Scenario

**Setup:** The assistant ("Olivia") has been pre-loaded with data from Vantage's **Carbon Club**, which has 28 member housing associations focused on decarbonisation — reducing carbon emissions from their housing stock through insulation, solar panels, heat pumps, window replacements, etc.

**The question (typed into the chat):**

> "Which of our Carbon Club members have done the most solar PV installations on their stock? Can you give me a quick summary I can share with Broadland Housing who are asking about it?"

**Expected assistant behaviour:**

1. Search across all ingested sources for mentions of solar PV, photovoltaic, solar panel installations.
2. Return a ranked list citing specific evidence:
   - **Greendale Homes** — Installed PV on 1,200 properties in 2024/25 under SHDF Wave 2. Mentioned in 1:1 call notes (12 Jan 2025, self-score: 4/5) and Carbon Club workshop transcript (14 Feb 2025, timestamp 00:34:12).
   - **Northfield Housing Association** — Rolled out 850 rooftop PV installations with battery storage across three estates in Birmingham. Shared a case study PDF on the knowledge hub (uploaded 20 Nov 2024).
   - **Beacon Dwellings** — Completed 340 PV installations as part of a broader EPC-upgrade programme. Discussed on Zoho Connect portal (thread: "Solar PV costs and procurement", 8 Oct 2024).
3. Generate a 1-page `.docx` brief formatted with Vantage branding header, addressed to Broadland Housing.

#### Artifacts to Generate

**Artifact 1 — Carbon Club One-to-One Notes Spreadsheet (`carbon_club_1to1_notes.xlsx`)**

Create with Python (`openpyxl`). The workbook should have:

- **Sheet per organisation** (28 sheets, named by organisation). Example sheet names: `Greendale Homes`, `Northfield HA`, `Beacon Dwellings`, `Westmoor Housing Group`, `Riverside Comm Housing`, `Broadland Housing`, etc.
- **Columns:** `Date` (dd/mm/yyyy), `Attendees` (Vantage staff + member staff names), `Topic Tags` (semicolon-separated, e.g. `solar PV; EPC upgrades; SHDF funding`), `Self-Score` (integer 1–5, member's self-assessment of their progress on the topic), `Notes` (2–5 sentence free-text summary).
- **Row count:** 6–12 rows per sheet spanning Jan 2022 – Dec 2025.
- Ensure that Greendale Homes, Northfield HA, and Beacon Dwellings have explicit mentions of solar PV in their notes. Other organisations should mention other retrofit topics (loft insulation, cavity wall insulation, heat pumps, window glazing, EWI — external wall insulation) to provide realistic variety.

Example rows for the `Greendale Homes` sheet:

| Date | Attendees | Topic Tags | Self-Score | Notes |
|------|-----------|------------|------------|-------|
| 12/01/2025 | Steph Hosny; Mark Jennings (Greendale) | solar PV; SHDF Wave 2 | 4 | Greendale have completed 1,200 PV installs under SHDF Wave 2. Targeting remaining 600 properties by end of FY 25/26. Procurement through Solarfix Ltd framework. Mark noted tenant satisfaction with PV has been very high — 89% positive in their internal survey. |
| 18/07/2024 | Rob Bryan; Mark Jennings (Greendale) | EPC upgrades; data quality | 3 | Greendale struggling with EPC data accuracy for pre-1940 stock. Working with Elmhurst Energy to resurvey 2,000 properties. Currently 68% of stock at EPC C or above. |
| 03/03/2024 | Steph Hosny; Sarah Patel (Greendale) | loft insulation; contractor management | 4 | Completed loft insulation on 3,100 properties. Using three approved contractors. Quality assurance checks on 10% sample — 96% pass rate. |

**Artifact 2 — Workshop Transcript (`carbon_club_workshop_14feb2025.txt`)**

Create with a Python script that generates a realistic Teams-exported transcript. Format:

```
Carbon Club Workshop — 14 February 2025
Participants: Steph Hosny (Vantage), Rob Bryan (Vantage), Mark Jennings (Greendale Homes),
              Fiona Clarke (Northfield HA), David Osei (Beacon Dwellings), ...

[00:00:15] Steph Hosny: Welcome everyone to the February Carbon Club session. Today we've got
spotlights from Greendale and Northfield on their solar programmes, and then we'll do a
roundtable on SHDF Wave 2 progress.

[00:02:30] Steph Hosny: Mark, do you want to kick us off with the Greendale update?

[00:02:45] Mark Jennings: Sure. So as most of you know, we started our PV programme back in
late 2023. We've now completed 1,200 installations across our general needs stock — mainly
3-bed semis and end-terraces which have the best roof orientation. We've been using 4kW
systems with the Solarfix framework...

[00:05:10] Mark Jennings: The average cost has come in at about £4,200 per property including
scaffolding, which is slightly below our original budget of £4,500. Tenant feedback has been
really positive — we're seeing about 89% satisfaction in our post-install surveys.

...

[00:14:22] Fiona Clarke: Thanks Steph. So Northfield — we took a slightly different approach.
We went with PV plus battery storage from day one. We've done 850 properties across three
estates in south Birmingham — the Selly Oak estate, Kings Heath, and Moseley Gardens.

[00:16:45] Fiona Clarke: The battery storage adds about £1,800 per property on top of the PV
cost, but we're seeing tenants save an additional £150 per year compared to PV-only
installations. The total package is coming in around £6,200 per unit.

...

[00:34:12] David Osei: For Beacon, we're a bit earlier in the journey. We've completed 340
installations so far as part of our wider EPC programme. Our focus has been getting everything
up to EPC C first, so PV is one component alongside cavity wall insulation and new windows.

...
```

Generate 800–1200 lines covering a 75-minute meeting with 8–10 speakers from different organisations. Include natural conversational markers (filler words, interruptions, questions from the floor). Topics should include: solar PV progress, SHDF funding updates, EPC improvement targets, contractor procurement, tenant engagement, data reporting challenges.

**Artifact 3 — Zoho Connect Portal Export (`zoho_portal_threads.docx`)**

Create with Python (`python-docx`). Structure as a series of Q&A threads:

```
Thread: "Solar PV costs and procurement" — Posted 8 October 2024

[Original Post — Lena Morris, Broadland Housing]
Hi all, we're looking to start a solar PV programme next financial year and would appreciate
any insights on procurement routes and typical costs per unit. We're looking at about 500
properties initially. Has anyone used a framework agreement?

    [Reply — David Osei, Beacon Dwellings — 9 Oct 2024]
    Hi Lena, we've been using a mix of direct procurement and the Fusion21 framework.
    Costs have been around £4,000-£4,500 per property for a 4kW system...

    [Reply — Mark Jennings, Greendale Homes — 10 Oct 2024]
    We went through Solarfix Ltd framework — very smooth procurement process...

---

Thread: "Void standards and turnaround thresholds" — Posted 22 November 2024

[Original Post — Karen Blackwell, Westmoor Housing Group]
Does anyone have documented void standards they'd be willing to share? We're reviewing
our lettable standard and trying to benchmark our turnaround times...

    [Reply — Fiona Clarke, Northfield HA — 23 Nov 2024]
    Our current target is 18 calendar days key-to-key for standard voids...

    [Reply — James Thornton, Riverside Comm Housing — 24 Nov 2024]
    We're at 22 days average currently but targeting 16 by end of FY...
```

Include 6–8 threads covering: solar PV, void standards, EV fleet, heat pump experiences, EPC data quality, SHDF funding applications, tenant communications for retrofit works, and damp & mould management.

**Artifact 4 — Knowledge Hub Articles (folder of PDFs)**

Generate 3–4 simple PDF documents using Python (`reportlab` or `fpdf2`) that look like member-submitted case studies. Each should be 2–3 pages with:

- A header with the housing association's logo placeholder (coloured rectangle + org name)
- Title (e.g., "Northfield Housing Association — Solar PV with Battery Storage: A Case Study")
- Sections: Background, Approach, Results, Lessons Learned, Key Metrics table
- A table with metrics: number of properties, cost per unit, annual CO₂ savings, tenant satisfaction %, EPC improvement (before/after)

**Artifact 5 — OneNote Excerpts (`onenote_renewal_notes.txt`)**

Plain text file simulating exported OneNote snippets:

```
=== Renewal Call Notes ===

--- Broadland Housing — 5 September 2024 ---
Attendees: Rob Bryan, Lena Morris (Director of Assets, Broadland)
- Broadland very interested in starting a PV programme — asked for contacts at orgs
  who have done it at scale
- Renewed for Carbon Club + Exec Club (2-year deal, £8,500/yr)
- Lena mentioned they're also looking at air source heat pumps for off-gas properties
- ACTION: Steph to send examples of PV case studies from other members

--- Westmoor Housing Group — 12 September 2024 ---
Attendees: Steph Hosny, Karen Blackwell (Head of Sustainability, Westmoor)
- Renewed Carbon Club only this year (£4,200/yr)
- Karen flagged concerns about void standards — their turnaround times have increased
  to 28 days avg and board is asking questions
- Interested in benchmarking void costs against peers
- ACTION: Share void standards thread from Zoho portal
...
```

Include 8–10 entries spanning 2024–2025.

### 5. Realism Notes

- **TSM data** is published annually by the Regulator of Social Housing in Excel format and includes perception survey scores and management metrics across ~500,000 tenant responses. The 2023/24 dataset was the first mandatory release ([GOV.UK TSM data](https://www.gov.uk/government/statistics/tenant-satisfaction-measures-202324)).
- **Void turnaround benchmarks**: The sector average is 20 days for housing associations per HouseMark data. Costs of £8,400 per void period for properties taking >30 days are documented.
- **Solar PV in social housing**: Together Housing Group's 250-property pilot and Sentinel HA's 337-home programme provide realistic cost/savings benchmarks (£4,000–£6,200 per unit; £150–£200 annual tenant savings). These informed the demo figures.
- **SHDF (Social Housing Decarbonisation Fund)**: A UK Government programme that has delivered ~33,100 retrofit measures across ~17,300 households in Waves 1 and 2. Insulation measures account for 59–62% of installations.
- **Vantage's actual club structure** is documented at [yourvantage.co.uk](https://yourvantage.co.uk) — the Performance Improvement Club has a 90% renewal rate and works with organisations like Midland Heart, Accent, Longhurst, and Great Places.
- **Excel-based one-to-one tracking** with self-scoring is a common practice in UK housing consultancy benchmarking — Acuity Benchmarking and HouseMark use similar approaches.

---

## Workflow 2 — Financial and Non-Financial Data Triangulation

### 1. Transcript Citations

> **Robert Bryan (04:01):** *"We've moved into collecting non-financial data. And one of the areas that we just started to contemplate and think well I wonder how AI could support is around that data analytics. At the moment we tend to use simple like templates, Excel templates."*

> **Robert Bryan (05:04):** *"We pick up lots of TSM information. The regulator publishes a data set on that each November with housing. We then go off and do analysis for individual organisations from some of that. But we'd quite like to look at the financial data that we gather, the non-financial data that we gather across all these organisations."*

> **Robert Bryan (05:18):** *"We're like normally we'd be in well okay, let's look at say Microsoft BI, but that seems quite expensive. But what we'd need, we can obviously do modelling in Excel, which is fine but I'm becoming steadily more old-fashioned I guess."*

> **Stephanie Hosny (26:14):** *"They're just notes in an Excel spreadsheet… I think we've probably got about four years worth of meetings in there as well. People, they score themselves, you see. And then we track their progress and then we look at the trends and things."*

### 2. Plain-English Explanation

Vantage collects three categories of data from its member housing associations:

1. **Financial data** — Key financial metrics like cost per housing unit, operating margin (revenue minus costs as a percentage of revenue), gearing ratio (how much debt the organisation carries relative to its assets), and reinvestment percentage (how much is being ploughed back into properties). This data comes partly from the Regulator of Social Housing's public "Value for Money" (VfM) benchmarking tool (a large Excel workbook published annually by the UK government) and partly from data that members submit directly to Vantage.

2. **TSM (Tenant Satisfaction Measures) data** — A standardised set of survey results that every large housing association in England must now report to the Regulator. Covers satisfaction with repairs, complaint handling, safety, communication, and overall service. Published once a year (in November) as a downloadable Excel file from GOV.UK.

3. **Non-financial / operational data** — Things like void turnaround times, number of retrofit installations completed, EPC ratings of housing stock, complaint volumes, anti-social behaviour cases, and similar operational KPIs that Vantage collects via its own surveys and workshops.

Today, Vantage analyses each of these in isolation using Excel. The pain is that they cannot easily **triangulate** — for example, answering "Which organisations have the highest tenant satisfaction AND the lowest cost per unit AND the most retrofit activity?" requires manually cross-referencing three separate spreadsheets.

The demo should show the assistant loading all three data categories, performing cross-dataset queries, and producing benchmarking reports that compare organisations across financial, satisfaction, and operational dimensions simultaneously.

### 3. Inputs and Outputs

**Inputs:**

| # | Artifact | Format | Description |
|---|----------|--------|-------------|
| 1 | VfM financial data | `.xlsx` workbook | Modelled on the Regulator's Value for Money benchmarking tool — one row per housing association, columns for key financial metrics across 2–3 financial years |
| 2 | TSM satisfaction data | `.xlsx` workbook | Modelled on the Regulator's TSM data release — one row per housing association, columns for each satisfaction measure (percentage scores) |
| 3 | Operational KPI data | `.xlsx` workbook | Vantage's own collection — one row per housing association per year, columns for void turnaround days, EPC % at Band C+, retrofit units completed, complaints per 1,000 homes, etc. |

**Outputs:**

| # | Output | Format | Description |
|---|--------|--------|-------------|
| A | Triangulated benchmarking report | `.xlsx` with charts | A workbook combining all three data sources with an organisation-level dashboard: financial health, tenant satisfaction, and operational performance side-by-side |
| B | Peer comparison narrative | Chat message or `.docx` | A plain-English narrative for a specific organisation showing how they compare to the club average and top/bottom quartiles |
| C | Trend analysis | Chat message with embedded table | Year-over-year trend for a specific metric across all organisations |

### 4. Demo Scenario

**Setup:** Olivia has been loaded with financial, TSM, and operational data for 28 Carbon Club member organisations.

**The question:**

> "Can you compare Broadland Housing's overall performance against the Carbon Club average? I want to see their financials, tenant satisfaction, and decarbonisation progress side by side. We've got a renewal call with them next week and I want to show them where they stand."

**Expected assistant behaviour:**

1. Pull Broadland Housing's row from each of the three datasets.
2. Calculate the Carbon Club average (mean and median) for each metric.
3. Produce a comparison table and a short narrative.
4. Optionally generate an Excel report with conditional formatting (green/amber/red) showing where Broadland is above/below the club average.

#### Artifacts to Generate

**Artifact 6 — VfM Financial Data (`vfm_financial_data.xlsx`)**

Create with Python (`openpyxl`). Single sheet with:

- **Rows:** 28 housing associations (same names as Workflow 1)
- **Columns:**

| Column | Description | Realistic Range |
|--------|-------------|-----------------|
| `Organisation` | Name | — |
| `Total Units` | Number of homes managed | 3,000 – 45,000 |
| `Headline Social Housing Cost Per Unit (£) 2024/25` | Operating cost per home | £3,200 – £7,500 |
| `Headline Social Housing Cost Per Unit (£) 2023/24` | Prior year comparison | £3,000 – £7,000 |
| `Reinvestment % 2024/25` | Capital spending as % of property value | 4.0% – 12.0% |
| `Operating Margin (Social Housing) % 2024/25` | Profit margin on social housing activities | 15% – 35% |
| `Operating Margin (Overall) % 2024/25` | Profit margin across all activities | 10% – 30% |
| `Gearing % 2024/25` | Debt / total assets | 30% – 70% |
| `EBITDA MRI Interest Cover % 2024/25` | Earnings coverage of interest payments | 120% – 300% |
| `Return on Capital Employed % 2024/25` | Efficiency of capital use | 2.0% – 6.0% |

Use the Regulator's published median figures as anchors: median cost per unit of £5,136, median reinvestment of 7.7%, median operating margin of ~23%.

Broadland Housing should be slightly below average on cost efficiency (£5,800/unit) but above average on reinvestment (9.2%) — telling a story that they're spending more because they're investing heavily.

**Artifact 7 — TSM Satisfaction Data (`tsm_satisfaction_data.xlsx`)**

Create with Python (`openpyxl`). Single sheet with:

- **Rows:** Same 28 organisations
- **Columns** (all percentages):

| Column | Sector Median |
|--------|--------------|
| `Overall Satisfaction` | 70% |
| `Satisfaction with Repairs` | 72% |
| `Satisfaction with Time Taken for Repairs` | 65% |
| `Home Safety` | 77% |
| `Kept Informed` | 68% |
| `Fair & Respectful Treatment` | 77% |
| `Complaint Handling Satisfaction` | 35% |
| `Anti-Social Behaviour Handling` | 55% |
| `Contribution to Neighbourhood` | 60% |
| `Well-Maintained Home` | 65% |
| `Well-Maintained Communal Areas` | 58% |

Broadland Housing: Overall 72%, Repairs 74%, Complaint Handling 31% (below average — a coaching point for the renewal call).

**Artifact 8 — Operational KPI Data (`operational_kpi_data.xlsx`)**

Create with Python (`openpyxl`). Single sheet with:

- **Rows:** 28 organisations, with FY columns for 2023/24 and 2024/25
- **Columns:**

| Column | Realistic Range |
|--------|-----------------|
| `Void Turnaround (days)` | 14 – 35 |
| `Void Rent Loss (%)` | 0.8% – 3.5% |
| `Stock at EPC C or above (%)` | 55% – 92% |
| `Retrofit Measures Installed (count)` | 50 – 3,500 |
| `PV Installations (count)` | 0 – 1,500 |
| `Complaints per 1,000 homes` | 25 – 120 |
| `EV Charging Points Installed` | 0 – 80 |
| `Damp & Mould Cases (per 1,000 homes)` | 5 – 45 |

Broadland Housing: Void turnaround 24 days, EPC C+ 63% (below avg), PV installations 0 (they're asking about starting), retrofit measures 280.

### 5. Realism Notes

- **VfM benchmarking tool**: The Regulator publishes a 4+ MB Excel workbook annually with financial metrics for all registered providers. The 2024 version was published March 2025. Column names and median values used here are drawn directly from the published tool ([GOV.UK VfM 2024](https://www.gov.uk/government/publications/value-for-money-2024-benchmarking-tool)).
- **TSM data**: The first mandatory TSM data release covered 2023/24. Survey categories and median satisfaction percentages (70% overall, 35% complaint handling, 77% safety) are from the published dataset ([GOV.UK TSM 2023/24](https://www.gov.uk/government/statistics/tenant-satisfaction-measures-202324)).
- **Vantage's Global Accounts Plus** product already provides web-based benchmarking of the top 200 registered providers — this demo mirrors that value proposition but with AI-driven cross-dataset analysis.
- **Operational KPI ranges** are informed by HouseMark benchmarking data and the SHDF statistics (e.g., retrofit measure counts, EPC distribution data from the English Housing Survey 2024/25).

---

## Workflow 3 — Member Management & Engagement Tracking

### 1. Transcript Citations

> **Robert Bryan (40:01):** *"We've got about 90 organisations that are working with us but they're — most of them are in multiple clubs. Right. So you might have an average of — let's turn around and say for simple terms there's 250 memberships that we've got across 90 members. We then probably have an average of about four colleagues for each of those clubs. So quite quickly it could be into like a thousand individuals."*

> **Robert Bryan (40:38):** *"Quite often we're getting asked the question of like, you know, well, remind me again, what club am I in? Or like, you know, who's our stakeholder for this?"*

> **Robert Bryan (41:07):** *"Our default position would have been — actually let's go and get ourselves a CRM option for dealing with the membership management stuff. When's the renewal due? Did they respond to this when you email them that type of stuff? That's all basically manual and through spreadsheets at the moment."*

> **Stephanie Hosny (43:33):** *"We've had a session this morning with like 50 members. I've now got to go into our spreadsheet and log who came. And then when I look at the sort of the RAG every month, you know who's not coming? Are they engaging?"*

> **Stephanie Hosny (20:10):** *"When we have an email exchange with a client because at the moment they're kind of coming to me going have you got this data or can you get some examples? And I'm doing that and that's easy to forget that you've done that. So we were like creating a bit of a manual log… So there's lots of information that I would then want to retrieve when I'm having like a renewal call."*

### 2. Plain-English Explanation

Vantage manages memberships manually in spreadsheets. Each of their ~90 client organisations may belong to multiple clubs (e.g., Carbon Club + Executive Club + Finance Directors Club). Each organisation sends ~4 different individuals to attend sessions. This creates a complex matrix: ~250 organisation-to-club memberships, ~1,000 individual contacts.

The day-to-day pain points:

- **Attendance logging**: After every workshop (which can have 50+ attendees), someone must manually open a spreadsheet and tick off who attended.
- **Engagement RAG (Red-Amber-Green) scoring**: Monthly, staff review each membership to score engagement — are they attending sessions? Asking questions? Participating in one-to-ones? A "Red" score means the member is disengaging and may not renew.
- **Renewal tracking**: Memberships renew annually. Staff need to know when each renewal is due, what the fee is, whether the member has been engaged, and what value Vantage has delivered to them (so they can justify the renewal in a call).
- **Interaction history**: When staff email data or examples to a member, this goes unlogged. During renewal calls, staff want to recall everything they've done for that member — but it's scattered across email, spreadsheets, and memory.

The demo should show the assistant automatically logging attendance from a Teams meeting participant list, maintaining a live engagement dashboard, and preparing a renewal brief that summarises all interactions with a given member.

### 3. Inputs and Outputs

**Inputs:**

| # | Artifact | Format | Description |
|---|----------|--------|-------------|
| 1 | Membership master spreadsheet | `.xlsx` | All 90 organisations, their club memberships, renewal dates, annual fees, key contacts (name, role, email), assigned Vantage account manager |
| 2 | Attendance logs | `.xlsx` | Historical attendance records per club session (date, session title, attendee names) — partial data, some gaps |
| 3 | Teams meeting participant list | `.csv` or plain text extracted from Teams | Output from a just-completed session listing participant names and join/leave times |
| 4 | Email interaction log | `.txt` or `.docx` | Manually maintained log of data requests, examples sent, follow-ups |
| 5 | Renewal call notes | `.txt` | Exported from OneNote — past renewal conversation summaries |

**Outputs:**

| # | Output | Format | Description |
|---|--------|--------|-------------|
| A | Updated attendance log | `.xlsx` (updated in place) | Attendance column auto-filled for the latest session by matching participant names to the membership database |
| B | Monthly engagement RAG | Chat message or `.xlsx` | RAG-scored table showing engagement status per membership: Green (attended 2+ sessions, active in portal, had 1:1), Amber (some activity), Red (no attendance or interaction in 2+ months) |
| C | Renewal preparation brief | `.docx` | A 1-page brief for a specific organisation's upcoming renewal: membership history, attendance record, data/examples provided, 1:1 notes summary, recommended talking points |

### 4. Demo Scenario

**Setup:** Olivia has the full membership spreadsheet and historical attendance data. A Carbon Club workshop just ended and the Teams participant list is available.

**Interaction 1 — Attendance logging:**

> "Hi Olivia, we just finished the Carbon Club workshop. Here's the Teams attendee list. Can you update the attendance log?"

The assistant receives a participant list like:

```
Meeting: Carbon Club Workshop — 5 March 2025
Duration: 10:00 – 11:15 GMT

Participants:
- Steph Hosny (Vantage) — joined 09:58, left 11:15
- Mark Jennings (Greendale Homes) — joined 10:01, left 11:14
- Fiona Clarke (Northfield HA) — joined 10:00, left 11:15
- Lena Morris (Broadland Housing) — joined 10:05, left 10:58
- Karen Blackwell (Westmoor Housing Group) — joined 10:00, left 11:15
- David Osei (Beacon Dwellings) — joined 10:02, left 11:12
- James Thornton (Riverside Comm Housing) — joined 10:08, left 11:15
- Priya Sharma (Oaktree Living) — joined 10:00, left 11:10
- Tom Henderson (Millbrook Homes) — joined 10:03, left 11:15
...
```

The assistant matches each participant to the membership database and updates the attendance spreadsheet.

**Interaction 2 — Engagement RAG:**

> "Can you give me the engagement RAG for Carbon Club this month?"

Expected output table:

| Organisation | Attendance (last 3 months) | Portal Activity | Last 1:1 | Data Requests Served | RAG |
|---|---|---|---|---|---|
| Greendale Homes | 3/3 sessions | 2 posts | 12 Jan 2025 | 1 | 🟢 Green |
| Northfield HA | 3/3 sessions | 4 posts | 3 Feb 2025 | 2 | 🟢 Green |
| Broadland Housing | 2/3 sessions | 1 post | 5 Sep 2024 | 3 | 🟡 Amber |
| Riverview Estates | 0/3 sessions | 0 posts | 14 Jun 2024 | 0 | 🔴 Red |
| Westmoor Housing Group | 3/3 sessions | 3 posts | 12 Sep 2024 | 1 | 🟢 Green |
| ... | ... | ... | ... | ... | ... |

**Interaction 3 — Renewal brief:**

> "Broadland Housing's renewal is coming up on 1 April. Can you prepare a brief for my call with Lena Morris?"

Expected output: A `.docx` with sections:

- **Organisation Summary**: Broadland Housing, 4,200 homes, based in Norwich. Member since 2021. Current subscriptions: Carbon Club (£4,200/yr), Exec Club (£4,300/yr). Renewal date: 1 April 2025. Total annual fee: £8,500.
- **Engagement Summary**: Attended 8 of 12 Carbon Club sessions in 2024/25. Participated in 2 Zoho portal threads. Had 2 one-to-one calls (Sep 2024, Jan 2025).
- **Value Delivered**: Shared 3 data packs (void benchmarking data, PV case studies from Greendale & Northfield, SHDF funding guidance). Connected Lena with Mark Jennings (Greendale) for a PV peer visit.
- **Coaching Points**: Complaint handling satisfaction (31%) is below club average (35%) — opportunity to discuss improvement strategies. EPC C+ stock (63%) below club average (72%) — main driver for PV interest.
- **Recommended Discussion**: Broadland's interest in starting a PV programme is a strong retention hook. Emphasise Carbon Club's value in connecting them with experienced peers.

#### Artifacts to Generate

**Artifact 9 — Membership Master Spreadsheet (`membership_master.xlsx`)**

Create with Python (`openpyxl`). Two sheets:

**Sheet 1: "Organisations"** — 90 rows

| Column | Example |
|--------|---------|
| `Org ID` | ORG-001 |
| `Organisation Name` | Broadland Housing |
| `Total Homes` | 4,200 |
| `HQ Location` | Norwich |
| `Member Since` | 2021 |
| `Account Manager` | Steph Hosny |
| `Primary Contact` | Lena Morris |
| `Primary Contact Role` | Director of Assets |
| `Primary Contact Email` | l.morris@broadlandhousing.org.uk |

**Sheet 2: "Memberships"** — 250 rows

| Column | Example |
|--------|---------|
| `Org ID` | ORG-001 |
| `Organisation Name` | Broadland Housing |
| `Club` | Carbon Club |
| `Membership Fee (£/yr)` | 4,200 |
| `Renewal Date` | 01/04/2025 |
| `Contract Length` | 1 year |
| `Status` | Active |

**Sheet 3: "Contacts"** — ~400 rows (subset of the 1,000 — generate enough for Carbon Club)

| Column | Example |
|--------|---------|
| `Org ID` | ORG-001 |
| `Name` | Lena Morris |
| `Role` | Director of Assets |
| `Email` | l.morris@broadlandhousing.org.uk |
| `Club(s)` | Carbon Club; Exec Club |
| `Phone` | 01603 555 0142 |

Use realistic UK housing association names (mix of invented ones like "Greendale Homes", "Beacon Dwellings", "Millbrook Homes" with realistic-sounding names like "Northfield HA", "Westmoor Housing Group", "Riverside Community Housing", "Oaktree Living", "Pennine Valleys Housing", "Severn Vale Homes"). Generate realistic Norwich, Birmingham, Manchester, Leeds, London, Bristol postcodes.

**Artifact 10 — Attendance Log (`attendance_log.xlsx`)**

Create with Python (`openpyxl`). One sheet per club (start with Carbon Club only for the demo). Columns:

| Column | Description |
|--------|-------------|
| `Session Date` | dd/mm/yyyy |
| `Session Title` | e.g., "Carbon Club Workshop — February 2025" |
| Then one column per member organisation | Cell value: name of attendee, or blank if absent |

Generate 12 sessions spanning April 2024 – March 2025. Ensure Riverview Estates has zero attendance in the last 3 months (Red RAG). Broadland Housing should have 2/3 attendance in the latest quarter.

**Artifact 11 — Teams Participant List (`teams_participants_5mar2025.txt`)**

Plain text file as shown in the interaction example above. List 18–22 participants with realistic join/leave times.

**Artifact 12 — Email Interaction Log (`email_interaction_log.xlsx`)**

Create with Python (`openpyxl`). Single sheet:

| Column | Example |
|--------|---------|
| `Date` | 15/10/2024 |
| `Organisation` | Broadland Housing |
| `Contact` | Lena Morris |
| `Direction` | Outbound |
| `Subject` | Solar PV case studies — Greendale & Northfield |
| `Summary` | Sent PDF case studies for Greendale (1,200 installs) and Northfield (850 installs with battery). Lena requested these after Sep 1:1. |
| `Vantage Staff` | Steph Hosny |

Generate 30–40 rows spanning 2024–2025 across 15+ organisations.

### 5. Realism Notes

- **Membership pricing**: Vantage's Performance Improvement Club and similar clubs typically charge £4,000–£8,500/year per membership, based on the size of the housing association. This is consistent with the figures Robert mentions in the transcript.
- **RAG scoring for engagement** is a standard practice in UK membership organisations and housing consultancies. HouseMark and Acuity Benchmarking use similar engagement tracking.
- **Teams meeting participant exports**: Microsoft Teams allows meeting organisers to download a CSV of participants with join/leave timestamps. This is a standard feature used across the sector.
- **Zoho Connect** is used by Vantage as a community portal — the Q&A and discussion thread format described here matches Zoho Connect's actual "Questions" and "Forums" features.
- **Renewal call briefs**: Membership-based organisations commonly prepare one-page "relationship summaries" before renewal conversations. This is analogous to account review documents used in SaaS and professional services.

---

## Workflow 4 — Automated Meeting Attendance & Post-Session Admin

### 1. Transcript Citations

> **Stephanie Hosny (43:33):** *"We've had a session this morning with like 50 members. I've now got to go into our spreadsheet and log who came. And then when I look at the sort of the RAG every month, you know who's not coming? Are they engaging? You know, I'm checking across. If they come to this session then there's a working group log as well."*

> **Robert Bryan (11:47):** *"We are now gathering, you know, for each of the calls, we are now using Teams. All of our clients use Teams, but we're using the Teams transcript as well."*

> **Daniel Lenton (44:00):** *"She'll remember that as a skill and next time she'll be able to just join the meeting, listen in and then fill in the spreadsheet for you. It's done in 30 seconds."*

> **Robert Bryan (12:04):** *"Each time she's getting them, she's getting all — in all the one-to-one feedback sessions she's getting all that. She's also potentially going to get the financial data and the quantitative data as well."*

### 2. Plain-English Explanation

After every workshop or group call, a Vantage staff member must:

1. Open the attendance spreadsheet for that club.
2. Manually check which members attended (either from memory, the Teams participant list, or the chat).
3. Mark attendance in the correct cells.
4. Check if anyone attended who isn't in the membership database (a guest or new contact).
5. Optionally, update the engagement RAG for any member whose attendance pattern has changed.

This is done after every session — with 6 clubs running multiple sessions per month, it adds up to significant admin overhead. The Teams transcript from the session is also available but currently goes unused beyond ad-hoc reference.

The demo should show the assistant receiving a Teams meeting transcript and participant list, automatically extracting attendance, summarising key discussion topics and any action items, and updating the relevant spreadsheets — all triggered by a single message like "Process the Carbon Club session from this morning."

### 3. Inputs and Outputs

**Inputs:**

| # | Artifact | Format | Description |
|---|----------|--------|-------------|
| 1 | Teams meeting transcript | `.vtt` or `.txt` | Full transcript of the session |
| 2 | Teams participant list | `.csv` or `.txt` | Names and join/leave times |
| 3 | Existing attendance log | `.xlsx` | The spreadsheet to be updated |
| 4 | Membership master | `.xlsx` | To match participant names to organisations |

**Outputs:**

| # | Output | Format | Description |
|---|--------|--------|-------------|
| A | Updated attendance log | `.xlsx` | New column/row filled in for this session |
| B | Session summary | Chat message or `.docx` | 1-page summary: topics discussed, key takeaways per organisation, action items |
| C | Engagement alerts | Chat message | Flags: "Riverview Estates has now missed 3 consecutive sessions — suggest reaching out" |
| D | New contact alert | Chat message | "Priya Sharma (Oaktree Living) attended for the first time — not in contacts database. Should I add her?" |

### 4. Demo Scenario

**The trigger message:**

> "Olivia, here's the transcript and attendee list from this morning's Carbon Club session. Can you update the attendance log, give me a summary of what was discussed, and flag anyone I should be worried about?"

**Expected assistant behaviour:**

1. Parse the participant list and match names to the membership database.
2. Update the attendance log spreadsheet (Artifact 10) — add a new column for "5 Mar 2025" and populate it.
3. Parse the transcript to extract:
   - **Topics discussed**: Solar PV progress updates (Greendale, Northfield), SHDF Wave 2 funding deadlines, EPC data quality challenges, tenant communication best practices for retrofit works.
   - **Action items**: "Steph to circulate Greendale's PV procurement framework details to interested members", "Fiona (Northfield) to share battery storage cost-benefit analysis with the group", "Broadland requested introduction to Greendale for a peer visit".
   - **Key quotes**: "Mark Jennings (Greendale): 'The average cost has come in at about £4,200 per property including scaffolding' — useful benchmark for members starting PV programmes."
4. Flag engagement concerns: "Riverview Estates has not attended the last 3 Carbon Club sessions (Jan, Feb, Mar 2025). Last one-to-one was June 2024. Recommend outreach."
5. Flag new contacts: "Tom Henderson (Millbrook Homes) — first time attending Carbon Club. He is in the membership database under Exec Club. Should I add Carbon Club to his profile?"

#### Artifacts to Generate

All artifacts for this workflow are shared with Workflows 1 and 3 (Artifacts 2, 10, 11, and 9). No additional artifacts are needed — this workflow demonstrates how the same data sources are used in an event-triggered automation context rather than an on-demand query context.

Additionally, generate:

**Artifact 13 — Session Summary Template (`session_summary_template.docx`)**

Create with Python (`python-docx`). A formatted template showing what a post-session summary looks like:

- **Header**: Vantage logo placeholder, "Carbon Club — Session Summary"
- **Date**: 5 March 2025
- **Attendees**: List of attendees with organisations (18 of 28 members represented)
- **Apologies / Absent**: List of organisations not represented
- **Discussion Summary**: 3–5 bullet points per agenda item
- **Action Items**: Table with columns: Action, Owner, Deadline
- **Engagement Notes**: Any flags for follow-up

### 5. Realism Notes

- **Teams transcript format**: Microsoft Teams generates `.vtt` (WebVTT) transcript files that include timestamps and speaker identification. Many UK housing associations use Teams as their primary meeting platform.
- **Post-meeting admin in membership organisations** is a well-documented pain point. Industry bodies like the Chartered Institute of Housing (CIH) and National Housing Federation (NHF) face similar challenges with event attendance tracking.
- **RAG-based engagement scoring** is standard across UK housing sector benchmarking — HouseMark's reporting tools use RAG dashboards extensively.

---

## Workflow 5 — Portal Data Extraction & Synthesis

### 1. Transcript Citations

> **Stephanie Hosny (22:13):** *"Stuff on the portal. So we use a Zoho Connect portal. Members can ask questions of each other. So the other week for example people were asking about void standards and thresholds and things. So I took, you know, maybe there's like 10. I exported all that data. I just put it in a spreadsheet [or] Word document actually and shared it with a few people that have been asking about void standards."*

> **Stephanie Hosny (26:21):** *"I'm not sure on the portal, though, because I don't think it can be exported cleanly."*

> **Daniel Lenton (27:04):** *"In terms of things being gated in a portal where there's no way to export the data easily, that actually isn't a problem because [Olivia has a computer]. Just as one example of the kind of workflow that's in place right now with Colliers, they need to go through this gated website, they pay quite a large subscription for the access to the data… Now Olivia can do that because again, Olivia has mouse, keyboard, computer."*

### 2. Plain-English Explanation

Vantage uses a **Zoho Connect portal** — an online community platform (similar to a private forum) where member housing associations can post questions, share documents, and reply to each other's queries. When a topic gets discussed on the portal (e.g., "What are your void standards?"), Steph currently has to:

1. Go into the portal and manually read through all replies (sometimes 10+ responses across multiple threads).
2. Copy-paste the relevant information into a Word document or spreadsheet.
3. Clean it up and format it.
4. Email the synthesised document to the members who asked.

The portal does not export data cleanly — there's no "download all responses as CSV" button. This is the same class of problem that Colliers faces with gated websites: the data is behind a login, in a web interface, with no easy export.

The demo should show the assistant navigating the Zoho Connect portal (or a simulated version), extracting all responses from a given discussion thread, and producing a formatted summary document.

### 3. Inputs and Outputs

**Inputs:**

| # | Artifact | Format | Description |
|---|----------|--------|-------------|
| 1 | Zoho Connect portal thread | Web page (HTML) or pre-exported `.docx` | A discussion thread with 8–12 responses from different housing associations on a specific topic |
| 2 | User instruction | Chat message | "Can you pull together the responses from the void standards thread and send me a summary?" |

**Outputs:**

| # | Output | Format | Description |
|---|--------|--------|-------------|
| A | Synthesised summary document | `.docx` | A formatted document with: topic header, each organisation's response summarised in 2–3 sentences, a comparison table of key metrics mentioned, and a "common themes" section |
| B | Quick-reference table | `.xlsx` | A structured spreadsheet extracting any quantifiable data from the responses (e.g., void turnaround targets in days, lettable standard descriptions) |

### 4. Demo Scenario

**Setup:** A thread on the Zoho Connect portal about "Void Standards and Turnaround Thresholds" has received 10 responses over the past two weeks.

**The question:**

> "Olivia, can you go through the void standards thread on the portal and put together a summary for me? Karen at Westmoor has been asking for it."

**Expected assistant behaviour:**

1. Access the portal content (either by navigating the Zoho Connect web interface, or from a pre-exported document).
2. Extract each organisation's response, identifying: their current void turnaround target, their actual performance, their lettable standard description, and any notable practices.
3. Produce a `.docx` with:

**Void Standards & Turnaround Thresholds — Summary of Member Responses**
*Compiled from Zoho Connect Portal, November 2024*

| Organisation | Target Turnaround (days) | Actual Turnaround (days) | Lettable Standard | Notable Practice |
|---|---|---|---|---|
| Northfield HA | 18 | 16 | Enhanced (includes full redecoration) | Pre-void inspections 4 weeks before tenancy end |
| Riverside Comm Housing | 16 | 22 | Basic (safety checks + clean only) | Piloting "void packs" with new tenant welcome kits |
| Greendale Homes | 20 | 19 | Standard (safety + essential repairs) | Dedicated void contractor team on 5-day SLA |
| Beacon Dwellings | 15 | 21 | Enhanced | Using Plentific platform for void works scheduling |
| Oaktree Living | 20 | 25 | Basic | Struggling with contractor availability — exploring in-house team |
| Millbrook Homes | 18 | 17 | Standard | Integrated void and lettings process — single team handles end-to-end |
| Pennine Valleys Housing | 22 | 24 | Basic | Recently tightened from 28-day target after board scrutiny |
| Severn Vale Homes | 16 | 15 | Enhanced | Top performer — attribute success to in-house DLO (Direct Labour Organisation) |

**Common Themes:**
- Organisations with in-house maintenance teams consistently outperform those relying on external contractors.
- "Enhanced" lettable standards (full redecoration, new flooring) add 3–5 days to turnaround but improve tenant satisfaction.
- Several organisations mentioned pre-void inspections as a key improvement lever.

#### Artifacts to Generate

**Artifact 14 — Zoho Connect Portal Thread (HTML mockup or `.docx`)**

This is an extension of Artifact 3 but with more detailed responses in the void standards thread. Create with Python (`python-docx`) or as a static HTML file. Include 8–10 responses with 3–5 sentences each, mentioning specific numbers, practices, and challenges. Each response should include the poster's name, organisation, and date.

**Artifact 15 — Void Standards Summary Document (`void_standards_summary.docx`)**

Create with Python (`python-docx`). This is the expected output that the assistant should produce — serves as a reference/golden example for demo validation. Include:

- Vantage header/branding placeholder
- Title and compilation date
- The comparison table shown above
- A "Common Themes" narrative section (3–4 bullet points)
- A footer: "Prepared by the Vantage Digital Assistant. Data sourced from Zoho Connect Portal discussions, November 2024."

### 5. Realism Notes

- **Void turnaround times**: The UK sector average is 20 days for housing associations, 23 days for local authorities (HouseMark 2019/20). The range of 15–25 days in the demo is realistic. Organisations taking >30 days face average costs of £8,400 per void period.
- **Lettable standards**: UK housing associations define "lettable standards" — the minimum condition a property must be in before a new tenant moves in. These range from "basic" (safety checks and clean) to "enhanced" (full redecoration, new flooring, garden clearance). The Decent Homes Standard and the Social Housing Regulation Act 2023 set minimum requirements.
- **Zoho Connect**: Vantage's use of Zoho Connect as a member community portal is confirmed in the transcript. Zoho Connect's "Questions" feature allows threaded Q&A with voting, and "Forums" support longer-form discussions. Data export from Zoho Connect is limited — there is no built-in "export all replies" feature, confirming Steph's comment about clean exports being difficult.
- **Direct Labour Organisations (DLOs)**: Several UK housing associations run in-house maintenance teams rather than outsourcing. This is a common discussion topic in sector benchmarking.

---

## Artifact Generation Summary

| # | Artifact | Generation Method | Key Libraries |
|---|----------|------------------|---------------|
| 1 | `carbon_club_1to1_notes.xlsx` | Python script | `openpyxl` |
| 2 | `carbon_club_workshop_14feb2025.txt` | Python script | Built-in string formatting |
| 3 | `zoho_portal_threads.docx` | Python script | `python-docx` |
| 4 | Knowledge hub PDFs (3–4 files) | Python script | `fpdf2` or `reportlab` |
| 5 | `onenote_renewal_notes.txt` | Python script | Built-in string formatting |
| 6 | `vfm_financial_data.xlsx` | Python script | `openpyxl` |
| 7 | `tsm_satisfaction_data.xlsx` | Python script | `openpyxl` |
| 8 | `operational_kpi_data.xlsx` | Python script | `openpyxl` |
| 9 | `membership_master.xlsx` | Python script | `openpyxl` |
| 10 | `attendance_log.xlsx` | Python script | `openpyxl` |
| 11 | `teams_participants_5mar2025.txt` | Python script | Built-in string formatting |
| 12 | `email_interaction_log.xlsx` | Python script | `openpyxl` |
| 13 | `session_summary_template.docx` | Python script | `python-docx` |
| 14 | `void_standards_portal_thread.docx` | Python script | `python-docx` |
| 15 | `void_standards_summary.docx` | Python script | `python-docx` |

### Shared Organisation Names (use consistently across all artifacts)

To ensure cross-artifact consistency, use the following 28 organisations for the Carbon Club:

| # | Organisation Name | HQ City | Total Homes |
|---|---|---|---|
| 1 | Greendale Homes | Nottingham | 12,500 |
| 2 | Northfield Housing Association | Birmingham | 8,200 |
| 3 | Beacon Dwellings | Coventry | 5,800 |
| 4 | Broadland Housing | Norwich | 4,200 |
| 5 | Westmoor Housing Group | Sheffield | 9,400 |
| 6 | Riverside Community Housing | Manchester | 15,300 |
| 7 | Oaktree Living | Leeds | 6,100 |
| 8 | Millbrook Homes | Bristol | 7,600 |
| 9 | Pennine Valleys Housing | Halifax | 4,800 |
| 10 | Severn Vale Homes | Gloucester | 3,900 |
| 11 | Riverview Estates | Leicester | 11,200 |
| 12 | Ashworth Housing Trust | Bolton | 5,500 |
| 13 | Meridian Housing Group | Southampton | 14,700 |
| 14 | Crestwood Homes | Wolverhampton | 6,300 |
| 15 | Thornbury Housing Association | Bradford | 8,900 |
| 16 | Lakeside Living | Liverpool | 10,100 |
| 17 | Harrowfield Homes | Derby | 4,400 |
| 18 | Summit Housing Partnership | Huddersfield | 7,200 |
| 19 | Wychwood Housing Association | Oxford | 3,200 |
| 20 | Dales & Moorland Housing | Skipton | 2,800 |
| 21 | Ironbridge Homes | Telford | 5,100 |
| 22 | Thameside Housing Trust | London (SE) | 18,500 |
| 23 | Avondale Community Homes | Bath | 3,600 |
| 24 | Stonebridge Housing Group | Stoke-on-Trent | 9,800 |
| 25 | Foxley Homes | Swindon | 4,500 |
| 26 | Eastgate Housing Association | Ipswich | 3,400 |
| 27 | Chiltern Edge Housing | High Wycombe | 6,700 |
| 28 | Maplewood Living | Cambridge | 5,900 |

### Contact Name Bank (use consistently)

| Organisation | Primary Contact | Role |
|---|---|---|
| Greendale Homes | Mark Jennings | Head of Sustainability |
| Northfield HA | Fiona Clarke | Director of Assets |
| Beacon Dwellings | David Osei | Sustainability Manager |
| Broadland Housing | Lena Morris | Director of Assets |
| Westmoor Housing Group | Karen Blackwell | Head of Sustainability |
| Riverside Comm Housing | James Thornton | Head of Property Services |
| Oaktree Living | Priya Sharma | Sustainability Lead |
| Millbrook Homes | Tom Henderson | Director of Development |
| Pennine Valleys Housing | Claire Whitfield | Asset Strategy Manager |
| Severn Vale Homes | Andrew Marsh | Operations Director |
| Riverview Estates | Helen Foster | Head of Housing |
| Ashworth Housing Trust | Nadeem Hussain | Property Director |
| Meridian Housing Group | Sarah Linehan | Chief Operating Officer |
| Crestwood Homes | Paul Gorman | Head of Asset Management |
| Thornbury HA | Rachel Iqbal | Sustainability Director |
| Lakeside Living | Chris Doyle | Head of Repairs & Maintenance |
| Harrowfield Homes | Joanna Briggs | Asset Manager |
| Summit Housing Partnership | Darren Walsh | Director of Property |
| Wychwood HA | Mei-Lin Chen | Sustainability Lead |
| Dales & Moorland Housing | Ian Calvert | Chief Executive |
| Ironbridge Homes | Samira Begum | Head of Compliance |
| Thameside Housing Trust | Oliver Grant | Group Director of Assets |
| Avondale Community Homes | Emma Stubbs | Operations Manager |
| Stonebridge Housing Group | Wayne Kirkpatrick | Director of Neighbourhoods |
| Foxley Homes | Danielle Webb | Head of Sustainability |
| Eastgate HA | Martin Osborne | Asset Management Lead |
| Chiltern Edge Housing | Abigail Reeves | Property Services Director |
| Maplewood Living | George Kaplan | Head of Development |

### Vantage Staff

| Name | Role |
|---|---|
| Rob Bryan | Managing Director |
| Steph Hosny | Club Lead (Carbon Club, Exec Club) |
| Jane | Club Lead (Finance, Performance) |
| John | Club Lead (Operations) |
| Sophie | Club Lead (Customer Experience) |
