# Vantage Demo Scenarios

These are concrete demo prompts to give the assistant, with exact expected outputs. Each scenario describes what to ask, which data sources the assistant should access, and the specific values it should return.

The assistant has access to:
- **Outlook** (Steph's inbox — ~72 emails with member organisations and internal Vantage staff)
- **OneDrive** (Steph's drive — Excel benchmarking files, workshop transcripts, case studies, membership data in `Vantage/` folder)
- **OneNote** (Steph's "Member Notes" notebook — 20 renewal call pages)
- **Mock Portal** (Zoho Connect mock — 10 discussion threads, login: steph / demo123)

---

## Scenario 1 — "Who's doing the most solar PV?"

**What it demonstrates:** Cross-source knowledge retrieval — the assistant searches across emails, OneDrive files, OneNote, and the portal to compile a comprehensive answer.

**Prompt:**

> Which of our Carbon Club members have done the most solar PV installations on their stock? Can you give me a quick summary I can share with Broadland Housing who are asking about it?

**Where the data lives:**

| Source | Location | What it contains |
|--------|----------|-----------------|
| 1:1 call notes | OneDrive: `Vantage/Carbon Club/1-to-1 Notes/carbon_club_1to1_notes.xlsx` | PV install counts in Greendale, Northfield, Beacon sheets |
| Workshop transcript | OneDrive: `Vantage/Carbon Club/Workshop Transcripts/carbon_club_workshop_14feb2025.txt` | Presentations from Greendale (1,200), Northfield (850), Beacon (340) |
| Portal thread | Mock portal: Thread 1 "Solar PV costs and procurement" | David (Beacon: 340→420), Mark (Greendale: 1,200), Fiona (Northfield: 850) reply |
| Case studies | OneDrive: `Vantage/Knowledge Hub/greendale_pv_case_study.txt`, `northfield_pv_battery_case_study.txt` | Detailed metrics for Greendale and Northfield programmes |
| Renewal notes | OneNote: Member Notes > Renewal Calls | Greendale, Northfield, Beacon renewal entries mention PV |
| Emails | Outlook: search "solar PV" | Steph sent PV case studies to Lena (Broadland) |

**Expected Answer:**

1. **Greendale Homes — 1,200 PV installations**
   - Cost: £4,200/unit via Solarfix Ltd framework
   - Tenant satisfaction: 89%
2. **Northfield Housing Association — 850 PV installations (with battery storage)**
   - Cost: £6,200/unit including £1,800 battery
   - Tenant satisfaction: 92%
3. **Beacon Dwellings — 340 PV installations**
   - Cost: £4,400/unit via Fusion21 framework
   - Part of wider EPC upgrade programme

Note: Broadland Housing has zero PV installations but is actively interested in starting a programme.

---

## Scenario 2 — "Compare Broadland against the club average"

**What it demonstrates:** Data triangulation — the assistant reads three separate Excel datasets and produces a side-by-side comparison.

**Prompt:**

> Compare Broadland Housing against the Carbon Club average across financials, tenant satisfaction, and operational KPIs. Show me a table.

**Where the data lives:**

| Source | Location |
|--------|----------|
| Financial data | OneDrive: `Vantage/Benchmarking Data/vfm_financial_data.xlsx` |
| TSM satisfaction | OneDrive: `Vantage/Benchmarking Data/tsm_satisfaction_data.xlsx` |
| Operational KPIs | OneDrive: `Vantage/Benchmarking Data/operational_kpi_data.xlsx` |

**Expected Answer:**

| Metric | Broadland Housing | Club Average |
|--------|-------------------|--------------|
| Cost Per Unit (£) | £5,800 | £4,729 |
| Reinvestment % | 9.2% | 7.9% |
| Operating Margin (SH) % | 24.2% | 24.5% |
| Overall Satisfaction % | 72.0% | 72.7% |
| Complaint Handling % | 31.0% | 36.8% |
| Void Turnaround (days) | 24.0 | 20.4 |
| EPC C+ Stock % | 63.0% | 74.5% |
| PV Installations | 0 | 358 |
| Retrofit Measures | 280 | 572 |

---

## Scenario 3 — "Prepare a renewal brief for Broadland Housing"

**What it demonstrates:** The assistant pulls data from multiple systems to compile a structured brief — membership records, attendance history, email interactions, and OneNote renewal notes.

**Prompt:**

> Broadland Housing's renewal is coming up on 1 April. Can you prepare a brief for my call with Lena Morris?

**Where the data lives:**

| Source | Location | What it contains |
|--------|----------|-----------------|
| Membership details | OneDrive: `Vantage/Membership/membership_master.xlsx` | Club memberships, fees, renewal dates |
| Attendance history | OneDrive: `Vantage/Carbon Club/Attendance/attendance_log.xlsx` | Session-by-session attendance |
| Email interactions | Outlook: search "Broadland" | 4 emails (PV case studies, void benchmarking, SHDF guidance, Lena's reply) |
| Renewal notes | OneNote: Member Notes > "Broadland Housing — 5 September 2024" | Renewed Carbon+Exec £8,500/yr, interested in PV |
| Benchmarking data | OneDrive: Benchmarking Data folder | Complaint handling 31%, EPC C+ 63% |

**Expected Answer — Key Facts:**

- **Organisation:** Broadland Housing, 4,200 homes, Norwich
- **Member Since:** 2021
- **Account Manager:** Steph Hosny
- **Primary Contact:** Lena Morris (Director of Assets)

**Memberships:**
- Carbon Club: £4,200/yr, renewal 01/04/2025
- Executive Club: £4,300/yr, renewal 01/04/2025
- **Total: £8,500/yr**

**Attendance:** 11/12 Carbon Club sessions (missed January 2025)

**Email Interactions (4):**
- 15/10/2024: Sent PV case studies (Greendale + Northfield)
- 16/10/2024: Lena replied requesting peer visit + SHDF guidance
- 22/11/2024: Sent void benchmarking data pack
- 08/01/2025: Sent SHDF Wave 2 funding guidance

**Coaching Points:**
- Complaint handling satisfaction 31% (below club avg ~37%)
- EPC C+ stock 63% (below club avg ~75%) — main driver for PV interest
- Broadland's interest in PV is a strong retention hook

---

## Scenario 4 — "Process today's Carbon Club session"

**What it demonstrates:** The assistant processes a meeting transcript and participant list, extracts attendance, topics, action items, and engagement alerts.

**Prompt:**

> Here is the transcript and attendee list from this morning's Carbon Club session. Can you update the attendance log, give me a summary of what was discussed, and flag anyone I should be worried about?

**Where the data lives:**

| Source | Location |
|--------|----------|
| Meeting transcript | OneDrive: `Vantage/Carbon Club/Workshop Transcripts/carbon_club_workshop_5mar2025.txt` |
| Participant list | OneDrive: `Vantage/Carbon Club/Teams Participant Lists/teams_participants_5mar2025.txt` |
| Attendance log | OneDrive: `Vantage/Carbon Club/Attendance/attendance_log.xlsx` |
| Membership master | OneDrive: `Vantage/Membership/membership_master.xlsx` |

**Expected — Attendance:**

18 organisations attended, 10 absent. Key absences: Riverview Estates (4th consecutive miss), Thornbury HA, Crestwood Homes.

**Expected — Topics Discussed:**
1. Greendale Homes — Solar PV Programme Update (1,200 installs, £4,200/unit)
2. Northfield HA — PV with Battery Storage (850 installs, £6,200/unit)
3. SHDF Wave 2 Funding Update (application window 1 Apr – 30 Jun 2025)
4. EPC Data Quality Challenges (15-25% inaccuracy reported by multiple members)
5. Tenant Communications for Retrofit Works (template letters, tenant liaison officers)

**Expected — Action Items:**

| Action | Owner | Deadline |
|--------|-------|----------|
| Circulate Greendale's PV procurement framework details | Steph Hosny | 12 March 2025 |
| Share Northfield's battery storage cost-benefit analysis | Fiona Clarke | 19 March 2025 |
| Introduce Lena (Broadland) to Mark (Greendale) for PV peer visit | Steph Hosny | 12 March 2025 |
| Submit EPC data quality survey responses | All members | 31 March 2025 |

**Expected — Engagement Alerts:**
- **Riverview Estates** has now missed 4 consecutive sessions (Dec 2024, Jan 2025, Feb 2025, Mar 2025). Steph noted in the transcript: "I notice Riverview Estates aren't with us again today — that's the fourth session in a row." Primary contact: Helen Foster. Recommend immediate outreach.
- **Tom Henderson (Millbrook Homes)** attended Carbon Club for the first time. He said: "this is my first Carbon Club session — I'm usually in the Exec Club." He is in the membership database under Executive Club only. Consider adding Carbon Club to his profile.

---

## Scenario 5 — "Summarise the void standards thread from the portal"

**What it demonstrates:** The assistant navigates the mock Zoho portal, reads an unstructured discussion thread, extracts quantitative data from natural prose, and produces a structured comparison.

**Prompt:**

> Can you go to the portal and pull together the responses from the void standards thread? Karen at Westmoor has been asking for a summary.

**Where the data lives:**

| Source | Location |
|--------|----------|
| Portal thread | Mock portal: Thread 7 "Void standards and turnaround thresholds" |

Note: The portal also has an exported text version on OneDrive at `Vantage/Portal Exports/void_standards_portal_thread.txt`. The assistant could use either source.

**Expected Answer:**

| Organisation | Target (days) | Actual (days) | Lettable Standard | Notable Practice |
|---|---|---|---|---|
| Northfield Housing Association | 18 | 16 | Enhanced | Pre-void inspections 4 weeks before tenancy end |
| Riverside Community Housing | 16 | 22 | Basic | Piloting 'void packs' with new tenant welcome kits |
| Greendale Homes | 20 | 19 | Standard | Dedicated void contractor team on 5-day SLA |
| Beacon Dwellings | 15 | 21 | Enhanced | Using Plentific platform for void works scheduling |
| Oaktree Living | 20 | 25 | Basic | Struggling with contractor availability — exploring in-house team |
| Millbrook Homes | 18 | 17 | Standard | Integrated void and lettings process — single team handles end-to-end |
| Pennine Valleys Housing | 22 | 24 | Basic | Recently tightened from 28-day target after board scrutiny |
| Severn Vale Homes | 16 | 15 | Enhanced | Top performer — in-house DLO |

**Common Themes:**
- In-house teams outperform external contractors
- Enhanced lettable standards add 3-5 days but improve tenant satisfaction
- Pre-void inspections are a high-impact lever

---

## Scenario 6 — "Flag any members I should be worried about"

**What it demonstrates:** The assistant cross-references attendance data, email history, and OneNote notes to identify disengaging members.

**Prompt:**

> Looking at Carbon Club, which members should I be worried about? Are there any engagement red flags?

**Where the data lives:**

| Source | Location |
|--------|----------|
| Attendance log | OneDrive: `Vantage/Carbon Club/Attendance/attendance_log.xlsx` |
| Emails | Outlook: Steph's inbox (Riverview correspondence) |
| Renewal notes | OneNote: "Riverview Estates — 14 November 2024" |

**Expected Answer:**

1. **Riverview Estates (RED)** — Has not attended the last 4 Carbon Club sessions (Dec 2024, Jan 2025, Feb 2025, Mar 2025). Helen Foster (Head of Housing) cited damp & mould caseload (42 per 1,000 homes) as the reason. Email from Helen (12 Feb 2025) confirms team is stretched. OneNote renewal notes (14 Nov 2024) flag risk of non-renewal — renewal due April 2025. Recommend immediate outreach.

2. **Thornbury Housing Association (RED)** — Attendance has dropped significantly in recent months. Low engagement across the board.

3. **Broadland Housing (AMBER)** — Good engagement overall (11/12 sessions) but missed the January session. Lena Morris is actively engaged via email. Renewal due 1 April 2025 — should be straightforward given strong engagement and active PV interest.
