"""
Seed Steph's Microsoft 365 account with Vantage demo data.

Seeds Outlook (52 emails) and OneNote (15 renewal call pages).
Optionally wipes existing data before seeding with --wipe flag.

Usage:
    uv run seed.py                        # Seed only (additive)
    uv run seed.py --wipe                 # Wipe everything, then seed
    uv run seed.py --token-file path.txt  # Read token from specific file
    MS365_ACCESS_TOKEN=xxx uv run seed.py # Token from env var
"""

import argparse
import os
import sys
import requests
import time
from pathlib import Path

BASE = "https://graph.microsoft.com/v1.0"
STEPH = "steph@unifyailtd123.onmicrosoft.com"
ROB = "yasser@unifyailtd123.onmicrosoft.com"

TOKEN = None
H_JSON = {}
H_HTML = {}
H_AUTH = {}


def _init_headers(token):
    global TOKEN, H_JSON, H_HTML, H_AUTH
    TOKEN = token
    H_JSON = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    H_HTML = {"Authorization": f"Bearer {token}", "Content-Type": "text/html"}
    H_AUTH = {"Authorization": f"Bearer {token}"}


# =========================================================================
# WIPE
# =========================================================================


def wipe_emails():
    print("Wiping emails...")
    for folder in ["Inbox", "SentItems"]:
        while True:
            r = requests.get(
                f"{BASE}/me/mailFolders/{folder}/messages",
                headers=H_AUTH,
                params={"$top": 50, "$select": "id"},
            )
            msgs = r.json().get("value", [])
            if not msgs:
                break
            for m in msgs:
                requests.delete(f"{BASE}/me/messages/{m['id']}", headers=H_AUTH)
            print(f"  Deleted {len(msgs)} from {folder}")
            time.sleep(0.5)
    print("  Done.")


def wipe_onenote():
    print("Wiping OneNote pages from 'Member Notes'...")
    r = requests.get(f"{BASE}/me/onenote/notebooks", headers=H_AUTH)
    for nb in r.json().get("value", []):
        if nb["displayName"] == "Member Notes":
            r2 = requests.get(
                f"{BASE}/me/onenote/notebooks/{nb['id']}/sections",
                headers=H_AUTH,
            )
            for sec in r2.json().get("value", []):
                r3 = requests.get(
                    f"{BASE}/me/onenote/sections/{sec['id']}/pages",
                    headers=H_AUTH,
                    params={"$top": 100},
                )
                for page in r3.json().get("value", []):
                    requests.delete(
                        f"{BASE}/me/onenote/pages/{page['id']}",
                        headers=H_AUTH,
                    )
                    time.sleep(0.3)
                print(f"  Deleted pages from section: {sec['displayName']}")
    print("  Done.")


# =========================================================================
# EMAIL DATA — expanded to ~45 emails
# =========================================================================


def _e(subject, date, frm, to, body, direction):
    return {
        "subject": subject,
        "date": date,
        "from": {"name": frm[0], "email": frm[1]},
        "to": {"name": to[0], "email": to[1]},
        "body": body,
        "direction": direction,
    }


S = ("Steph Hosny", STEPH)
R = ("Rob Bryan", ROB)
LM = ("Lena Morris", "l.morris@broadlandhousing.org.uk")
MJ = ("Mark Jennings", "m.jennings@greendalehousing.org.uk")
FC = ("Fiona Clarke", "f.clarke@northfieldhousing.org.uk")
KB = ("Karen Blackwell", "k.blackwell@westmoorhousing.org.uk")
DO = ("David Osei", "d.osei@beaconhousing.org.uk")
JT = ("James Thornton", "j.thornton@riversidehousing.org.uk")
PS = ("Priya Sharma", "p.sharma@oaktreehousing.org.uk")
TH = ("Tom Henderson", "t.henderson@millbrookhousing.org.uk")
AM = ("Andrew Marsh", "a.marsh@severnvalehousing.org.uk")
HF = ("Helen Foster", "h.foster@riverviewhousing.org.uk")
CW = ("Claire Whitfield", "c.whitfield@penninehousing.org.uk")
OG = ("Oliver Grant", "o.grant@thamesidehousing.org.uk")
CD = ("Chris Doyle", "c.doyle@lakesidehousing.org.uk")
SL = ("Sarah Linehan", "s.linehan@meridianhousing.org.uk")
NH = ("Nadeem Hussain", "n.hussain@ashworthhousing.org.uk")
DW = ("Darren Walsh", "d.walsh@summithousing.org.uk")
SB = ("Samira Begum", "s.begum@ironbridgehousing.org.uk")
JB = ("Joanna Briggs", "j.briggs@harrowfieldhousing.org.uk")
DnW = ("Danielle Webb", "d.webb@foxleyhousing.org.uk")

EMAILS = [
    # === BROADLAND HOUSING — full thread ===
    _e(
        "Solar PV case studies - Greendale & Northfield",
        "2024-10-15T14:30:00Z",
        S,
        LM,
        "Hi Lena,\n\nFollowing our one-to-one last month, please find attached the case studies from Greendale Homes and Northfield HA on their solar PV programmes.\n\nKey highlights:\n- Greendale: 1,200 installations at \u00a34,200/unit via Solarfix framework, 89% tenant satisfaction\n- Northfield: 850 installations with battery storage at \u00a36,200/unit, tenants saving additional \u00a3150/yr vs PV-only\n\nBoth are happy to host peer visits if you'd like to see the programmes in action. Let me know and I can make introductions.\n\nBest,\nSteph",
        "sent",
    ),
    _e(
        "RE: Solar PV case studies - Greendale & Northfield",
        "2024-10-16T09:15:00Z",
        LM,
        S,
        "Hi Steph,\n\nThese are really helpful, thank you. The Greendale numbers in particular are very encouraging \u2014 their costs are lower than I expected.\n\nI'd definitely be interested in a peer visit to Greendale. Could you set that up for me? We're looking at a programme of about 500 properties initially and their experience with the Solarfix framework would be really valuable.\n\nI've shared the case studies with our Director of Finance and he's keen to understand the SHDF funding angle better. Could you send over the latest SHDF guidance when you get a chance?\n\nThanks,\nLena",
        "inbox",
    ),
    _e(
        "Void benchmarking data pack",
        "2024-11-22T11:00:00Z",
        S,
        LM,
        "Hi Lena,\n\nAs promised, please find attached the void turnaround benchmarking data compiled from Carbon Club members.\n\nThe pack includes:\n- Target vs actual turnaround times for 15 organisations\n- Breakdown by lettable standard (Basic, Standard, Enhanced)\n- Rent loss calculations per void\n- Best practice examples from top performers\n\nSevern Vale Homes are the standout at 15 days average \u2014 they attribute this to their in-house DLO. Might be worth a conversation with Andrew Marsh there if you're looking to improve your turnaround times.\n\nBest,\nSteph",
        "sent",
    ),
    _e(
        "SHDF Wave 2 funding guidance document",
        "2025-01-08T10:20:00Z",
        S,
        LM,
        "Hi Lena,\n\nAs requested, here is the DESNZ guidance on the SHDF Wave 2 application process and eligibility criteria.\n\nKey points for Broadland:\n- Application window for Wave 2.2 opens 1 April 2025, closes 30 June 2025\n- PV on its own IS eligible as long as properties are below EPC C\n- Minimum 33% match funding from the housing association\n- You'll need property-level EPC data and contractor quotes in the application\n\nClaire Whitfield at Pennine Valleys went through the Wave 2.1 process and said the application took about 6 weeks \u2014 I'd recommend starting to pull your data together now.\n\nBest,\nSteph",
        "sent",
    ),
    _e(
        "RE: SHDF Wave 2 funding guidance document",
        "2025-01-09T11:45:00Z",
        LM,
        S,
        "Thanks Steph, this is really useful. I've shared it with our finance team and we're starting to pull the property-level data together.\n\nOne question \u2014 do we need actual contractor quotes at application stage, or just indicative pricing? We haven't gone to market yet.\n\nAlso, could you set up that peer visit to Greendale? I'd like to go before the application window opens if possible.\n\nLena",
        "inbox",
    ),
    _e(
        "RE: SHDF Wave 2 funding guidance document",
        "2025-01-10T09:00:00Z",
        S,
        LM,
        "Hi Lena,\n\nIndicative pricing should be fine at application stage \u2014 you just need to demonstrate you've got a realistic cost estimate. Claire at Pennine Valleys can confirm but I believe they used framework prices rather than actual quotes.\n\nI'll set up the Greendale visit \u2014 will email Mark today.\n\nSteph",
        "sent",
    ),
    # === GREENDALE HOMES ===
    _e(
        "Carbon Club spotlight - February session",
        "2025-01-20T16:45:00Z",
        MJ,
        S,
        "Hi Steph,\n\nJust confirming I'm happy to do the PV spotlight at the February Carbon Club session. I'll prepare a short presentation covering:\n\n- Programme progress (1,200 installs completed)\n- Costs and procurement approach\n- Tenant satisfaction results\n- Plans for the remaining 600 properties\n\nShould be about 15 minutes. Let me know if you want me to cover anything else specifically.\n\nMark",
        "inbox",
    ),
    _e(
        "RE: Carbon Club spotlight - February session",
        "2025-01-21T09:30:00Z",
        S,
        MJ,
        "Perfect, thanks Mark. That all sounds great.\n\nOne thing that might be useful to mention is the grid connection process with Western Power Distribution \u2014 a few members have been asking about potential bottlenecks with the DNO. Your experience with batch applications would be really helpful for them to hear.\n\nAlso, Lena Morris from Broadland is keen to visit your programme. Would you be open to hosting a peer visit? She's looking at starting a PV programme of about 500 properties.\n\nSteph",
        "sent",
    ),
    _e(
        "RE: Carbon Club spotlight - February session",
        "2025-01-21T14:20:00Z",
        MJ,
        S,
        "Good shout on the DNO bit \u2014 I'll definitely include that. Happy to host Lena. Get her to drop me a line and we'll sort a date. February or early March would work well.\n\nMark",
        "inbox",
    ),
    _e(
        "Greendale PV procurement framework details",
        "2025-03-06T14:00:00Z",
        MJ,
        S,
        "Hi Steph,\n\nAs promised in yesterday's session, here are the details of the Solarfix procurement framework:\n\n- Framework operator: Solarfix Ltd (accessed via Fusion21)\n- Contract value: \u00a35.4m over 3 years\n- Procurement timeline: 12 weeks from going to market to appointment\n- Key benefits: pre-vetted contractors, competitive pricing, simplified process\n- Contact at Solarfix: James Wright, j.wright@solarfix.co.uk\n\nHappy for you to share this with anyone who's interested. And yes, Lena from Broadland is very welcome to visit \u2014 just get her to drop me a line and we'll sort a date.\n\nMark",
        "inbox",
    ),
    _e(
        "Greendale 12-month PV performance report",
        "2025-02-18T10:30:00Z",
        MJ,
        S,
        "Hi Steph,\n\nAs mentioned at the session, here's our 12-month PV performance report that went to the board in January. Key findings:\n\n- Average generation: 3,800 kWh per property per year (vs predicted 4,000)\n- Average tenant saving: \u00a3180/year on electricity\n- 8% tenant refusal rate (mostly elderly residents concerned about disruption)\n- 96% system reliability (only 48 faults across 1,200 installations)\n\nFeel free to share the format with other members \u2014 I think having a standardised performance report template could be useful for the group.\n\nMark",
        "inbox",
    ),
    # === NORTHFIELD HA ===
    _e(
        "Battery storage cost-benefit analysis",
        "2025-03-07T11:30:00Z",
        FC,
        S,
        "Hi Steph,\n\nAs discussed at Wednesday's session, here's our battery storage cost-benefit analysis.\n\nSummary of key findings:\n- Additional cost per property for battery: \u00a31,800\n- Additional tenant saving per year: \u00a3150 vs PV-only\n- Battery payback period: 10-12 years\n- System used: GivEnergy 5.2kWh, 12-year warranty\n- Failure rate: <1% (6 out of 850)\n\nThe full spreadsheet model is attached. Feel free to share with the group.\n\nFiona",
        "inbox",
    ),
    _e(
        "Heat pump pilot update",
        "2024-11-28T15:00:00Z",
        FC,
        S,
        "Hi Steph,\n\nQuick update on our heat pump pilot \u2014 we've now completed all 50 installations.\n\nResults are mixed honestly:\n- Energy cost savings are good for most tenants\n- But 15% have complained about noise (mainly semi-detached where the unit is near a neighbour's window)\n- Installation costs came in at \u00a39,500 vs our \u00a38,000 estimate\n- Overall satisfaction at 76%\n\nWe're holding off on expanding until we've resolved the noise issue. Might need to look at different unit placements or acoustic enclosures.\n\nFiona",
        "inbox",
    ),
    # === WESTMOOR HOUSING GROUP ===
    _e(
        "RE: Void standards - urgent",
        "2024-09-15T08:45:00Z",
        KB,
        S,
        "Hi Steph,\n\nThanks for the call last week. The board meeting was tough \u2014 they're really pushing us on void turnaround times. We're at 28 days average which is well above what other HAs seem to be achieving.\n\nCould you share any benchmarking data you have on void standards? Specifically I'd like to know:\n- What targets other Carbon Club members are working to\n- What lettable standards they use (Basic/Standard/Enhanced)\n- Any practical tips that have made a real difference\n\nI also posted on the portal about this so hopefully we'll get some responses there too.\n\nKaren",
        "inbox",
    ),
    _e(
        "RE: Void standards - urgent",
        "2024-09-16T10:20:00Z",
        S,
        KB,
        "Hi Karen,\n\nSorry to hear the board meeting was difficult. I'll pull together a benchmarking data pack for you \u2014 should have it ready by end of next week.\n\nIn the meantime, a few quick pointers from what I'm hearing across the club:\n- Severn Vale Homes are the best performer at 15 days avg \u2014 they run an in-house DLO which makes a big difference\n- Northfield are at 16 days with pre-void inspections 4 weeks before tenancy end \u2014 that seems to be a key lever\n- Millbrook integrated their void and lettings teams into one \u2014 eliminated a 3-4 day handover gap\n\nI'll include all this detail in the data pack.\n\nSteph",
        "sent",
    ),
    _e(
        "RE: Void standards - urgent",
        "2024-09-17T11:30:00Z",
        KB,
        S,
        "That's really helpful Steph, thank you. The Severn Vale example is interesting \u2014 we've been thinking about going in-house for a while but the setup costs have put us off. Would be good to understand Andrew's experience.\n\nLooking forward to the data pack.\n\nKaren",
        "inbox",
    ),
    _e(
        "Insulation programme tenant engagement plan",
        "2025-01-22T14:15:00Z",
        KB,
        S,
        "Hi Steph,\n\nWe're finalising our tenant engagement plan for the insulation programme starting in April. Karen mentioned at the last one-to-one that she'd value any examples from other members.\n\nWe're planning face-to-face visits for elderly tenants (which seems to be best practice from what we've heard) and letters for everyone else. James at Riverside offered to share his templates \u2014 has he sent those through yet?\n\nKaren",
        "inbox",
    ),
    _e(
        "RE: Insulation programme tenant engagement plan",
        "2025-01-23T09:00:00Z",
        S,
        KB,
        "Hi Karen,\n\nYes, James sent the templates through last week \u2014 I'll forward them to you now. They're really well structured, especially the part about being specific with savings figures.\n\nYour plan sounds solid. The face-to-face approach for elderly tenants is definitely the way to go \u2014 several members have seen refusal rates drop from 15% to under 5% with home visits.\n\nSteph",
        "sent",
    ),
    # === SEVERN VALE HOMES ===
    _e(
        "Whole-house retrofit spotlight offer",
        "2025-03-06T16:15:00Z",
        AM,
        S,
        "Hi Steph,\n\nFollowing yesterday's session, I'd be happy to do a spotlight on our whole-house retrofit programme at a future session. We've done 200 properties with insulation, windows, and heat pumps \u2014 EPC E to B in most cases.\n\nI could also cover the in-house DLO approach since a few people were asking about it, particularly Priya from Oaktree who's thinking about going in-house for their void works.\n\nLet me know what month works best.\n\nAndrew",
        "inbox",
    ),
    _e(
        "DLO setup costs breakdown",
        "2024-10-22T10:00:00Z",
        AM,
        S,
        "Hi Steph,\n\nFollowing the interest from a few members about our in-house DLO, I've put together a breakdown of the setup costs:\n\n- 8 operatives recruited (painters x3, plumber x1, electrician x1, general x2, supervisor x1)\n- Vehicles: \u00a3145k (8 vans, bought outright)\n- Tools and equipment: \u00a328k\n- Recruitment costs: \u00a312k\n- Total setup: approximately \u00a3185k\n\nWe broke even within 14 months through reduced contractor costs and dramatically lower rent loss. Our void turnaround went from 24 days to 15 days.\n\nHappy to share more detail with anyone who's interested.\n\nAndrew",
        "inbox",
    ),
    # === RIVERVIEW ESTATES ===
    _e(
        "Checking in - Carbon Club attendance",
        "2025-02-10T09:00:00Z",
        S,
        HF,
        "Hi Helen,\n\nI just wanted to check in \u2014 I noticed Riverview Estates hasn't attended the last couple of Carbon Club sessions. I hope everything is okay.\n\nWe've had some really useful discussions recently on solar PV programmes, EPC data quality, and SHDF funding \u2014 I can send you the notes if that would be helpful.\n\nYour renewal is coming up in April so it would be good to catch up before then. Are you free for a one-to-one in the next couple of weeks?\n\nBest,\nSteph",
        "sent",
    ),
    _e(
        "RE: Checking in - Carbon Club attendance",
        "2025-02-12T14:30:00Z",
        HF,
        S,
        "Hi Steph,\n\nThanks for reaching out. To be honest, things have been very stretched here. We've had a significant increase in damp and mould cases \u2014 we're at 42 per 1,000 homes now \u2014 and that's been taking up most of the team's time.\n\nI know we should be attending the sessions but it's been hard to prioritise with everything going on. The notes would be really helpful though, yes please.\n\nI can do a one-to-one on the 25th if that works?\n\nHelen",
        "inbox",
    ),
    _e(
        "RE: Checking in - Carbon Club attendance",
        "2025-02-13T08:30:00Z",
        S,
        HF,
        "Hi Helen,\n\nCompletely understand \u2014 damp and mould is a huge pressure for everyone right now. The 25th works for me, let's do 10am.\n\nIn the meantime, Chris Doyle at Lakeside Living has done some really innovative work with environmental monitoring sensors for damp prevention. Might be worth a conversation? I can connect you if useful.\n\nI'll send the session notes over today.\n\nSteph",
        "sent",
    ),
    # === OAKTREE LIVING ===
    _e(
        "In-house void team - business case help",
        "2025-02-20T11:00:00Z",
        PS,
        S,
        "Hi Steph,\n\nFollowing the discussion at the last Carbon Club about void turnaround times, I'm now seriously looking at setting up an in-house void team. Andrew at Severn Vale mentioned he broke even within 14 months.\n\nCould you connect me with Andrew? I'd like to understand the setup costs, team structure, and how they manage the recruitment. Our current contractor availability is terrible \u2014 25 days average turnaround and it's costing us over \u00a3400k a year in lost rent.\n\nPriya",
        "inbox",
    ),
    _e(
        "RE: In-house void team - business case help",
        "2025-02-21T09:15:00Z",
        S,
        PS,
        "Hi Priya,\n\nAbsolutely \u2014 I've CC'd Andrew Marsh on this email. Andrew, would you mind sharing the DLO setup costs breakdown you sent me last month? I think it would be really helpful for Priya.\n\nPriya, Andrew's team went from 24 days average to 15 days after setting up the DLO. The setup cost was about \u00a3185k but they broke even within 14 months through reduced contractor costs and lower rent loss.\n\nSteph",
        "sent",
    ),
    # === BEACON DWELLINGS ===
    _e(
        "EPC data quality - resurvey programme",
        "2024-12-05T15:30:00Z",
        DO,
        S,
        "Hi Steph,\n\nWe've been digging into our EPC data following the discussions at Carbon Club and it's worse than I thought \u2014 about 25% of our solid wall stock has inaccurate EPCs.\n\nMark at Greendale mentioned they're working with Elmhurst Energy on a resurvey. Could you connect us? We're budgeting for a full resurvey in 2025/26 and it would be good to compare approaches.\n\nAlso, our PV programme is at 340 installs now. Hoping to get to 500 by end of the financial year.\n\nDavid",
        "inbox",
    ),
    _e(
        "RE: EPC data quality - resurvey programme",
        "2024-12-06T09:45:00Z",
        S,
        DO,
        "Hi David,\n\nThanks for the update. 25% inaccuracy is significant \u2014 you're not alone though, we're hearing similar numbers from several members.\n\nI've copied Mark Jennings (Greendale) on this email \u2014 Mark, would you mind sharing your Elmhurst Energy contact with David?\n\nGood progress on the PV programme. 500 by year end sounds achievable.\n\nSteph",
        "sent",
    ),
    _e(
        "Beacon PV programme Q3 update",
        "2025-02-05T16:00:00Z",
        DO,
        S,
        "Hi Steph,\n\nQuick update \u2014 we hit 420 PV installs by end of January. On track for 500 by end of March. Costs holding steady at around \u00a34,400/unit through the Fusion21 framework.\n\nWe've also started the EPC resurvey with a firm called EPC Solutions (not Elmhurst in the end \u2014 they were cheaper). First batch of 300 properties done, confirming about 22% inaccuracy rate.\n\nDavid",
        "inbox",
    ),
    # === PENNINE VALLEYS HOUSING ===
    _e(
        "SHDF Wave 2.1 success - thank you",
        "2024-11-05T10:00:00Z",
        CW,
        S,
        "Hi Steph,\n\nJust wanted to let you know we were successful with our SHDF Wave 2.1 bid \u2014 \u00a31.2 million for 450 measures across 280 properties!\n\nThe guidance notes you shared and the examples from other Carbon Club members were really helpful in putting the application together. Took about 6 weeks but definitely worth the effort.\n\nHappy to share our experience with anyone else in the club who's thinking of applying for Wave 2.2.\n\nClaire",
        "inbox",
    ),
    _e(
        "SHDF Wave 2.1 - delivery progress",
        "2025-02-15T11:00:00Z",
        CW,
        S,
        "Hi Steph,\n\nThought you'd like an update on our SHDF programme. We've completed 180 of the 450 planned measures so far \u2014 mainly loft insulation and cavity wall. The remaining 270 are scheduled for Q1-Q2 2025/26.\n\nThe trickiest part has been the tenant engagement for external wall insulation \u2014 scaffolding access is always a challenge. We're using face-to-face visits which has helped but it's time-consuming.\n\nClaire",
        "inbox",
    ),
    # === THAMESIDE HOUSING TRUST ===
    _e(
        "EPC regulatory reporting concerns",
        "2024-11-18T13:00:00Z",
        OG,
        S,
        "Hi Steph,\n\nI've been thinking more about the EPC data quality issue we discussed at the last session. I'm worried about the regulatory implications.\n\nIf we're reporting 75% of stock at EPC C or above but the reality is 65%, that's a governance issue. The consumer standards require accurate data and I don't think we can defend inaccurate EPC reporting.\n\nAre any other members looking at this from a compliance angle? Would be useful to know how others are approaching the risk.\n\nOliver",
        "inbox",
    ),
    _e(
        "Heat pump programme - board presentation",
        "2025-01-28T09:30:00Z",
        OG,
        S,
        "Hi Steph,\n\nWe've got a board presentation next month on our heat pump programme. Our satisfaction is still at 78% which the board aren't happy about.\n\nDo you have any examples of how other HAs have improved their heat pump satisfaction scores? I know the sector is struggling with this generally but any evidence of what works would be helpful.\n\nOliver",
        "inbox",
    ),
    _e(
        "RE: Heat pump programme - board presentation",
        "2025-01-29T10:00:00Z",
        S,
        OG,
        "Hi Oliver,\n\nThe 78% is actually in line with what we're seeing across the club \u2014 heat pumps are a harder sell than PV because of the behavioural change required.\n\nKey things that seem to help:\n- Education sessions before installation (Severn Vale do a 1-hour home visit)\n- Setting expectations about how heat pumps work differently to gas boilers\n- Follow-up visits 2 weeks post-install to address any issues early\n\nI'll pull together a summary of heat pump satisfaction data across all members \u2014 it might help to show the board that 78% is actually mid-table, not bottom.\n\nSteph",
        "sent",
    ),
    # === LAKESIDE LIVING ===
    _e(
        "Environmental monitoring sensors - results",
        "2025-01-15T14:30:00Z",
        CD,
        S,
        "Hi Steph,\n\nYou asked me to share an update on our environmental monitoring sensors programme. We've now got 500 sensors deployed across our highest-risk properties.\n\nResults so far:\n- 23 properties flagged for high humidity before any mould developed\n- Cost: about \u00a380 per sensor including installation\n- Reduced reactive damp & mould callouts by 30% in monitored properties\n- Tenants like the proactive approach \u2014 satisfaction up 12 points\n\nHappy to present on this at a future session if useful.\n\nChris",
        "inbox",
    ),
    _e(
        "Sensor supplier details",
        "2025-02-25T10:00:00Z",
        CD,
        S,
        "Hi Steph,\n\nA few members asked about our sensor supplier after I mentioned it on the portal. We use Switchee \u2014 they offer a housing association package at \u00a380/unit installed with a 5-year data contract.\n\nContact: Sarah Mitchell, s.mitchell@switchee.co, 020 7946 0958.\n\nThey also do a free pilot of 50 units so you can test before committing to a larger rollout.\n\nChris",
        "inbox",
    ),
    # === MERIDIAN HOUSING GROUP ===
    _e(
        "Asset database approach to EPC tracking",
        "2024-11-20T10:15:00Z",
        SL,
        S,
        "Hi Steph,\n\nFollowing the EPC data quality discussion, a few members asked about our approach to tracking installed measures.\n\nWe've built a simple Excel-based asset database that tracks what measures have actually been installed in each property (wall insulation, loft insulation, glazing, heating system) and calculates what the EPC should be. It's not official but it gives us a much more accurate picture than relying on the EPC register.\n\nI've attached the template if anyone wants to adapt it. Happy to walk through it on a call.\n\nSarah",
        "inbox",
    ),
    # === RIVERSIDE COMMUNITY HOUSING ===
    _e(
        "Tenant comms templates for retrofit works",
        "2025-02-03T09:30:00Z",
        JT,
        S,
        "Hi Steph,\n\nAs discussed, I've attached our template tenant notification letters for different types of retrofit works \u2014 PV installations, insulation, window replacements, and heat pumps.\n\nThe key thing we learned is to be specific about savings \u2014 'you could save \u00a3150-200/year' rather than just 'you'll save money'. That made a big difference to our take-up rates.\n\nHappy for you to share with the group.\n\nJames",
        "inbox",
    ),
    _e(
        "Stock condition survey results",
        "2025-01-08T15:45:00Z",
        JT,
        S,
        "Hi Steph,\n\nOur stock condition survey is now complete. Key findings:\n- 20% discrepancy between EPC records and actual condition\n- 340 properties need urgent window replacement\n- 85 properties identified as potential damp & mould risks\n- Average SAP rating across stock: 62 (target: 69)\n\nWe're using this to prioritise our retrofit investment plan for 2025-2028. Happy to share the methodology if other members are planning similar surveys.\n\nJames",
        "inbox",
    ),
    # === ASHWORTH HOUSING TRUST ===
    _e(
        "SHDF match funding question",
        "2024-12-08T14:00:00Z",
        NH,
        S,
        "Hi Steph,\n\nFollowing the SHDF discussion on the portal, I'm still trying to clarify the match funding rules. Does the 33% match have to come from our own reserves, or can we use Affordable Homes Programme funding?\n\nOur finance director is asking and I can't find a clear answer in the guidance.\n\nNadeem",
        "inbox",
    ),
    _e(
        "RE: SHDF match funding question",
        "2024-12-09T09:30:00Z",
        S,
        NH,
        "Hi Nadeem,\n\nI checked with DESNZ and the answer is: the match funding must come from the housing association's own resources or private borrowing. You cannot use other government grant funding (including AHP) as match.\n\nHowever, you CAN use the Boiler Upgrade Scheme alongside SHDF for heat pump installations specifically \u2014 they count as different pots.\n\nSteph",
        "sent",
    ),
    # === SUMMIT HOUSING PARTNERSHIP ===
    _e(
        "Tenant liaison officer - job description",
        "2025-02-18T11:30:00Z",
        DW,
        S,
        "Hi Steph,\n\nFollowing the tenant comms discussion, we've been looking at hiring a dedicated tenant liaison officer for our retrofit programme. It's made such a difference for Riverside and Lakeside.\n\nWould you be able to share any example job descriptions from other members who've recruited for this role?\n\nDarren",
        "inbox",
    ),
    _e(
        "RE: Tenant liaison officer - job description",
        "2025-02-19T09:00:00Z",
        S,
        DW,
        "Hi Darren,\n\nGood move! I'll ask James at Riverside and Chris at Lakeside if they're happy to share their JDs. Both have had great success with dedicated liaison officers.\n\nKey things to include based on what I've heard:\n- Single point of contact for tenants throughout the process\n- Both pre-visit engagement and post-install follow-up\n- Ability to handle complaints and escalate where needed\n- Some technical knowledge (doesn't need to be an expert but should understand the basics of the measures being installed)\n\nSteph",
        "sent",
    ),
    # === IRONBRIDGE HOMES ===
    _e(
        "Regulatory compliance query - Awaab's Law",
        "2025-01-20T16:00:00Z",
        SB,
        S,
        "Hi Steph,\n\nWith Awaab's Law now in force, we're reviewing our damp and mould response times. We currently target 5 working days for initial inspection but I'm not sure that's fast enough.\n\nWhat are other members targeting? I saw on the portal that Northfield are doing 48 hours which seems ambitious but they've had great results.\n\nSamira",
        "inbox",
    ),
    _e(
        "RE: Regulatory compliance query - Awaab's Law",
        "2025-01-21T09:45:00Z",
        S,
        SB,
        "Hi Samira,\n\nGood question and very timely. The range across Carbon Club members is:\n- Northfield: 48 hours (best in class)\n- Lakeside: 72 hours (supported by sensor monitoring)\n- Most others: 3-5 working days\n\nI'd recommend tightening to at least 72 hours for initial inspection. The key is having a triage system \u2014 Northfield categorise into emergency (same day), urgent (48 hours), and routine (5 days).\n\nI'll share the full damp & mould portal discussion with you.\n\nSteph",
        "sent",
    ),
    # === HARROWFIELD HOMES ===
    _e(
        "Damp & mould training programme",
        "2025-02-08T13:00:00Z",
        JB,
        S,
        "Hi Steph,\n\nWe've just rolled out damp and mould assessment training for all our housing officers. The idea is early identification during routine visits before tenants even report it.\n\nIt's a half-day session covering:\n- Visual signs of damp vs condensation\n- Moisture meter use\n- When to escalate to a specialist surveyor\n- How to advise tenants on ventilation\n\nWould other members be interested in the training materials? We developed it with a surveying firm and they've said we can share it.\n\nJoanna",
        "inbox",
    ),
    # === FOXLEY HOMES ===
    _e(
        "SHDF guidance request",
        "2025-03-04T10:00:00Z",
        DnW,
        S,
        "Hi Steph,\n\nI know you shared the SHDF Wave 2 guidance with some members individually but I don't think everyone got it. Could you send it round to the whole group? We're starting to think about an application.\n\nAlso, are there any upcoming sessions focused specifically on SHDF bid preparation? I think there's quite a lot of interest.\n\nDanielle",
        "inbox",
    ),
    # === INTERNAL — STEPH <> ROB ===
    _e(
        "Carbon Club - March session agenda",
        "2025-02-28T16:00:00Z",
        S,
        R,
        "Hi Rob,\n\nDraft agenda for next week's Carbon Club session (5 March):\n\n1. Welcome and housekeeping\n2. Spotlight: Greendale PV update (Mark Jennings)\n3. Spotlight: Northfield battery storage (Fiona Clarke)\n4. SHDF Wave 2 funding update\n5. EPC data quality roundtable\n6. Tenant communications for retrofit\n7. Wrap-up and actions\n\nI want to flag that Riverview Estates haven't attended the last 3 sessions. Helen says they're stretched with damp & mould cases. Their renewal is in April \u2014 I'm worried we might lose them.\n\nSteph",
        "sent",
    ),
    _e(
        "RE: Carbon Club - March session agenda",
        "2025-03-01T08:45:00Z",
        R,
        S,
        "Agenda looks good Steph.\n\nRe Riverview \u2014 yes, let's discuss. Maybe we need to offer them some targeted support on the damp & mould side to demonstrate value before the renewal conversation. Chris at Lakeside has done great work with those sensors \u2014 could we connect them?\n\nRob",
        "inbox",
    ),
    _e(
        "Broadland Housing renewal - April 2025",
        "2025-03-03T11:00:00Z",
        S,
        R,
        "Hi Rob,\n\nHeads up that Broadland Housing's renewal is coming up on 1 April. They're in Carbon Club (\u00a34,200/yr) and Exec Club (\u00a34,300/yr) \u2014 total \u00a38,500.\n\nThey've been good members \u2014 attended 10 of 11 sessions this year. Lena Morris has been very engaged, particularly around starting a PV programme. I've sent her case studies, connected her with Mark at Greendale for a peer visit, and shared SHDF funding guidance.\n\nOne coaching point for the renewal call: their complaint handling satisfaction is at 31%, below the club average of 35%. And their EPC C+ stock is only 63% vs club average of 72% \u2014 which is actually a good argument for staying in the Carbon Club as they've got a lot to learn from peers.\n\nI'll prepare a full renewal brief before the call.\n\nSteph",
        "sent",
    ),
    _e(
        "Q3 club performance summary",
        "2025-01-12T10:00:00Z",
        S,
        R,
        "Hi Rob,\n\nHere's the Q3 performance summary across all clubs:\n\n- Carbon Club: 28 members, 85% average session attendance, 2 renewals at risk (Riverview, Thornbury)\n- Executive Club: 50 members, 78% attendance, no immediate concerns\n- Finance Directors: 42 members, 72% attendance, 1 renewal at risk\n- Operations: 45 members, 80% attendance, strong engagement\n- Customer Experience: 42 members, 75% attendance, growing well\n- Performance Improvement: 45 members, 82% attendance, flagship club\n\nTotal membership revenue on track at \u00a31.1m for the year. Main risk is Riverview \u2014 they're in Carbon Club only at \u00a34,800/yr but the disengagement pattern is concerning.\n\nSteph",
        "sent",
    ),
    _e(
        "New member enquiry - Pinnacle Living",
        "2025-02-24T14:00:00Z",
        R,
        S,
        "Steph,\n\nHad a call with Sarah Thompson at Pinnacle Living (Doncaster, ~6,800 homes). They're interested in joining the Carbon Club \u2014 they're just starting their decarbonisation strategy and want to learn from peers.\n\nCan you follow up and send them the membership info? I think they'd be a good fit.\n\nRob",
        "inbox",
    ),
    _e(
        "RE: New member enquiry - Pinnacle Living",
        "2025-02-25T09:00:00Z",
        S,
        R,
        "Will do \u2014 I'll reach out to Sarah today. Another Carbon Club member would be great, especially as they're early in their journey. Lots to learn from the established members.\n\nSteph",
        "sent",
    ),
]

# =========================================================================
# ONENOTE DATA — expanded to 15 renewal notes
# =========================================================================

RENEWAL_NOTES = [
    {
        "org": "Broadland Housing",
        "date": "5 September 2024",
        "attendees": "Rob Bryan, Lena Morris (Director of Assets, Broadland)",
        "bullets": [
            "Broadland very interested in starting a PV programme \u2014 asked for contacts at orgs who have done it at scale",
            "Renewed for Carbon Club + Exec Club (2-year deal, \u00a38,500/yr)",
            "Lena mentioned they're also looking at air source heat pumps for off-gas properties",
            "Currently 63% of stock at EPC C or above \u2014 wants to improve significantly",
            "Lena very engaged \u2014 attended 10 of 11 sessions this year",
        ],
        "action": "Steph to send examples of PV case studies from other members",
    },
    {
        "org": "Westmoor Housing Group",
        "date": "12 September 2024",
        "attendees": "Steph Hosny, Karen Blackwell (Head of Sustainability, Westmoor)",
        "bullets": [
            "Renewed Carbon Club only this year (\u00a34,200/yr)",
            "Karen flagged concerns about void standards \u2014 turnaround times increased to 28 days avg, board asking questions",
            "Interested in benchmarking void costs against peers",
            "Also asked about tenant engagement approaches for upcoming insulation programme",
            "Mentioned budget for face-to-face home visits as part of insulation rollout",
        ],
        "action": "Share void standards thread from Zoho portal",
    },
    {
        "org": "Greendale Homes",
        "date": "20 September 2024",
        "attendees": "Steph Hosny, Mark Jennings (Head of Sustainability, Greendale)",
        "bullets": [
            "PV programme going very well \u2014 on track for 1,200 installs by year end",
            "Mark happy to do a spotlight at Carbon Club and host peer visits",
            "Renewed Carbon Club + Performance Club (\u00a39,200/yr)",
            "Raised EPC data quality as emerging concern \u2014 working with Elmhurst Energy to resurvey 2,000 properties",
            "Shared 12-month PV performance report \u2014 3,800 kWh avg per property",
        ],
        "action": "Schedule Greendale PV spotlight for Feb 2025 session",
    },
    {
        "org": "Northfield Housing Association",
        "date": "3 October 2024",
        "attendees": "Steph Hosny, Fiona Clarke (Director of Assets, Northfield)",
        "bullets": [
            "PV+battery programme at 500 installs, targeting 850 by end of Q4",
            "Fiona keen to share the cost-benefit analysis showing battery value",
            "Also piloting 50 heat pumps \u2014 mixed results so far on tenant satisfaction (76%)",
            "Noise complaints from 15% of tenants with heat pumps",
            "Renewed Carbon Club + Operations Club (\u00a38,100/yr)",
        ],
        "action": "Schedule Northfield battery storage spotlight for Feb session",
    },
    {
        "org": "Thameside Housing Trust",
        "date": "15 October 2024",
        "attendees": "Rob Bryan, Oliver Grant (Group Director of Assets, Thameside)",
        "bullets": [
            "120 heat pump installs completed across two estates",
            "Satisfaction at 78% \u2014 lower than target. Noise and instant heat concerns",
            "Oliver worried about EPC data quality and regulatory reporting implications",
            "Renewed all five clubs (\u00a322,500/yr) \u2014 largest single client",
            "Board presentation on heat pump programme scheduled for February",
        ],
        "action": "Connect Oliver with Sarah (Meridian) re asset database approach to EPC tracking",
    },
    {
        "org": "Oaktree Living",
        "date": "28 October 2024",
        "attendees": "Jane, Priya Sharma (Sustainability Lead, Oaktree)",
        "bullets": [
            "No PV programme yet but watching Greendale and Northfield closely",
            "Main focus currently on cavity wall insulation \u2014 400 properties targeted",
            "Contractor availability a major challenge \u2014 impacting both voids and retrofit",
            "Seriously considering in-house void team \u2014 current 25 day avg costing \u00a3400k/yr in lost rent",
            "Renewed Carbon Club only (\u00a33,800/yr)",
        ],
        "action": "Share contractor framework information from other members; connect with Andrew Marsh (Severn Vale) re DLO setup",
    },
    {
        "org": "Severn Vale Homes",
        "date": "8 November 2024",
        "attendees": "Steph Hosny, Andrew Marsh (Operations Director, Severn Vale)",
        "bullets": [
            "Whole-house retrofit programme completed \u2014 200 properties E to B",
            "Andrew very positive about in-house DLO approach \u2014 wants to present to the group",
            "DLO setup cost was \u00a3185k, broke even in 14 months",
            "Looking at expanding programme to additional 300 properties in 2025/26",
            "Renewed Carbon Club + Customer Experience Club (\u00a37,600/yr)",
        ],
        "action": "Schedule Severn Vale retrofit spotlight for future session",
    },
    {
        "org": "Riverview Estates",
        "date": "14 November 2024",
        "attendees": "Steph Hosny, Helen Foster (Head of Housing, Riverview)",
        "bullets": [
            "Helen seemed disengaged \u2014 said team is stretched and struggling to attend sessions",
            "Riverview haven't attended last two Carbon Club sessions",
            "Main focus currently on damp and mould compliance \u2014 42 cases per 1,000 homes",
            "Renewal coming up April 2025 \u2014 risk of non-renewal if engagement doesn't improve",
            "Suggested connecting Helen with Chris Doyle (Lakeside) re sensor monitoring approach",
        ],
        "action": "Steph to follow up with Helen in January to discuss engagement",
    },
    {
        "org": "Beacon Dwellings",
        "date": "22 November 2024",
        "attendees": "Rob Bryan, David Osei (Sustainability Manager, Beacon)",
        "bullets": [
            "PV programme at 340 installs as part of wider EPC programme",
            "EPC data quality issues \u2014 25% inaccuracy for solid wall stock",
            "Planning full resurvey in 2025/26 to get accurate baseline",
            "Hoping to reach 500 PV installs by end of financial year",
            "Renewed Carbon Club + Finance Directors Club (\u00a37,800/yr)",
        ],
        "action": "Connect David with Mark (Greendale) re Elmhurst Energy EPC surveys",
    },
    {
        "org": "Lakeside Living",
        "date": "5 December 2024",
        "attendees": "John, Chris Doyle (Head of Repairs & Maintenance, Lakeside)",
        "bullets": [
            "Invested in 500 environmental monitoring sensors for damp and mould prevention",
            "Proactive approach \u2014 sensors alert team before mould develops",
            "Cost about \u00a380 per sensor including installation, supplier is Switchee",
            "Reduced reactive damp & mould callouts by 30% in monitored properties",
            "Renewed Carbon Club + Operations Club (\u00a37,400/yr)",
        ],
        "action": "Chris to present on sensor programme at future session",
    },
    {
        "org": "Riverside Community Housing",
        "date": "10 December 2024",
        "attendees": "Steph Hosny, James Thornton (Head of Property Services, Riverside)",
        "bullets": [
            "Stock condition survey now complete \u2014 20% discrepancy with EPC records",
            "340 properties need urgent window replacement",
            "85 properties identified as potential damp & mould risks",
            "Developed excellent tenant comms templates \u2014 willing to share with group",
            "Renewed all three clubs (\u00a312,200/yr)",
        ],
        "action": "James to send tenant comms templates to Steph for distribution",
    },
    {
        "org": "Meridian Housing Group",
        "date": "15 December 2024",
        "attendees": "Steph Hosny, Sarah Linehan (COO, Meridian)",
        "bullets": [
            "Built own asset database for tracking installed measures against properties",
            "Excel-based tool that calculates expected EPC from installed measures",
            "More accurate than official EPC register for investment planning",
            "Several members interested in the methodology",
            "Renewed Carbon Club + Executive Club (\u00a39,800/yr)",
        ],
        "action": "Sarah to share asset database template with interested members",
    },
    {
        "org": "Summit Housing Partnership",
        "date": "8 January 2025",
        "attendees": "John, Darren Walsh (Director of Property, Summit)",
        "bullets": [
            "Planning to recruit dedicated tenant liaison officer for retrofit programme",
            "Inspired by Riverside and Lakeside examples",
            "Looking for example job descriptions from other members",
            "Retrofit programme covering 600 properties over next 2 years",
            "Renewed Carbon Club + Operations Club (\u00a38,200/yr)",
        ],
        "action": "Share tenant liaison officer JD examples from Riverside and Lakeside",
    },
    {
        "org": "Ironbridge Homes",
        "date": "15 January 2025",
        "attendees": "Steph Hosny, Samira Begum (Head of Compliance, Ironbridge)",
        "bullets": [
            "Concerned about Awaab's Law compliance \u2014 current damp response time is 5 working days",
            "Needs to tighten to at least 72 hours for initial inspection",
            "Interested in Northfield's triage system (emergency/urgent/routine)",
            "Renewed Carbon Club only (\u00a34,100/yr)",
            "Samira quite new in role \u2014 finding Carbon Club very valuable for peer learning",
        ],
        "action": "Share damp & mould portal thread and Northfield's triage approach with Samira",
    },
    {
        "org": "Ashworth Housing Trust",
        "date": "22 January 2025",
        "attendees": "Rob Bryan, Nadeem Hussain (Property Director, Ashworth)",
        "bullets": [
            "Exploring SHDF Wave 2.2 application for insulation programme",
            "Clarified match funding must come from own reserves (not AHP)",
            "Boiler Upgrade Scheme can be used alongside SHDF for heat pumps",
            "Planning 200-property insulation programme if SHDF bid successful",
            "Renewed Carbon Club + Finance Directors Club (\u00a37,200/yr)",
        ],
        "action": "Share SHDF guidance document and connect with Claire Whitfield (Pennine Valleys) for application advice",
    },
]


# =========================================================================
# SEED FUNCTIONS
# =========================================================================


def seed_emails():
    print(f"\nSeeding {len(EMAILS)} emails...")
    success = 0
    for email in EMAILS:
        folder = "SentItems" if email["direction"] == "sent" else "Inbox"
        message = {
            "subject": email["subject"],
            "body": {"contentType": "Text", "content": email["body"]},
            "from": {
                "emailAddress": {
                    "name": email["from"]["name"],
                    "address": email["from"]["email"],
                },
            },
            "toRecipients": [
                {
                    "emailAddress": {
                        "name": email["to"]["name"],
                        "address": email["to"]["email"],
                    },
                },
            ],
            "receivedDateTime": email["date"],
            "sentDateTime": email["date"],
            "isRead": True,
        }
        r = requests.post(
            f"{BASE}/me/mailFolders/{folder}/messages",
            headers=H_JSON,
            json=message,
        )
        d = "SENT" if email["direction"] == "sent" else "RECV"
        other = (
            email["to"]["name"]
            if email["direction"] == "sent"
            else email["from"]["name"]
        )
        if r.status_code in (200, 201):
            success += 1
            print(
                f"  [OK] {d} | {email['date'][:10]} | {other} | {email['subject'][:60]}",
            )
        else:
            print(f"  [FAIL] {r.status_code} | {email['subject'][:60]}")
        time.sleep(0.3)
    print(f"  {success}/{len(EMAILS)} emails created.")


def seed_onenote():
    print(f"\nSeeding {len(RENEWAL_NOTES)} OneNote pages...")

    r = requests.get(f"{BASE}/me/onenote/notebooks", headers=H_AUTH)
    nb_id = None
    for nb in r.json().get("value", []):
        if nb["displayName"] == "Member Notes":
            nb_id = nb["id"]
    if not nb_id:
        r = requests.post(
            f"{BASE}/me/onenote/notebooks",
            headers=H_JSON,
            json={"displayName": "Member Notes"},
        )
        nb_id = r.json()["id"]
        print("  Created notebook: Member Notes")
    time.sleep(2)

    r = requests.get(f"{BASE}/me/onenote/notebooks/{nb_id}/sections", headers=H_AUTH)
    sec_id = None
    for sec in r.json().get("value", []):
        if sec["displayName"] == "Renewal Calls":
            sec_id = sec["id"]
    if not sec_id:
        r = requests.post(
            f"{BASE}/me/onenote/notebooks/{nb_id}/sections",
            headers=H_JSON,
            json={"displayName": "Renewal Calls"},
        )
        sec_id = r.json()["id"]
        print("  Created section: Renewal Calls")
    time.sleep(2)

    success = 0
    for note in RENEWAL_NOTES:
        bullets_html = "\n".join(f"<li>{b}</li>" for b in note["bullets"])
        html = f"""<!DOCTYPE html>
<html><head><title>{note['org']} \u2014 {note['date']}</title></head>
<body>
<h1>{note['org']}</h1>
<p><strong>Date:</strong> {note['date']}</p>
<p><strong>Attendees:</strong> {note['attendees']}</p>
<h2>Discussion Notes</h2>
<ul>{bullets_html}</ul>
<h2>Action Item</h2>
<p style="background-color:#FFF3CD;padding:8px;border-left:4px solid #FFC107;">
<strong>ACTION:</strong> {note['action']}</p>
</body></html>"""
        r = requests.post(
            f"{BASE}/me/onenote/sections/{sec_id}/pages",
            headers=H_HTML,
            data=html.encode("utf-8"),
        )
        if r.status_code in (200, 201):
            success += 1
            print(f"  [OK] {note['org']} \u2014 {note['date']}")
        else:
            print(f"  [FAIL] {note['org']}: {r.status_code}")
        time.sleep(1)
    print(f"  {success}/{len(RENEWAL_NOTES)} pages created.")


# =========================================================================
# MAIN
# =========================================================================


def _resolve_token(args):
    if args.token_file:
        p = Path(args.token_file)
        if not p.exists():
            print(f"[ERROR] Token file not found: {p}")
            sys.exit(1)
        return p.read_text().strip()

    env_token = os.environ.get("MS365_ACCESS_TOKEN", "")
    if env_token:
        return env_token

    default_paths = [
        Path(__file__).parent.parent.parent.parent.parent / "demos" / "token.txt",
        Path.home() / "token.txt",
    ]
    for p in default_paths:
        if p.exists():
            return p.read_text().strip()

    print(
        "[ERROR] No token found. Provide via --token-file, MS365_ACCESS_TOKEN env var, or demos/token.txt",
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Seed Vantage demo data into M365")
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Wipe all existing data before seeding",
    )
    parser.add_argument(
        "--token-file",
        type=str,
        help="Path to file containing the Graph API token",
    )
    args = parser.parse_args()

    token = _resolve_token(args)
    _init_headers(token)

    r = requests.get(f"{BASE}/me", headers=H_AUTH)
    if r.status_code != 200:
        print(f"[ERROR] Token validation failed ({r.status_code}). Get a fresh token.")
        sys.exit(1)
    print(
        f"Authenticated as: {r.json()['displayName']} ({r.json()['userPrincipalName']})",
    )

    print("=" * 60)
    print(f"VANTAGE DEMO — {'WIPE & ' if args.wipe else ''}SEED")
    print("=" * 60)

    if args.wipe:
        wipe_emails()
        wipe_onenote()

    seed_emails()
    seed_onenote()

    print("\n" + "=" * 60)
    print(f"DONE — {len(EMAILS)} emails, {len(RENEWAL_NOTES)} OneNote pages")
    print("=" * 60)


if __name__ == "__main__":
    main()
