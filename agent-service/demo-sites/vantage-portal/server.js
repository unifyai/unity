#!/usr/bin/env node
/**
 * Mock Zoho Connect Portal for Vantage Carbon Club.
 *
 * Standalone Node.js HTTP server (zero npm dependencies) that simulates the
 * Zoho Connect community portal with 6 discussion threads.
 *
 * Usage:
 *   node server.js 4002
 *
 * Login credentials:
 *   Username: steph    Password: demo123
 *   Username: olivia   Password: demo123
 */

const http = require("http");
const { URL } = require("url");
const querystring = require("querystring");

const PORT = parseInt(process.argv[2] || "4002", 10);

// ── Credentials ──────────────────────────────────────────────────────────────

const VALID_USERS = { steph: "demo123", olivia: "demo123" };

// ── Palette ──────────────────────────────────────────────────────────────────

const AVATAR_COLORS = [
  "#e91e63","#9c27b0","#673ab7","#3f51b5","#2196f3","#009688",
  "#4caf50","#ff9800","#f44336","#795548","#607d8b","#00bcd4",
  "#8bc34a","#ffc107","#ff5722",
];
const CAT_COLORS = {
  "Decarbonisation": "#4caf50",
  "Data & Reporting": "#2196f3",
  "Funding": "#ff9800",
  "Tenant Engagement": "#9c27b0",
  "Compliance": "#f44336",
};
const BADGE_CLASS = {
  "Decarbonisation": "badge-green",
  "Data & Reporting": "badge-blue",
  "Funding": "badge-orange",
  "Tenant Engagement": "badge-purple",
  "Compliance": "badge-red",
};

function initials(name) {
  const p = name.split(" ");
  return p.length > 1 ? (p[0][0] + p[p.length - 1][0]).toUpperCase() : name[0].toUpperCase();
}
function color(name) {
  let h = 0;
  for (const ch of name) h = ((h << 5) - h + ch.charCodeAt(0)) | 0;
  return AVATAR_COLORS[Math.abs(h) % AVATAR_COLORS.length];
}

// ── Thread data ──────────────────────────────────────────────────────────────

const THREADS = [
  {
    id: 1, title: "Solar PV costs and procurement", date: "8 October 2024",
    category: "Decarbonisation",
    op_name: "Lena Morris", op_org: "Broadland Housing", op_role: "Director of Assets",
    op_text: "Hi all,\n\nWe\u2019re looking to start a solar PV programme next financial year and would appreciate any insights on procurement routes and typical costs per unit. We\u2019re looking at about 500 properties initially.\n\nHas anyone used a framework agreement? What kind of costs per property are you seeing for a standard 4kW system?\n\nAny advice would be really appreciated. Thanks!",
    replies: [
      { name: "David Osei", org: "Beacon Dwellings", date: "9 Oct 2024", text: "Hi Lena, we\u2019ve been using a mix of direct procurement and the Fusion21 framework. Costs have been around \u00a34,400 per property for a 4kW system. We\u2019ve completed 340 installs so far as part of our wider EPC upgrade programme. The Fusion21 framework was straightforward to access and gave us good competitive tension on pricing." },
      { name: "Mark Jennings", org: "Greendale Homes", date: "10 Oct 2024", text: "We went through the Solarfix Ltd framework \u2014 very smooth procurement process. Our costs have been about \u00a34,200 per property including scaffolding. We\u2019ve now done 1,200 installs. Happy to share more details if useful, Lena." },
      { name: "Fiona Clarke", org: "Northfield HA", date: "11 Oct 2024", text: "Just to add a different perspective \u2014 we went PV plus battery storage at \u00a36,200 per unit. More expensive but tenants save significantly more. Worth considering depending on your tenant demographic and usage patterns." },
      { name: "Karen Blackwell", org: "Westmoor Housing Group", date: "12 Oct 2024", text: "Lena, if you haven\u2019t already, check whether your DNO has capacity constraints in your area. We found that grid connection was a bottleneck and we had to submit applications well in advance. Worth investigating early." },
    ],
  },
  {
    id: 2, title: "Heat pump experiences", date: "15 September 2024",
    category: "Decarbonisation",
    op_name: "Andrew Marsh", op_org: "Severn Vale Homes", op_role: "Operations Director",
    op_text: "We\u2019ve just completed a pilot of 50 air source heat pumps on off-gas properties. Results are mixed \u2014 great energy savings but some tenant complaints about noise and running costs in very cold weather.\n\nWould be keen to hear others\u2019 experiences.",
    replies: [
      { name: "Fiona Clarke", org: "Northfield HA", date: "16 Sep 2024", text: "We\u2019ve had similar findings with our 50-property pilot. The key issue for us was installation costs coming in at \u00a39,500 vs our \u00a38,000 estimate. Noise complaints from about 15% of tenants, mainly in semi-detached properties where the unit is near a neighbour\u2019s bedroom window." },
      { name: "Priya Sharma", org: "Oaktree Living", date: "17 Sep 2024", text: "We haven\u2019t started heat pumps yet but we\u2019re watching closely. Our concern is the running costs for tenants on prepayment meters \u2014 the electricity tariff is higher than gas. Has anyone looked at whether tenants are actually better off financially?" },
      { name: "Oliver Grant", org: "Thameside Housing Trust", date: "18 Sep 2024", text: "We\u2019ve done 120 heat pump installs across two estates. Overall satisfaction is 78% which is lower than we\u2019d like. The main complaints are about the need for larger radiators and the fact that heat pumps don\u2019t provide instant heat like gas boilers. Education and expectation setting is crucial." },
      { name: "Claire Whitfield", org: "Pennine Valleys Housing", date: "19 Sep 2024", text: "Just flagging that the government\u2019s Boiler Upgrade Scheme provides \u00a37,500 towards ASHPs. We\u2019ve been using this alongside SHDF funding to bring our net costs down significantly." },
    ],
  },
  {
    id: 3, title: "EPC data quality challenges", date: "5 November 2024",
    category: "Data & Reporting",
    op_name: "James Thornton", op_org: "Riverside Community Housing", op_role: "Head of Property Services",
    op_text: "We\u2019ve been cross-referencing our EPC data with recent stock condition surveys and finding significant discrepancies \u2014 about 20% of our EPCs don\u2019t match what\u2019s actually in the property.\n\nIs anyone else experiencing this? How are you managing it?",
    replies: [
      { name: "Mark Jennings", org: "Greendale Homes", date: "6 Nov 2024", text: "Same problem here. We\u2019re working with Elmhurst Energy to resurvey about 2,000 properties. Initial estimate is 15% have inaccurate EPCs. The issue is mainly with pre-1940 solid wall stock where internal insulation has been done but the EPC wasn\u2019t updated." },
      { name: "Sarah Linehan", org: "Meridian Housing Group", date: "6 Nov 2024", text: "We\u2019ve taken a different approach \u2014 building our own asset database that tracks installed measures against each property and calculates what the EPC should be. Not official but gives us a much more accurate picture for investment planning." },
      { name: "David Osei", org: "Beacon Dwellings", date: "7 Nov 2024", text: "We found about 25% inaccuracy in our solid wall stock EPCs. Real problem when you\u2019re trying to target retrofit programmes at properties below EPC C. We\u2019ve budgeted for a full resurvey in 2025/26." },
      { name: "Samira Begum", org: "Ironbridge Homes", date: "8 Nov 2024", text: "The regulatory implications worry me. If our reported EPC data is significantly wrong, that could be a compliance issue when the regulator comes asking. Are there any consequences for inaccurate data at the moment?" },
      { name: "Oliver Grant", org: "Thameside Housing Trust", date: "9 Nov 2024", text: "No direct penalty mechanism currently but the expectation is that data should be accurate. With the new consumer standards, if you\u2019re claiming 75% at EPC C but it\u2019s actually 65%, that\u2019s a governance concern." },
    ],
  },
  {
    id: 4, title: "SHDF Wave 2 application tips", date: "2 December 2024",
    category: "Funding",
    op_name: "Lena Morris", op_org: "Broadland Housing", op_role: "Director of Assets",
    op_text: "We\u2019re planning to apply for SHDF Wave 2.2 when the window opens in April. This will be our first application.\n\nAny tips from those who\u2019ve been through the process?",
    replies: [
      { name: "Claire Whitfield", org: "Pennine Valleys Housing", date: "3 Dec 2024", text: "Start early! The application is very detailed \u2014 you need property-level data, EPC certificates, contractor quotes, tenant engagement plans. It took us about six weeks. We got \u00a31.2m for 450 measures across 280 properties." },
      { name: "David Osei", org: "Beacon Dwellings", date: "4 Dec 2024", text: "Agree with Claire \u2014 the data requirements are significant. Make sure your EPC data is accurate before you apply. We had to redo some of our submission because our EPC data didn\u2019t match the properties we were targeting." },
      { name: "Mark Jennings", org: "Greendale Homes", date: "5 Dec 2024", text: "Our tip would be to have your procurement route sorted before you apply. Being able to show you have a framework agreement in place strengthens the bid significantly. We referenced our Solarfix framework and I think that helped." },
      { name: "Nadeem Hussain", org: "Ashworth Housing Trust", date: "6 Dec 2024", text: "Does anyone know whether the 33% match funding can come from the Affordable Homes Programme, or does it have to be the organisation\u2019s own capital? We\u2019re trying to work out our budget." },
    ],
  },
  {
    id: 5, title: "Tenant communications for retrofit works", date: "15 January 2025",
    category: "Tenant Engagement",
    op_name: "Karen Blackwell", op_org: "Westmoor Housing Group", op_role: "Head of Sustainability",
    op_text: "We\u2019re about to start a major insulation programme and I\u2019m looking for examples of good tenant communication.\n\nHow are others managing expectations and getting buy-in?",
    replies: [
      { name: "James Thornton", org: "Riverside Community Housing", date: "16 Jan 2025", text: "We\u2019ve developed template letters for different retrofit types \u2014 PV, insulation, windows, heat pumps. The key is being specific about savings: \u2018you could save \u00a3150\u2013200/year\u2019 rather than just \u2018you\u2019ll save money\u2019. Happy to share our templates." },
      { name: "Darren Walsh", org: "Summit Housing Partnership", date: "17 Jan 2025", text: "We recruited a dedicated tenant liaison officer for our retrofit programme. Single point of contact throughout the process. Made a massive difference \u2014 tenants actually thank us now." },
      { name: "Karen Blackwell", org: "Westmoor Housing Group", date: "18 Jan 2025", text: "Thanks both. We\u2019ve found that face-to-face visits work much better than letters for our older tenants. Our refusal rate dropped from 15% to under 5% once we started doing home visits. More resource-intensive but worth it." },
    ],
  },
  {
    id: 6, title: "Damp & mould management approaches", date: "3 February 2025",
    category: "Compliance",
    op_name: "Helen Foster", op_org: "Riverview Estates", op_role: "Head of Housing",
    op_text: "Following the coroner\u2019s recommendations in the Awaab Ishak case, we\u2019ve been reviewing our damp and mould procedures.\n\nWhat response times are others targeting and how are you prioritising cases?",
    replies: [
      { name: "Fiona Clarke", org: "Northfield HA", date: "4 Feb 2025", text: "We implemented a 48-hour inspection target for all damp and mould reports. Triaged into three categories: emergency (structural/health risk \u2014 same day), urgent (visible mould \u2014 48 hours), routine (condensation advice \u2014 5 days). Reduced open cases from 180 to 45 in six months." },
      { name: "Chris Doyle", org: "Lakeside Living", date: "5 Feb 2025", text: "We\u2019ve invested in environmental monitoring sensors in our highest-risk properties \u2014 about 500 sensors across our stock. They alert us to high humidity before mould develops. Proactive rather than reactive. Cost about \u00a380 per sensor including installation." },
      { name: "Joanna Briggs", org: "Harrowfield Homes", date: "5 Feb 2025", text: "We\u2019re training all housing officers to do basic damp and mould assessments during routine visits. The idea is early identification before tenants even report it. Also providing dehumidifiers and extractor fans to tenants in high-risk properties." },
      { name: "Lena Morris", org: "Broadland Housing", date: "6 Feb 2025", text: "We\u2019ve been struggling with this \u2014 currently at 35 cases per 1,000 homes which is above the sector average. We\u2019ve just brought in a specialist surveyor to review our worst-affected properties and develop a remediation programme. Concerned about compliance with the new Awaab\u2019s Law requirements." },
    ],
  },
];

const CATEGORIES = [...new Set(THREADS.map(t => t.category))];

// ── HTML helpers ─────────────────────────────────────────────────────────────

function esc(s) { return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }
function nl2br(s) { return esc(s).replace(/\n/g, "<br>"); }

function baseCSS() {
  return `* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }
.sidebar { position: fixed; left: 0; top: 0; bottom: 0; width: 60px; background: #3b2d5f; display: flex; flex-direction: column; align-items: center; padding-top: 12px; z-index: 100; }
.sidebar-logo { width: 36px; height: 36px; background: #7c5cbf; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 14px; margin-bottom: 20px; }
.sidebar-item { width: 44px; height: 44px; display: flex; flex-direction: column; align-items: center; justify-content: center; margin-bottom: 4px; border-radius: 8px; cursor: pointer; text-decoration: none; color: #b0a0cc; font-size: 9px; transition: background 0.15s; }
.sidebar-item:hover, .sidebar-item.active { background: #4d3d6f; color: white; }
.sidebar-icon { font-size: 18px; margin-bottom: 2px; }
.left-panel { position: fixed; left: 60px; top: 50px; bottom: 0; width: 220px; background: #fff; border-right: 1px solid #e0e0e0; padding: 16px 0; overflow-y: auto; }
.left-panel h3 { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; padding: 8px 16px 4px; margin-top: 12px; }
.left-panel h3:first-child { margin-top: 0; }
.left-panel a { display: flex; align-items: center; padding: 6px 16px; text-decoration: none; color: #333; font-size: 13px; transition: background 0.1s; }
.left-panel a:hover { background: #f0f0f0; }
.left-panel a.active { background: #ede7f6; color: #5c3d99; font-weight: 600; }
.group-dot { width: 8px; height: 8px; border-radius: 50%; margin-right: 10px; flex-shrink: 0; display: inline-block; }
.main { margin-left: 280px; padding: 24px 32px; max-width: 900px; }
.topbar { margin-left: 60px; position: fixed; top: 0; right: 0; left: 60px; height: 50px; background: #fff; border-bottom: 1px solid #e0e0e0; display: flex; align-items: center; padding: 0 24px; z-index: 50; }
.topbar-search { margin-left: 230px; flex: 1; max-width: 400px; }
.topbar-search input { width: 100%; padding: 6px 12px; border: 1px solid #ddd; border-radius: 4px; font-size: 13px; background: #f5f5f5; }
.topbar-user { margin-left: auto; display: flex; align-items: center; gap: 12px; }
.topbar-avatar { width: 32px; height: 32px; border-radius: 50%; background: #7c5cbf; color: white; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 600; }
.content-area { margin-top: 50px; }
.card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 16px; border: 1px solid #e8e8e8; }
.thread-item { display: flex; align-items: flex-start; padding: 14px 0; border-bottom: 1px solid #f0f0f0; text-decoration: none; color: inherit; }
.thread-item:last-child { border-bottom: none; }
.thread-item:hover { background: #fafafa; }
.thread-avatar { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 14px; flex-shrink: 0; margin-right: 14px; }
.thread-content { flex: 1; }
.thread-title { font-size: 15px; font-weight: 600; color: #333; margin-bottom: 4px; }
.thread-meta { font-size: 12px; color: #888; }
.thread-replies { font-size: 12px; color: #888; display: flex; align-items: center; gap: 4px; margin-left: auto; flex-shrink: 0; padding-left: 16px; }
.post { padding: 16px 0; border-bottom: 1px solid #f0f0f0; }
.post:last-child { border-bottom: none; }
.post-header { display: flex; align-items: center; margin-bottom: 10px; }
.post-avatar { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 13px; flex-shrink: 0; margin-right: 12px; }
.post-author { font-weight: 600; font-size: 14px; color: #333; }
.post-org { font-size: 12px; color: #7c5cbf; }
.post-date { font-size: 11px; color: #aaa; margin-left: auto; }
.post-body { font-size: 14px; line-height: 1.6; color: #444; padding-left: 48px; }
.reply { margin-left: 24px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; }
.badge-green { background: #e8f5e9; color: #2e7d32; }
.badge-blue { background: #e3f2fd; color: #1565c0; }
.badge-orange { background: #fff3e0; color: #e65100; }
.badge-purple { background: #ede7f6; color: #5c3d99; }
.badge-red { background: #fce4ec; color: #c62828; }
.back-link { font-size: 13px; color: #7c5cbf; text-decoration: none; margin-bottom: 16px; display: inline-block; }
.back-link:hover { text-decoration: underline; }
h2 { font-size: 20px; color: #333; margin-bottom: 16px; }
.login-container { display: flex; align-items: center; justify-content: center; min-height: 100vh; background: #3b2d5f; }
.login-box { background: white; border-radius: 12px; padding: 40px; width: 380px; box-shadow: 0 4px 24px rgba(0,0,0,0.2); }
.login-box h1 { color: #3b2d5f; font-size: 24px; margin-bottom: 8px; text-align: center; }
.login-box .subtitle { color: #888; font-size: 13px; text-align: center; margin-bottom: 24px; }
.login-box label { font-size: 13px; color: #555; display: block; margin-bottom: 4px; }
.login-box input[type="text"], .login-box input[type="password"] { width: 100%; padding: 10px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; margin-bottom: 16px; }
.login-box button { width: 100%; padding: 10px; background: #7c5cbf; color: white; border: none; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; transition: background 0.15s; }
.login-box button:hover { background: #6a4daa; }
.login-error { color: #d32f2f; font-size: 13px; text-align: center; margin-bottom: 12px; }`;
}

function shell(title, bodyHTML) {
  return `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>${esc(title)}</title><style>${baseCSS()}</style></head><body>${bodyHTML}</body></html>`;
}

function sidebarHTML() {
  return `<div class="sidebar"><div class="sidebar-logo">V</div>
<a href="/forums" class="sidebar-item" title="Home"><span class="sidebar-icon">&#8962;</span>Home</a>
<a href="/forums" class="sidebar-item" title="Feeds"><span class="sidebar-icon">&#9881;</span>Feeds</a>
<a href="/forums" class="sidebar-item active" title="Forums"><span class="sidebar-icon">&#9993;</span>Forums</a>
<a href="#" class="sidebar-item" title="Groups"><span class="sidebar-icon">&#9734;</span>Groups</a>
<a href="#" class="sidebar-item" title="Files"><span class="sidebar-icon">&#128196;</span>Files</a></div>`;
}

function topbarHTML() {
  return `<div class="topbar"><div class="topbar-search"><input type="text" placeholder="Search in all apps"></div><div class="topbar-user"><div class="topbar-avatar">SH</div></div></div>`;
}

function leftPanelHTML(activeCategory) {
  let h = `<div class="left-panel"><h3>Forums</h3><a href="/forums"${!activeCategory ? ' class="active"' : ''}><span class="group-dot" style="background:#7c5cbf"></span>All Discussions</a><h3>Categories</h3>`;
  for (const cat of CATEGORIES) {
    const active = activeCategory === cat ? ' class="active"' : '';
    h += `<a href="/forums?category=${encodeURIComponent(cat)}"${active}><span class="group-dot" style="background:${CAT_COLORS[cat] || '#888'}"></span>${esc(cat)}</a>`;
  }
  h += `<h3>My Groups</h3>
<a href="#"><span class="group-dot" style="background:#4caf50"></span>Carbon Club</a>
<a href="#"><span class="group-dot" style="background:#2196f3"></span>Executive Club</a>
<a href="#"><span class="group-dot" style="background:#ff9800"></span>Operations Club</a></div>`;
  return h;
}

// ── Page renderers ───────────────────────────────────────────────────────────

function loginPage(error) {
  const errHTML = error ? `<p class="login-error">${esc(error)}</p>` : "";
  return shell("Sign In \u2014 Vantage Connect",
    `<div class="login-container"><div class="login-box">
<h1>Vantage Connect</h1><p class="subtitle">Carbon Club Community Portal</p>${errHTML}
<form method="post" action="/login">
<label for="username">Email</label><input type="text" id="username" name="username" placeholder="steph@yourvantage.co.uk" required>
<label for="password">Password</label><input type="password" id="password" name="password" placeholder="Enter your password" required>
<button type="submit">Sign In</button></form></div></div>`);
}

function forumsPage(categoryFilter) {
  const threads = categoryFilter ? THREADS.filter(t => t.category === categoryFilter) : THREADS;
  let items = "";
  for (const t of threads) {
    items += `<a href="/forums/${t.id}" class="thread-item" style="display:flex">
<div class="thread-avatar" style="background:${color(t.op_name)}">${initials(t.op_name)}</div>
<div class="thread-content"><div class="thread-title">${esc(t.title)}</div>
<div class="thread-meta">${esc(t.op_name)} &middot; ${esc(t.op_org)} &middot; ${esc(t.date)}
<span class="badge ${BADGE_CLASS[t.category] || 'badge-blue'}" style="margin-left:8px">${esc(t.category)}</span></div></div>
<div class="thread-replies">&#128172; ${t.replies.length} replies</div></a>`;
  }
  return shell("Forums \u2014 Vantage Connect",
    sidebarHTML() + topbarHTML() + leftPanelHTML(categoryFilter) +
    `<div class="main content-area"><h2>Forum Discussions</h2><div class="card">${items}</div></div>`);
}

function threadPage(thread) {
  let repliesHTML = "";
  for (const r of thread.replies) {
    repliesHTML += `<div class="post reply"><div class="post-header">
<div class="post-avatar" style="background:${color(r.name)}">${initials(r.name)}</div>
<div><div class="post-author">${esc(r.name)}</div><div class="post-org">${esc(r.org)}</div></div>
<div class="post-date">${esc(r.date)}</div></div>
<div class="post-body">${nl2br(r.text)}</div></div>`;
  }
  return shell(`${thread.title} \u2014 Vantage Connect`,
    sidebarHTML() + topbarHTML() + leftPanelHTML(null) +
    `<div class="main content-area"><a href="/forums" class="back-link">&larr; Back to Forum Discussions</a>
<div class="card"><h2>${esc(thread.title)}</h2>
<p style="font-size:12px;color:#888;margin-bottom:20px"><span class="badge ${BADGE_CLASS[thread.category] || 'badge-blue'}">${esc(thread.category)}</span> &middot; Posted ${esc(thread.date)} &middot; ${thread.replies.length} replies</p>
<div class="post"><div class="post-header">
<div class="post-avatar" style="background:${color(thread.op_name)}">${initials(thread.op_name)}</div>
<div><div class="post-author">${esc(thread.op_name)}</div><div class="post-org">${esc(thread.op_org)} &middot; ${esc(thread.op_role)}</div></div>
<div class="post-date">${esc(thread.date)}</div></div>
<div class="post-body">${nl2br(thread.op_text)}</div></div>${repliesHTML}</div></div>`);
}

// ── HTTP server ──────────────────────────────────────────────────────────────

function parseBody(req) {
  return new Promise((resolve) => {
    let data = "";
    req.on("data", c => { data += c; });
    req.on("end", () => resolve(querystring.parse(data)));
  });
}

function send(res, status, html) {
  res.writeHead(status, { "Content-Type": "text/html; charset=utf-8" });
  res.end(html);
}

function redirect(res, location) {
  res.writeHead(302, { Location: location });
  res.end();
}

const server = http.createServer(async (req, res) => {
  const parsed = new URL(req.url, `http://localhost:${PORT}`);
  const path = parsed.pathname;
  const method = req.method;

  if (path === "/" || path === "/login") {
    if (method === "POST") {
      const body = await parseBody(req);
      const user = (body.username || "").trim().toLowerCase().split("@")[0];
      if (VALID_USERS[user] && VALID_USERS[user] === body.password) {
        return redirect(res, "/forums");
      }
      return send(res, 200, loginPage("Invalid username or password"));
    }
    return send(res, 200, loginPage(null));
  }

  if (path === "/logout") {
    return redirect(res, "/login");
  }

  if (path === "/forums") {
    const cat = parsed.searchParams.get("category") || null;
    return send(res, 200, forumsPage(cat));
  }

  const threadMatch = path.match(/^\/forums\/(\d+)$/);
  if (threadMatch) {
    const thread = THREADS.find(t => t.id === parseInt(threadMatch[1], 10));
    if (thread) return send(res, 200, threadPage(thread));
    return redirect(res, "/forums");
  }

  send(res, 404, shell("Not Found", '<div style="text-align:center;padding:60px"><h2>Page not found</h2></div>'));
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Vantage portal demo site listening on http://0.0.0.0:${PORT}`);
});
