#!/usr/bin/env python3
"""Simulate a CodeActActor.act(persist=True) session by writing events to local Orchestra.

Writes ManagerMethod and ToolLoop log entries directly to the local Orchestra
database via its REST API. The console's action pane loads these on page
refresh via its historical polling path.

The trajectory follows the real CodeActActor flow:
  1. Discovery phase (tool_choice=required): GuidanceManager_search +
     FunctionManager_search_functions are gated before any doing tools.
  2. Doing phase: execute_function / execute_code become available.
  3. Interjections, notifications, clarification requests, pause/resume/stop.

Prerequisites:
    1. Local Orchestra running (via console/scripts/local.sh)
    2. Console running (http://localhost:3333)

Usage:
    .venv/bin/python scripts/dev/simulate_action_stream.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import requests

BASE = "http://127.0.0.1:8000/v0"
KEY = "BRE8zK0jon7e7Ix6mtKYZkqO6kIMY0w1QgFFcm3zg8o="
AUTH = {"Authorization": f"Bearer {KEY}"}
PROJECT = "Assistants"
ASSISTANT_ID = "1"
USER_ID = "test-user-001"

MM_CTX = "All/Events/ManagerMethod"
TL_CTX = "All/Events/ToolLoop"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def api(method: str, path: str, **kwargs) -> requests.Response:
    r = getattr(requests, method)(f"{BASE}{path}", headers=AUTH, **kwargs)
    return r


def setup() -> None:
    projects = api("get", "/projects").json()
    if PROJECT not in projects:
        api("post", "/project", json={"name": PROJECT})
        print(f"  Created project '{PROJECT}'")

    for ctx in [MM_CTX, TL_CTX]:
        api("post", "/context", json={"project_name": PROJECT, "name": ctx})
    print("  Contexts ready")


def log_entry(context: str, entries: dict) -> None:
    entries["_user_id"] = USER_ID
    entries["_assistant_id"] = ASSISTANT_ID
    r = api(
        "post",
        "/logs",
        json={
            "project_name": PROJECT,
            "context": context,
            "entries": entries,
        },
    )
    if not r.ok:
        print(f"    LOG ERROR: {r.status_code} {r.text[:200]}")


def log_mm(
    calling_id,
    hierarchy,
    *,
    phase,
    manager="CodeActActor",
    method="act",
    display_label=None,
    request=None,
    answer=None,
    persist=None,
    status="ok",
):
    entries = {
        "calling_id": calling_id,
        "event_id": str(uuid4()),
        "event_timestamp": now_iso(),
        "manager": manager,
        "method": method,
        "phase": phase,
        "hierarchy": hierarchy,
        "hierarchy_label": "->".join(hierarchy),
        "status": status,
    }
    if persist is not None:
        entries["persist"] = persist
    if display_label:
        entries["display_label"] = display_label
    if request:
        entries["request"] = request
    if answer:
        entries["answer"] = answer
    log_entry(MM_CTX, entries)
    print(f"  ManagerMethod: {phase} [{manager}.{method}]")


def log_tl(hierarchy, message, *, tool_aliases=None):
    entries = {
        "event_id": str(uuid4()),
        "event_timestamp": now_iso(),
        "message": message,
        "method": "CodeActActor.act",
        "hierarchy": hierarchy,
        "hierarchy_label": "->".join(hierarchy),
    }
    if tool_aliases:
        entries["tool_aliases"] = tool_aliases
    log_entry(TL_CTX, entries)
    kind = message.get("role", "?")
    if message.get("_steering"):
        kind = f"steering:{message.get('_steering_action')}"
    print(f"  ToolLoop: {kind}")


def _tc(tc_id, name, arguments):
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _tool_result(tc_id, name, content):
    return {
        "role": "tool",
        "tool_call_id": tc_id,
        "name": name,
        "content": json.dumps(content) if isinstance(content, dict) else content,
    }


def _thinking(text, tool_calls=None):
    msg = {
        "role": "assistant",
        "content": None,
        "reasoning_content": text,
        "thinking_blocks": [{"type": "thinking", "thinking": text}],
    }
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def run() -> None:
    print("\n--- Setup ---")
    setup()

    cid = str(uuid4())
    h = [f"CodeActActor.act({cid[:4]})"]

    print(f"\n=== Simulating persist=True session ===")
    print(f"  calling_id: {cid}")
    print(f"  hierarchy:  {h[0]}\n")

    # ── 1. ManagerMethod incoming ──────────────────────────────────────────
    print("[1] ManagerMethod incoming")
    log_mm(
        cid,
        h,
        phase="incoming",
        display_label="Taking Action",
        request="Help me set up Google Drive credentials for the unify-production project.",
        persist=True,
    )

    # ── 2. User message ───────────────────────────────────────────────────
    print("[2] User message")
    log_tl(
        h,
        {
            "role": "user",
            "content": "Help me set up Google Drive credentials for the unify-production project.",
        },
    )

    # ── 3. Discovery phase: GuidanceManager_search + FunctionManager_search_functions ──
    print("[3] Discovery: GuidanceManager_search + FunctionManager_search_functions")
    log_tl(
        h,
        _thinking(
            "The user wants to set up Google Drive credentials. Before I act, I need "
            "to check if there's any saved guidance or existing functions for this. "
            "Let me search both the guidance store and function catalog.",
            tool_calls=[
                _tc(
                    "tc_gm",
                    "GuidanceManager_search",
                    {"query": "Google Drive API credentials service account setup"},
                ),
                _tc(
                    "tc_fm",
                    "FunctionManager_search_functions",
                    {"query": "Google Drive credentials setup"},
                ),
            ],
        ),
        tool_aliases={
            "GuidanceManager_search": "Searching for relevant guidance",
            "FunctionManager_search_functions": "Searching for relevant skills",
        },
    )

    # ── 4. GuidanceManager_search result ──────────────────────────────────
    print("[4] GuidanceManager_search result")
    log_tl(
        h,
        _tool_result(
            "tc_gm",
            "GuidanceManager_search",
            {
                "results": [
                    {
                        "id": 42,
                        "title": "Google Cloud Service Account Setup",
                        "summary": "Step-by-step procedure for creating GCP service accounts "
                        "with Drive API access. Includes OAuth consent screen configuration.",
                    },
                ],
            },
        ),
    )

    # ── 5. FunctionManager_search_functions result ────────────────────────
    print("[5] FunctionManager_search_functions result")
    log_tl(
        h,
        _tool_result(
            "tc_fm",
            "FunctionManager_search_functions",
            {"results": [], "message": "No matching functions found."},
        ),
    )

    # ── 6. Doing phase: web search for current instructions ───────────────
    print("[6] Doing: execute_function(primitives.web.ask) — web search")
    log_tl(
        h,
        _thinking(
            "Good — I found guidance on GCP service account setup, but no saved functions. "
            "The guidance mentions the process but cloud console UIs change frequently. "
            "Let me search the web for the most current 2026 instructions before proceeding.",
            tool_calls=[
                _tc(
                    "tc_web",
                    "execute_function",
                    {
                        "function_name": "primitives.web.ask",
                        "call_kwargs": {
                            "text": "Google Cloud service account setup with Drive API access 2026",
                        },
                    },
                ),
            ],
        ),
        tool_aliases={"execute_function": "primitives.web.ask"},
    )

    # ── 7. Child ManagerMethod for execute_function(primitives.web.ask) ───
    ef_suffix = "a1b2"
    ef_h = [*h, f"execute_function(primitives.web.ask)({ef_suffix})"]
    ef_cid = str(uuid4())
    log_mm(
        ef_cid,
        ef_h,
        phase="incoming",
        manager="CodeActActor",
        method="execute_function",
        display_label="Running: primitives.web.ask",
    )
    log_mm(
        ef_cid,
        ef_h,
        phase="outgoing",
        manager="CodeActActor",
        method="execute_function",
        display_label="Running: primitives.web.ask",
        answer="Found current GCP setup instructions.",
    )

    # ── 8. Web search tool result ─────────────────────────────────────────
    print("[7] Web search result")
    log_tl(
        h,
        _tool_result(
            "tc_web",
            "execute_function",
            {
                "answer": (
                    "As of March 2026, to create a GCP service account with Drive access:\n"
                    "1. Go to console.cloud.google.com\n"
                    "2. Select your project from the dropdown\n"
                    "3. Navigate to IAM & Admin → Service Accounts\n"
                    "4. Click 'Create Service Account'\n"
                    "5. Name it and grant 'Google Drive API' role\n"
                    "6. Create a JSON key under the Keys tab\n"
                    "7. Enable the Drive API under APIs & Services → Library"
                ),
            },
        ),
    )

    # ── 9. Clarification: request_clarification tool call ────────────────
    print("[8] Doing: request_clarification — which project ID?")
    log_tl(
        h,
        _thinking(
            "I found the current setup process, but the user said 'unify-production'. "
            "GCP project IDs are often different from display names. I need to confirm "
            "the exact project ID before proceeding.",
            tool_calls=[
                _tc(
                    "tc_clar",
                    "request_clarification",
                    {
                        "question": (
                            "You mentioned 'unify-production' — is that the exact Google Cloud "
                            "project ID, or is the project ID different from the display name? "
                            "You can check at console.cloud.google.com under the project selector."
                        ),
                    },
                ),
            ],
        ),
        tool_aliases={"request_clarification": "Requesting clarification"},
    )

    # ── 10. Clarification result (user answered via ConversationManager) ──
    print("[9] Clarification result: user confirms project ID")
    log_tl(
        h,
        _tool_result(
            "tc_clar",
            "request_clarification",
            "The project ID is 'unify-prod-2026'. The display name is unify-production.",
        ),
    )

    # ── 11. Execute code: notification + knowledge update ─────────────────
    print("[10] Doing: execute_code — notify + store credentials info")
    code_block = (
        'notify({"type": "progress", "message": "Starting service account setup for unify-prod-2026..."})\n'
        "\n"
        "result = await primitives.knowledge.update(\n"
        '    instructions="Store the following: GCP project unify-prod-2026 (display name: '
        "unify-production) is being configured with a Drive API service account. "
        'Setup initiated on 2026-03-09."\n'
        ")"
    )
    log_tl(
        h,
        _thinking(
            "The user confirmed the project ID is 'unify-prod-2026'. Let me send a progress "
            "notification and store this configuration detail in the knowledge base for future reference.",
            tool_calls=[
                _tc(
                    "tc_code",
                    "execute_code",
                    {"code": code_block, "language": "python"},
                ),
            ],
        ),
    )

    # ── 12. Execute code result ───────────────────────────────────────────
    print("[11] execute_code result")
    log_tl(
        h,
        _tool_result(
            "tc_code",
            "execute_code",
            {"result": "Knowledge updated successfully.", "notification_sent": True},
        ),
    )

    # ── 13. Response: step-by-step instructions ───────────────────────────
    print("[12] Response: step-by-step instructions")
    log_tl(
        h,
        {
            "role": "assistant",
            "content": (
                "Great, I've noted the project details. Here's the setup process for **unify-prod-2026**:\n\n"
                "**Step 1:** Open [console.cloud.google.com](https://console.cloud.google.com) "
                "and select **unify-production** from the project dropdown.\n\n"
                "**Step 2:** Navigate to **IAM & Admin → Service Accounts** in the left sidebar.\n\n"
                "**Step 3:** Click **'Create Service Account'** at the top.\n"
                "- Name: `unify-drive-access`\n"
                "- ID: will auto-populate\n"
                "- Click **Create and Continue**\n\n"
                "Let me know when you've completed these first three steps."
            ),
        },
    )

    # ── 14. User completes step, asks for next ────────────────────────────
    print("[13] Interjection: steps completed, what's next")
    log_tl(
        h,
        {
            "role": "user",
            "_interjection": True,
            "content": "Done — service account created. Now what?",
        },
    )

    # ── 15. Response: grant role + create key ─────────────────────────────
    print("[14] Response: grant role and create key")
    log_tl(
        h,
        {
            "role": "assistant",
            "content": (
                "**Step 4:** On the 'Grant access' screen, add the role:\n"
                "- Search for **Google Drive API** and select it\n"
                "- Click **Continue**, then **Done**\n\n"
                "**Step 5:** Click on the service account you just created, then:\n"
                "- Go to the **Keys** tab\n"
                "- Click **Add Key → Create new key**\n"
                "- Choose **JSON** format\n"
                "- Click **Create** — this downloads the key file\n\n"
                "**Step 6:** Finally, enable the Drive API:\n"
                "- Go to **APIs & Services → Library**\n"
                "- Search for 'Google Drive API'\n"
                "- Click **Enable**\n\n"
                "Keep the JSON key file safe — we'll need it in a moment."
            ),
        },
    )

    # ── 16. User needs to pause ───────────────────────────────────────────
    print("[15] Interjection: user pauses")
    log_tl(
        h,
        {
            "role": "user",
            "_interjection": True,
            "content": "Hold on, I need to check with DevOps about the key storage policy. Pause for now.",
        },
    )

    # ── 17. Steering: pause ───────────────────────────────────────────────
    print("[16] Steering: pause")
    log_tl(
        h,
        {
            "role": "system",
            "_steering": True,
            "_steering_action": "pause",
            "content": "",
        },
    )

    # ── 18. Steering: resume ──────────────────────────────────────────────
    print("[17] Steering: resume")
    log_tl(
        h,
        {
            "role": "system",
            "_steering": True,
            "_steering_action": "resume",
            "content": "",
        },
    )

    # ── 19. Resume response ───────────────────────────────────────────────
    print("[18] Response: welcome back")
    log_tl(
        h,
        {
            "role": "assistant",
            "content": "Welcome back! Did you get the key storage policy sorted out?",
        },
    )

    # ── 20. User confirms, wants to store the key ─────────────────────────
    print("[19] Interjection: user is back, wants to store key")
    log_tl(
        h,
        {
            "role": "user",
            "_interjection": True,
            "content": (
                "Yes — DevOps says to store the JSON key in our secrets manager. "
                "I've downloaded the key file. Can you store the path in our knowledge base?"
            ),
        },
    )

    # ── 21. send_notification + execute_function (parallel tool calls) ────
    print(
        "[20] Doing: send_notification + execute_function(primitives.knowledge.update)",
    )
    log_tl(
        h,
        _thinking(
            "The user has the JSON key and wants the path stored. Let me notify them "
            "that I'm updating the knowledge base, and do the update in parallel.",
            tool_calls=[
                _tc(
                    "tc_notif",
                    "send_notification",
                    {
                        "message": "Storing credential details and DevOps policy in knowledge base...",
                    },
                ),
                _tc(
                    "tc_kb",
                    "execute_function",
                    {
                        "function_name": "primitives.knowledge.update",
                        "call_kwargs": {
                            "instructions": (
                                "Update the Google Drive credentials entry for project unify-prod-2026: "
                                "JSON key file downloaded and stored in the organization secrets manager "
                                "per DevOps policy. Service account: unify-drive-access. "
                                "Drive API enabled. Setup completed 2026-03-09."
                            ),
                        },
                    },
                ),
            ],
        ),
        tool_aliases={
            "send_notification": "Sending notification",
            "execute_function": "primitives.knowledge.update",
        },
    )

    # ── 22. send_notification result ──────────────────────────────────────
    print("[21] send_notification result")
    log_tl(
        h,
        _tool_result("tc_notif", "send_notification", "Notification sent."),
    )

    # ── 23. Child ManagerMethod for knowledge update ──────────────────────
    kb_suffix = "c3d4"
    kb_h = [*h, f"execute_function(primitives.knowledge.update)({kb_suffix})"]
    kb_cid = str(uuid4())
    log_mm(
        kb_cid,
        kb_h,
        phase="incoming",
        manager="CodeActActor",
        method="execute_function",
        display_label="Running: primitives.knowledge.update",
    )
    log_mm(
        kb_cid,
        kb_h,
        phase="outgoing",
        manager="CodeActActor",
        method="execute_function",
        display_label="Running: primitives.knowledge.update",
        answer="Knowledge base updated with credential details.",
    )

    # ── 24. Knowledge update result ───────────────────────────────────────
    print("[22] Knowledge update result")
    log_tl(
        h,
        _tool_result(
            "tc_kb",
            "execute_function",
            {"result": "Knowledge base updated successfully."},
        ),
    )

    # ── 25. Final response ────────────────────────────────────────────────
    print("[23] Response: setup complete")
    log_tl(
        h,
        {
            "role": "assistant",
            "content": (
                "All done! Here's a summary of what was set up:\n\n"
                "- **Project:** unify-prod-2026 (unify-production)\n"
                "- **Service Account:** unify-drive-access\n"
                "- **Role:** Google Drive API\n"
                "- **Key:** JSON format, stored in secrets manager per DevOps policy\n"
                "- **Drive API:** Enabled\n\n"
                "I've saved all these details in the knowledge base for future reference. "
                "The credentials are ready to use."
            ),
        },
    )

    # ── 26. User confirms done ────────────────────────────────────────────
    print("[24] Interjection: user confirms done")
    log_tl(
        h,
        {
            "role": "user",
            "_interjection": True,
            "content": "Perfect, that's everything. Thanks!",
        },
    )

    # ── 27. Steering: stop ────────────────────────────────────────────────
    print("[25] Steering: stop")
    log_tl(
        h,
        {
            "role": "system",
            "_steering": True,
            "_steering_action": "stop",
            "content": "User confirmed Google Drive credentials setup is complete.",
        },
    )

    # ── 28. ManagerMethod outgoing ────────────────────────────────────────
    print("[26] ManagerMethod outgoing")
    log_mm(
        cid,
        h,
        phase="outgoing",
        display_label="Taking Action",
        answer=(
            "Set up Google Drive API credentials for project unify-prod-2026. "
            "Created service account 'unify-drive-access', enabled Drive API, "
            "and stored configuration details in knowledge base."
        ),
    )

    print("\n=== Done — refresh the console ===\n")


if __name__ == "__main__":
    run()
