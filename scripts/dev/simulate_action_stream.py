#!/usr/bin/env python3
"""Simulate a CodeActActor.act(persist=True) session by writing events to local Orchestra.

Writes ManagerMethod and ToolLoop log entries directly to the local Orchestra
database via its REST API. The console's action pane loads these on page
refresh via its historical polling path.

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
    display_label=None,
    request=None,
    answer=None,
    status="ok",
):
    entries = {
        "calling_id": calling_id,
        "event_id": str(uuid4()),
        "event_timestamp": now_iso(),
        "manager": "CodeActActor",
        "method": "act",
        "phase": phase,
        "hierarchy": hierarchy,
        "hierarchy_label": "->".join(hierarchy),
        "status": status,
    }
    if display_label:
        entries["display_label"] = display_label
    if request:
        entries["request"] = request
    if answer:
        entries["answer"] = answer
    log_entry(MM_CTX, entries)
    print(f"  ManagerMethod: {phase}")


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


def run() -> None:
    print("\n--- Setup ---")
    setup()

    cid = str(uuid4())
    h = [f"CodeActActor.act({cid[:4]})"]

    print(f"\n=== Simulating persist=True session ===")
    print(f"  calling_id: {cid}")
    print(f"  hierarchy:  {h[0]}\n")

    print("[1] ManagerMethod incoming")
    log_mm(
        cid,
        h,
        phase="incoming",
        display_label="Taking Action",
        request="Help me set up Google Drive credentials step by step.",
    )

    print("[2] User message")
    log_tl(
        h,
        {
            "role": "user",
            "content": "Help me set up Google Drive credentials step by step.",
        },
    )

    print("[3] Assistant tool call")
    log_tl(
        h,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "execute_function",
                        "arguments": json.dumps(
                            {
                                "function_name": "primitives.web.ask",
                                "call_kwargs": {
                                    "text": "Google Cloud service account setup 2026",
                                },
                            },
                        ),
                    },
                },
            ],
        },
        tool_aliases={"execute_function": "Searching the web"},
    )

    print("[4] Tool result")
    log_tl(
        h,
        {
            "role": "tool",
            "tool_call_id": "tc1",
            "name": "execute_function",
            "content": json.dumps(
                {
                    "answer": "console.cloud.google.com -> IAM & Admin -> Service Accounts",
                },
            ),
        },
    )

    print("[5] send_response #1")
    log_tl(
        h,
        {
            "role": "assistant",
            "content": "Here's how to set up Google Drive credentials:\n\n"
            "1. Open console.cloud.google.com\n2. Go to IAM & Admin -> Service Accounts\n"
            "3. Click 'Create Service Account'\n4. Name it 'unify-drive-access'\n"
            "5. Click 'Create and Continue'\n\nLet me know when you've completed step 1.",
        },
    )

    print("[6] Interjection")
    log_tl(
        h,
        {
            "role": "user",
            "content": "Done! I'm on the Service Accounts page. What's next?",
        },
    )

    print("[7] send_response #2")
    log_tl(
        h,
        {
            "role": "assistant",
            "content": "Click 'Create Service Account' at the top.\n\n- **Name:** unify-drive-access\n- Click 'Create and Continue'",
        },
    )

    print("[8] Interjection (pause)")
    log_tl(
        h,
        {
            "role": "user",
            "content": "Hold on, I need to check something. Can you pause?",
        },
    )

    print("[9] Steering: pause")
    log_tl(
        h,
        {
            "role": "system",
            "_steering": True,
            "_steering_action": "pause",
            "content": "",
        },
    )

    print("[10] Steering: resume")
    log_tl(
        h,
        {
            "role": "system",
            "_steering": True,
            "_steering_action": "resume",
            "content": "",
        },
    )

    print("[11] send_response #3")
    log_tl(
        h,
        {
            "role": "assistant",
            "content": "Welcome back! Have you created the service account yet?",
        },
    )

    print("[12] Interjection (done)")
    log_tl(
        h,
        {
            "role": "user",
            "content": "Yes, all done. Credentials saved. You can stop now.",
        },
    )

    print("[13] Steering: stop")
    log_tl(
        h,
        {
            "role": "system",
            "_steering": True,
            "_steering_action": "stop",
            "content": "User confirmed setup is complete",
        },
    )

    print("[14] ManagerMethod outgoing")
    log_mm(
        cid,
        h,
        phase="outgoing",
        display_label="Taking Action",
        answer="Google Drive credentials set up successfully.",
    )

    print("\n=== Done — refresh the console ===\n")


if __name__ == "__main__":
    run()
