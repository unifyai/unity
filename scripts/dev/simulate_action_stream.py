#!/usr/bin/env python3
"""Simulate a CodeActActor.act(persist=True) session for the Console action pane.

Two modes:

  stream (default)  POST events to the Console's local SSE push endpoint with
                    realistic delays so you can watch them unpack in real time.
                    Pass --persist to also write events to Orchestra so that
                    historical data (child expansion, ToolLoop fetch on page
                    refresh) works too.

  upload            Write all events to Orchestra's log API at once (historical
                    path). Refresh the console to see the full session.

The trajectory follows the real CodeActActor flow:
  1. Discovery phase (tool_choice=required): GuidanceManager_search +
     FunctionManager_search_functions are gated before any doing tools.
  2. Doing phase: execute_function / execute_code become available.
  3. Nested inner tool loops (WebSearcher.ask with _search + _extract).
  4. Interjections, notifications, clarification requests, pause/resume/stop.

Prerequisites:
    stream mode:           Console running (http://localhost:3333)
    stream --persist mode: Console + local Orchestra (http://127.0.0.1:8000)
    upload mode:           Local Orchestra running (http://127.0.0.1:8000)

Usage:
    .venv/bin/python scripts/dev/simulate_action_stream.py                   # stream only
    .venv/bin/python scripts/dev/simulate_action_stream.py --persist         # stream + persist
    .venv/bin/python scripts/dev/simulate_action_stream.py upload            # upload only
    .venv/bin/python scripts/dev/simulate_action_stream.py --speed 2         # 2x faster
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

import requests

# =============================================================================
# Configuration
# =============================================================================

ORCHESTRA_BASE = "http://127.0.0.1:8000/v0"
CONSOLE_BASE = "http://localhost:3333"
KEY = "BRE8zK0jon7e7Ix6mtKYZkqO6kIMY0w1QgFFcm3zg8o="
AUTH = {"Authorization": f"Bearer {KEY}"}
PROJECT = "Assistants"
ASSISTANT_ID = "1"
USER_ID = "test-user-001"

MM_CTX = "All/Events/ManagerMethod"
TL_CTX = "All/Events/ToolLoop"


# =============================================================================
# Shared Helpers
# =============================================================================


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


# =============================================================================
# Event Trajectory
# =============================================================================
#
# Each step is a dict with:
#   label     Human-readable description
#   delay     Seconds to wait BEFORE emitting this step (stream mode only)
#   events    List of (context, entries_dict, extra) tuples for upload mode,
#             or pre-shaped SSE dicts for stream mode.
#
# We build the canonical event data once and both modes consume it.


def build_steps():
    """Return the full event trajectory as a list of steps.

    Each step: {"label": str, "delay": float, "events": [dict, ...]}
    Each event dict has two representations populated by the builder:
      "mm"  → (calling_id, hierarchy, kwargs) for ManagerMethod
      "tl"  → (hierarchy, message, kwargs) for ToolLoop
    """
    cid = str(uuid4())
    h = [f"CodeActActor.act({cid[:4]})"]

    ef_suffix = "a1b2"
    ef_h = [*h, f"execute_function(primitives.web.ask)({ef_suffix})"]
    ef_cid = str(uuid4())

    ws_suffix = "ws01"
    ws_h = [*ef_h, f"WebSearcher.ask({ws_suffix})"]
    ws_cid = str(uuid4())
    ws_method = "WebSearcher.ask"

    kb_suffix = "c3d4"
    kb_h = [*h, f"execute_function(primitives.knowledge.update)({kb_suffix})"]
    kb_cid = str(uuid4())

    def mm(calling_id, hierarchy, **kwargs):
        return {
            "kind": "mm",
            "calling_id": calling_id,
            "hierarchy": hierarchy,
            "kwargs": kwargs,
        }

    def tl(hierarchy, message, **kwargs):
        return {
            "kind": "tl",
            "hierarchy": hierarchy,
            "message": message,
            "kwargs": kwargs,
        }

    steps = [
        # ── 1. ManagerMethod incoming ──
        {
            "label": "ManagerMethod incoming",
            "delay": 0,
            "events": [
                mm(
                    cid,
                    h,
                    phase="incoming",
                    display_label="Taking Action",
                    request="Help me set up Google Drive credentials for the unify-production project.",
                    persist=True,
                ),
            ],
        },
        # ── 2. User message ──
        {
            "label": "User message",
            "delay": 0.5,
            "events": [
                tl(
                    h,
                    {
                        "role": "user",
                        "content": "Help me set up Google Drive credentials for the unify-production project.",
                    },
                ),
            ],
        },
        # ── 3. LLM thinking (in flight) + Discovery ──
        {
            "label": "LLM thinking (in flight)",
            "delay": 0.5,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
        {
            "label": "Discovery: GuidanceManager_search + FunctionManager_search_functions",
            "delay": 3.5,
            "events": [
                tl(
                    h,
                    _thinking(
                        "The user wants to set up Google Drive credentials. Before I act, I need "
                        "to check if there's any saved guidance or existing functions for this. "
                        "Let me search both the guidance store and function catalog.",
                        tool_calls=[
                            _tc(
                                "tc_gm",
                                "GuidanceManager_search",
                                {
                                    "query": "Google Drive API credentials service account setup",
                                },
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
                ),
            ],
        },
        # ── 4. GuidanceManager_search result ──
        # First parallel tool completes. The loop eagerly starts an LLM call.
        {
            "label": "GuidanceManager_search result",
            "delay": 2.0,
            "events": [
                tl(
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
                ),
            ],
        },
        # Eager LLM call starts (will be cancelled by second tool completing)
        {
            "label": "LLM thinking (in flight, cancelled by next tool)",
            "delay": 0.3,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
        # ── 5. FunctionManager_search_functions result ──
        # Second parallel tool completes → cancels the in-flight LLM call.
        {
            "label": "FunctionManager_search_functions result",
            "delay": 0.7,
            "events": [
                tl(
                    h,
                    _tool_result(
                        "tc_fm",
                        "FunctionManager_search_functions",
                        {"results": [], "message": "No matching functions found."},
                    ),
                ),
            ],
        },
        # ── 6. LLM thinking (in flight) + web search ──
        # Now both results are in, LLM restarts with full context.
        {
            "label": "LLM thinking (in flight)",
            "delay": 0.5,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
        {
            "label": "Doing: execute_function(primitives.web.ask) — web search",
            "delay": 4.5,
            "events": [
                tl(
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
                ),
            ],
        },
        # ── 7. execute_function boundary: incoming ──
        {
            "label": "execute_function(primitives.web.ask) — boundary incoming",
            "delay": 0.5,
            "events": [
                mm(
                    ef_cid,
                    ef_h,
                    phase="incoming",
                    manager="CodeActActor",
                    method="execute_function",
                    display_label="Running: primitives.web.ask",
                ),
            ],
        },
        # ── 8. WebSearcher.ask: ManagerMethod incoming ──
        {
            "label": "WebSearcher.ask — ManagerMethod incoming",
            "delay": 0.5,
            "events": [
                mm(
                    ws_cid,
                    ws_h,
                    phase="incoming",
                    manager="WebSearcher",
                    method="ask",
                    display_label="Searching the Web",
                    request="Google Cloud service account setup with Drive API access 2026",
                ),
            ],
        },
        # ── 9. WebSearcher inner: user question ──
        {
            "label": "WebSearcher inner: user question",
            "delay": 0.5,
            "events": [
                tl(
                    ws_h,
                    {
                        "role": "user",
                        "content": "Google Cloud service account setup with Drive API access 2026",
                    },
                    method=ws_method,
                ),
            ],
        },
        # ── 10. LLM thinking (in flight) + _search ──
        {
            "label": "LLM thinking (in flight)",
            "delay": 0.5,
            "events": [
                tl(
                    ws_h,
                    {"role": "assistant", "_thinking_in_flight": True},
                    method=ws_method,
                ),
            ],
        },
        {
            "label": "WebSearcher inner: thinking + _search tool call",
            "delay": 2.5,
            "events": [
                tl(
                    ws_h,
                    _thinking(
                        "I need to find current instructions for creating a GCP service account "
                        "with Drive API access. Let me search for the latest documentation.",
                        tool_calls=[
                            _tc(
                                "tc_ws_search",
                                "_search",
                                {
                                    "query": "Google Cloud service account setup Drive API 2026",
                                    "max_results": 5,
                                },
                            ),
                        ],
                    ),
                    method=ws_method,
                    tool_aliases={"_search": "Searching the web"},
                ),
            ],
        },
        # ── 11. WebSearcher inner: _search result ──
        {
            "label": "WebSearcher inner: _search result",
            "delay": 3.0,
            "events": [
                tl(
                    ws_h,
                    _tool_result(
                        "tc_ws_search",
                        "_search",
                        {
                            "answer": (
                                "To create a GCP service account with Drive API access, go to "
                                "IAM & Admin → Service Accounts in the Google Cloud Console."
                            ),
                            "results": [
                                {
                                    "title": "Create a service account | IAM Documentation",
                                    "url": "https://cloud.google.com/iam/docs/service-accounts-create",
                                    "content": "Step-by-step guide for creating service accounts...",
                                },
                                {
                                    "title": "Enable Google Workspace APIs | Google for Developers",
                                    "url": "https://developers.google.com/workspace/guides/enable-apis",
                                    "content": "How to enable APIs including Google Drive API...",
                                },
                            ],
                        },
                    ),
                    method=ws_method,
                ),
            ],
        },
        # ── 12. LLM thinking (in flight) + _extract ──
        {
            "label": "LLM thinking (in flight)",
            "delay": 0.5,
            "events": [
                tl(
                    ws_h,
                    {"role": "assistant", "_thinking_in_flight": True},
                    method=ws_method,
                ),
            ],
        },
        {
            "label": "WebSearcher inner: thinking + _extract tool call",
            "delay": 3.5,
            "events": [
                tl(
                    ws_h,
                    _thinking(
                        "The search gave a good overview but the IAM docs page likely has the most "
                        "up-to-date step-by-step procedure. Let me extract the full content.",
                        tool_calls=[
                            _tc(
                                "tc_ws_extract",
                                "_extract",
                                {
                                    "urls": "https://cloud.google.com/iam/docs/service-accounts-create",
                                },
                            ),
                        ],
                    ),
                    method=ws_method,
                    tool_aliases={"_extract": "Extracting page content"},
                ),
            ],
        },
        # ── 13. WebSearcher inner: _extract result ──
        {
            "label": "WebSearcher inner: _extract result",
            "delay": 4.0,
            "events": [
                tl(
                    ws_h,
                    _tool_result(
                        "tc_ws_extract",
                        "_extract",
                        {
                            "results": [
                                {
                                    "url": "https://cloud.google.com/iam/docs/service-accounts-create",
                                    "raw_content": (
                                        "Create a service account\n\n"
                                        "1. In the Google Cloud console, go to IAM & Admin > Service Accounts.\n"
                                        "2. Click Create service account.\n"
                                        "3. Enter a name and optional description, then click Create and continue.\n"
                                        "4. Grant roles: select the Google Drive API role.\n"
                                        "5. Click Done.\n"
                                        "6. Click the service account → Keys tab → Add Key → Create new key → JSON.\n"
                                        "7. Go to APIs & Services → Library, search 'Google Drive API', click Enable."
                                    ),
                                },
                            ],
                            "failed_results": [],
                        },
                    ),
                    method=ws_method,
                ),
            ],
        },
        # ── 14. WebSearcher inner: final response ──
        {
            "label": "WebSearcher inner: final response",
            "delay": 5.0,
            "events": [
                tl(
                    ws_h,
                    {
                        "role": "assistant",
                        "content": (
                            "As of March 2026, to create a GCP service account with Drive access:\n"
                            "1. Go to console.cloud.google.com\n"
                            "2. Select your project from the dropdown\n"
                            "3. Navigate to IAM & Admin → Service Accounts\n"
                            "4. Click 'Create Service Account'\n"
                            "5. Name it and grant 'Google Drive API' role\n"
                            "6. Create a JSON key under the Keys tab\n"
                            "7. Enable the Drive API under APIs & Services → Library\n\n"
                            "Source: https://cloud.google.com/iam/docs/service-accounts-create"
                        ),
                    },
                    method=ws_method,
                ),
            ],
        },
        # ── 15. WebSearcher.ask: ManagerMethod outgoing ──
        {
            "label": "WebSearcher.ask — ManagerMethod outgoing",
            "delay": 0.5,
            "events": [
                mm(
                    ws_cid,
                    ws_h,
                    phase="outgoing",
                    manager="WebSearcher",
                    method="ask",
                    display_label="Searching the Web",
                    answer=(
                        "As of March 2026, to create a GCP service account with Drive access:\n"
                        "1. Go to console.cloud.google.com\n"
                        "2. Select your project from the dropdown\n"
                        "3. Navigate to IAM & Admin → Service Accounts\n"
                        "4. Click 'Create Service Account'\n"
                        "5. Name it and grant 'Google Drive API' role\n"
                        "6. Create a JSON key under the Keys tab\n"
                        "7. Enable the Drive API under APIs & Services → Library"
                    ),
                ),
            ],
        },
        # ── 16. execute_function boundary: outgoing ──
        {
            "label": "execute_function(primitives.web.ask) — boundary outgoing",
            "delay": 0.3,
            "events": [
                mm(
                    ef_cid,
                    ef_h,
                    phase="outgoing",
                    manager="CodeActActor",
                    method="execute_function",
                    display_label="Running: primitives.web.ask",
                    answer="Found current GCP setup instructions.",
                ),
            ],
        },
        # ── 17. Web search result (parent ToolLoop) ──
        {
            "label": "Web search result (parent ToolLoop)",
            "delay": 0.5,
            "events": [
                tl(
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
                ),
            ],
        },
        # ── 18. LLM thinking (in flight) + clarification ──
        {
            "label": "LLM thinking (in flight)",
            "delay": 0.5,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
        {
            "label": "Doing: request_clarification — which project ID?",
            "delay": 4.5,
            "events": [
                tl(
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
                ),
            ],
        },
        # ── 19. Clarification result ──
        {
            "label": "Clarification result: user confirms project ID",
            "delay": 8.0,
            "events": [
                tl(
                    h,
                    _tool_result(
                        "tc_clar",
                        "request_clarification",
                        "The project ID is 'unify-prod-2026'. The display name is unify-production.",
                    ),
                ),
            ],
        },
        # ── 20. LLM thinking (in flight) + execute_code ──
        {
            "label": "LLM thinking (in flight)",
            "delay": 0.5,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
        {
            "label": "Doing: execute_code — notify + store credentials info",
            "delay": 5.5,
            "events": [
                tl(
                    h,
                    _thinking(
                        "The user confirmed the project ID is 'unify-prod-2026'. Let me send a progress "
                        "notification and store this configuration detail in the knowledge base for future reference.",
                        tool_calls=[
                            _tc(
                                "tc_code",
                                "execute_code",
                                {
                                    "code": (
                                        'notify({"type": "progress", "message": "Starting service account setup for unify-prod-2026..."})\n'
                                        "\n"
                                        "result = await primitives.knowledge.update(\n"
                                        '    instructions="Store the following: GCP project unify-prod-2026 (display name: '
                                        "unify-production) is being configured with a Drive API service account. "
                                        'Setup initiated on 2026-03-09."\n'
                                        ")"
                                    ),
                                    "language": "python",
                                },
                            ),
                        ],
                    ),
                ),
            ],
        },
        # ── 21. Execute code result ──
        {
            "label": "execute_code result",
            "delay": 3.0,
            "events": [
                tl(
                    h,
                    _tool_result(
                        "tc_code",
                        "execute_code",
                        {
                            "result": "Knowledge updated successfully.",
                            "notification_sent": True,
                        },
                    ),
                ),
            ],
        },
        # ── 22. Response: step-by-step instructions ──
        {
            "label": "Response: step-by-step instructions",
            "delay": 6.0,
            "events": [
                tl(
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
                ),
            ],
        },
        # ── 23. Interjection: steps completed ──
        {
            "label": "Interjection: steps completed, what's next",
            "delay": 12.0,
            "events": [
                tl(
                    h,
                    {
                        "role": "user",
                        "_interjection": True,
                        "content": "Done — service account created. Now what?",
                    },
                ),
            ],
        },
        # ── 24. Response: grant role + create key ──
        {
            "label": "Response: grant role and create key",
            "delay": 7.0,
            "events": [
                tl(
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
                ),
            ],
        },
        # ── 25. Interjection: user pauses ──
        {
            "label": "Interjection: user pauses",
            "delay": 15.0,
            "events": [
                tl(
                    h,
                    {
                        "role": "user",
                        "_interjection": True,
                        "content": "Hold on, I need to check with DevOps about the key storage policy. Pause for now.",
                    },
                ),
            ],
        },
        # ── 26. Steering: pause ──
        {
            "label": "Steering: pause",
            "delay": 0.5,
            "events": [
                tl(
                    h,
                    {
                        "role": "system",
                        "_steering": True,
                        "_steering_action": "pause",
                        "content": "",
                    },
                ),
            ],
        },
        # ── 27. Steering: resume ──
        {
            "label": "Steering: resume",
            "delay": 20.0,
            "events": [
                tl(
                    h,
                    {
                        "role": "system",
                        "_steering": True,
                        "_steering_action": "resume",
                        "content": "",
                    },
                ),
            ],
        },
        # ── 28. Response: welcome back ──
        {
            "label": "Response: welcome back",
            "delay": 3.0,
            "events": [
                tl(
                    h,
                    {
                        "role": "assistant",
                        "content": "Welcome back! Did you get the key storage policy sorted out?",
                    },
                ),
            ],
        },
        # ── 29. Interjection: user is back ──
        {
            "label": "Interjection: user is back, wants to store key",
            "delay": 10.0,
            "events": [
                tl(
                    h,
                    {
                        "role": "user",
                        "_interjection": True,
                        "content": (
                            "Yes — DevOps says to store the JSON key in our secrets manager. "
                            "I've downloaded the key file. Can you store the path in our knowledge base?"
                        ),
                    },
                ),
            ],
        },
        # ── 30. LLM thinking (in flight) + parallel tools ──
        {
            "label": "LLM thinking (in flight)",
            "delay": 0.5,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
        {
            "label": "Doing: send_notification + execute_function(primitives.knowledge.update)",
            "delay": 4.5,
            "events": [
                tl(
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
                ),
            ],
        },
        # ── 31. send_notification result ──
        # First parallel tool completes. The loop eagerly starts an LLM call.
        {
            "label": "send_notification result",
            "delay": 0.5,
            "events": [
                tl(
                    h,
                    _tool_result("tc_notif", "send_notification", "Notification sent."),
                ),
            ],
        },
        # Eager LLM call starts (will be cancelled by knowledge update completing)
        {
            "label": "LLM thinking (in flight, cancelled by next tool)",
            "delay": 0.3,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
        # ── 32. Child ManagerMethod for knowledge update ──
        # Second parallel tool completes → cancels the in-flight LLM call.
        {
            "label": "Child ManagerMethod: primitives.knowledge.update",
            "delay": 1.7,
            "events": [
                mm(
                    kb_cid,
                    kb_h,
                    phase="incoming",
                    manager="CodeActActor",
                    method="execute_function",
                    display_label="Running: primitives.knowledge.update",
                ),
                mm(
                    kb_cid,
                    kb_h,
                    phase="outgoing",
                    manager="CodeActActor",
                    method="execute_function",
                    display_label="Running: primitives.knowledge.update",
                    answer="Knowledge base updated with credential details.",
                ),
            ],
        },
        # ── 33. Knowledge update result ──
        {
            "label": "Knowledge update result",
            "delay": 1.0,
            "events": [
                tl(
                    h,
                    _tool_result(
                        "tc_kb",
                        "execute_function",
                        {"result": "Knowledge base updated successfully."},
                    ),
                ),
            ],
        },
        # ── 34. Final response ──
        {
            "label": "Response: setup complete",
            "delay": 6.0,
            "events": [
                tl(
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
                ),
            ],
        },
        # ── 35. Interjection: user confirms done ──
        {
            "label": "Interjection: user confirms done",
            "delay": 8.0,
            "events": [
                tl(
                    h,
                    {
                        "role": "user",
                        "_interjection": True,
                        "content": "Perfect, that's *everything*. Thanks!",
                    },
                ),
            ],
        },
        # ── 36. Steering: stop ──
        {
            "label": "Steering: stop",
            "delay": 1.0,
            "events": [
                tl(
                    h,
                    {
                        "role": "system",
                        "_steering": True,
                        "_steering_action": "stop",
                        "content": "User confirmed Google Drive credentials setup is complete.",
                    },
                ),
            ],
        },
        # ── 37. ManagerMethod outgoing ──
        {
            "label": "ManagerMethod outgoing",
            "delay": 0.5,
            "events": [
                mm(
                    cid,
                    h,
                    phase="outgoing",
                    display_label="Taking Action",
                    answer=(
                        "Set up Google Drive API credentials for project unify-prod-2026. "
                        "Created service account 'unify-drive-access', enabled Drive API, "
                        "and stored configuration details in knowledge base."
                    ),
                ),
            ],
        },
    ]

    return steps


# =============================================================================
# Orchestra Helpers (used by both stream and upload modes)
# =============================================================================


def orchestra_api(method: str, path: str, **kwargs) -> requests.Response:
    return getattr(requests, method)(f"{ORCHESTRA_BASE}{path}", headers=AUTH, **kwargs)


def setup_orchestra() -> None:
    projects = orchestra_api("get", "/projects").json()
    if PROJECT not in projects:
        orchestra_api("post", "/project", json={"name": PROJECT})
        print(f"  Created project '{PROJECT}'")
    for ctx in [MM_CTX, TL_CTX]:
        orchestra_api("post", "/context", json={"project_name": PROJECT, "name": ctx})
    print("  Contexts ready")


def upload_log_entry(context: str, entries: dict) -> None:
    entries["_user_id"] = USER_ID
    entries["_assistant_id"] = ASSISTANT_ID
    r = orchestra_api(
        "post",
        "/logs",
        json={"project_name": PROJECT, "context": context, "entries": entries},
    )
    if not r.ok:
        print(f"    LOG ERROR: {r.status_code} {r.text[:200]}")


def _event_to_upload(event: dict) -> tuple[str, dict]:
    """Convert a step event dict into (context, entries) for Orchestra upload."""
    if event["kind"] == "mm":
        kw = event["kwargs"]
        entries = {
            "calling_id": event["calling_id"],
            "event_id": str(uuid4()),
            "event_timestamp": now_iso(),
            "manager": kw.get("manager", "CodeActActor"),
            "method": kw.get("method", "act"),
            "phase": kw["phase"],
            "hierarchy": event["hierarchy"],
            "hierarchy_label": "->".join(event["hierarchy"]),
            "status": kw.get("status", "ok"),
        }
        for field in ("persist", "display_label", "request", "answer"):
            val = kw.get(field)
            if val is not None:
                entries[field] = val
        return MM_CTX, entries
    else:
        kw = event["kwargs"]
        entries = {
            "event_id": str(uuid4()),
            "event_timestamp": now_iso(),
            "message": event["message"],
            "method": kw.get("method", "CodeActActor.act"),
            "hierarchy": event["hierarchy"],
            "hierarchy_label": "->".join(event["hierarchy"]),
        }
        if "tool_aliases" in kw:
            entries["tool_aliases"] = kw["tool_aliases"]
        return TL_CTX, entries


# =============================================================================
# Stream Mode
# =============================================================================

_row_counter = 0


def _next_row_id() -> int:
    global _row_counter
    _row_counter += 1
    return _row_counter


def _event_to_sse(event: dict) -> dict:
    """Convert a step event dict into the pre-shaped SSE format the push endpoint expects."""
    if event["kind"] == "mm":
        kw = event["kwargs"]
        entries = {
            "callingId": event["calling_id"],
            "eventId": str(uuid4()),
            "manager": kw.get("manager", "CodeActActor"),
            "method": kw.get("method", "act"),
            "phase": kw["phase"],
            "hierarchy": event["hierarchy"],
            "hierarchyLabel": "->".join(event["hierarchy"]),
            "status": kw.get("status", "ok"),
        }
        for field in ("display_label", "request", "answer", "persist"):
            val = kw.get(field)
            if val is not None:
                camel = (
                    field.replace("_l", "L")
                    .replace("_a", "A")
                    .replace("_r", "R")
                    .replace("_p", "P")
                )
                if field == "display_label":
                    camel = "displayLabel"
                entries[camel] = val
        return {
            "type": "ManagerMethod",
            "data": {"id": _next_row_id(), "ts": now_iso(), "entries": entries},
        }
    else:
        kw = event["kwargs"]
        entries = {
            "eventId": str(uuid4()),
            "eventTimestamp": now_iso(),
            "message": event["message"],
            "method": kw.get("method", "CodeActActor.act"),
            "hierarchy": event["hierarchy"],
            "hierarchyLabel": "->".join(event["hierarchy"]),
        }
        if "tool_aliases" in kw:
            entries["toolAliases"] = kw["tool_aliases"]
        return {
            "type": "ToolLoop",
            "data": {
                "id": _next_row_id(),
                "ts": entries["eventTimestamp"],
                "entries": entries,
            },
        }


def push_to_console(assistant_id: str, sse_event: dict) -> None:
    r = requests.post(
        f"{CONSOLE_BASE}/api/assistant/{assistant_id}/actions/push",
        json=sse_event,
    )
    if not r.ok:
        print(f"    PUSH ERROR: {r.status_code} {r.text[:200]}")


def run_stream(speed: float, persist: bool = False) -> None:
    total_time = sum(s["delay"] for s in build_steps()) / speed
    mode_label = "stream + persist" if persist else "stream"
    print(f"\n=== {mode_label} (speed={speed}x, ~{total_time:.0f}s total) ===\n")

    try:
        requests.get(f"{CONSOLE_BASE}", timeout=3)
    except requests.ConnectionError:
        print(f"ERROR: Console not reachable at {CONSOLE_BASE}")
        print("Start it with: console/scripts/local.sh")
        sys.exit(1)

    # Ensure the Orchestra project and contexts exist so the console's initial
    # load doesn't 404 (which blocks SSE connection setup).
    setup_orchestra()
    print()

    steps = build_steps()

    for i, step in enumerate(steps, 1):
        delay = step["delay"] / speed
        if delay > 0 and i > 1:
            time.sleep(delay)

        for event in step["events"]:
            sse = _event_to_sse(event)
            kind = sse["type"]
            entries = sse["data"]["entries"]
            phase = entries.get("phase", "")
            role = ""
            msg = entries.get("message")
            if isinstance(msg, dict):
                role = msg.get("role", "")
                if msg.get("_steering"):
                    role = f"steering:{msg.get('_steering_action')}"
            detail = phase or role
            print(f"  [{i:2d}] {kind:<16s} {detail:<20s} {step['label']}")
            push_to_console(ASSISTANT_ID, sse)

            if persist:
                ctx, upload_entries = _event_to_upload(event)
                upload_log_entry(ctx, upload_entries)

    print(f"\n=== Done — {len(steps)} steps streamed ===\n")


# =============================================================================
# Upload Mode
# =============================================================================


def run_upload() -> None:
    print("\n--- Setup ---")
    setup_orchestra()

    steps = build_steps()
    print(f"\n=== Uploading {len(steps)} steps to Orchestra ===\n")

    for i, step in enumerate(steps, 1):
        for event in step["events"]:
            ctx, entries = _event_to_upload(event)
            upload_log_entry(ctx, entries)

            kind = "ManagerMethod" if event["kind"] == "mm" else "ToolLoop"
            phase = event.get("kwargs", {}).get("phase", "")
            role = ""
            msg = event.get("message")
            if isinstance(msg, dict):
                role = msg.get("role", "")
                if msg.get("_steering"):
                    role = f"steering:{msg.get('_steering_action')}"
            detail = phase or role
            print(f"  [{i:2d}] {kind:<16s} {detail:<20s} {step['label']}")

    print("\n=== Done — refresh the console ===\n")


# =============================================================================
# Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Simulate a CodeActActor.act session for the Console action pane",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="stream",
        choices=["stream", "upload"],
        help="stream (default): real-time SSE via Console push endpoint. "
        "upload: write all events to Orchestra at once.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier for stream mode (default: 1.0). "
        "Use 2 for 2x faster, 0.5 for slower, etc.",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="(stream mode) Also write events to Orchestra for historical "
        "persistence (child expansion, ToolLoop fetch on page refresh).",
    )
    args = parser.parse_args()

    if args.mode == "upload":
        run_upload()
    else:
        run_stream(args.speed, persist=args.persist)


if __name__ == "__main__":
    main()
