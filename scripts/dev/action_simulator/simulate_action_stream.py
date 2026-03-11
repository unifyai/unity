#!/usr/bin/env python3
"""Simulate CodeActActor.act sessions for the Console action pane.

Scenarios (--scenario):

  persistent (default)  A long-running act(persist=True) session with discovery,
                        nested WebSearcher.ask, clarification, interjections,
                        pause/resume/stop, and parallel tool completion.

  single_action         A one-shot act(persist=False) action that delegates to a
                        sub-agent, with a post-completion StorageCheck phase.

Delivery (--stream / --save):

  --stream    POST events via SSE to the Console with realistic delays.
  --save      Write events to Orchestra for historical access / page refresh.

  Both can be combined. If neither is given, --stream is the default.
  --speed only applies when --stream is active.

Prerequisites:
    --stream only:     Console running (http://localhost:3333)
    --save only:       Local Orchestra running (http://127.0.0.1:8000)
    --stream --save:   Both Console and Orchestra running

Usage:
    .venv/bin/python scripts/dev/action_simulator/simulate_action_stream.py                            # stream persistent
    .venv/bin/python scripts/dev/action_simulator/simulate_action_stream.py --scenario single_action   # stream single action
    .venv/bin/python scripts/dev/action_simulator/simulate_action_stream.py --save                     # save only (no streaming)
    .venv/bin/python scripts/dev/action_simulator/simulate_action_stream.py --stream --save            # stream + save
    .venv/bin/python scripts/dev/action_simulator/simulate_action_stream.py --speed 2                  # 2x faster streaming
    .venv/bin/python scripts/dev/action_simulator/simulate_action_stream.py --clear                    # wipe old events first
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
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


def build_persistent_steps():
    """Persistent session: act(persist=True) with interjections and steering.

    Each step: {"label": str, "delay": float, "events": [dict, ...]}
    Each event dict has two representations populated by the builder:
      "mm"  → (calling_id, hierarchy, kwargs) for ManagerMethod
      "tl"  → (hierarchy, message, kwargs) for ToolLoop
    """
    cid = str(uuid4())
    h = [f"CodeActActor.act({cid[:4]})"]

    # Inner managers share the root lineage — execute_function is just a tool,
    # not a boundary. Nesting emerges from the managers' own events.
    ws_suffix = "ws01"
    ws_h = [*h, f"WebSearcher.ask({ws_suffix})"]
    ws_cid = str(uuid4())
    ws_method = "WebSearcher.ask"

    kb_suffix = "c3d4"
    kb_h = [*h, f"KnowledgeManager.update({kb_suffix})"]
    kb_cid = str(uuid4())

    # Inner primitives spawned by execute_code share the parent lineage
    # (no intermediate execute_code boundary — code is just a tool).
    ec_kb_suffix = "g7h8"
    ec_kb_h = [*h, f"KnowledgeManager.update({ec_kb_suffix})"]
    ec_kb_cid = str(uuid4())
    ec_ct_suffix = "i9j0"
    ec_ct_h = [*h, f"ContactManager.update({ec_ct_suffix})"]
    ec_ct_cid = str(uuid4())

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
                    display_label="Session",
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
        # ── 7. WebSearcher.ask: ManagerMethod incoming ──
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
        # ── 14. LLM thinking (in flight) + WebSearcher final response ──
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
            "label": "WebSearcher inner: final response",
            "delay": 4.5,
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
        # ── 16. Web search result (parent ToolLoop) ──
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
        # ── 20. LLM thinking (in flight) + execute_code (multi-child) ──
        {
            "label": "LLM thinking (in flight)",
            "delay": 0.5,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
        {
            "label": "Doing: execute_code — store credentials + update contacts",
            "delay": 5.5,
            "events": [
                tl(
                    h,
                    _thinking(
                        "The user confirmed the project ID is 'unify-prod-2026'. I need to store "
                        "the configuration in the knowledge base AND update the contact record for "
                        "the DevOps team with the new credential details. Let me do both in one block.",
                        tool_calls=[
                            _tc(
                                "tc_code",
                                "execute_code",
                                {
                                    "code": (
                                        "kb_result = await primitives.knowledge.update(\n"
                                        '    instructions="Store the following: GCP project unify-prod-2026 '
                                        "(display name: unify-production) is being configured with a Drive API "
                                        'service account. Setup initiated on 2026-03-09."\n'
                                        ")\n"
                                        "\n"
                                        "ct_result = await primitives.contacts.update(\n"
                                        '    instructions="Update the DevOps team contact with the new '
                                        "Google Drive service account credentials for project "
                                        'unify-prod-2026."\n'
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
        # ── 21–32. Concurrent: KnowledgeManager.update + ContactManager.update ──
        # Both fire simultaneously and their inner events interleave, as they
        # would in real concurrent tool execution.
        #
        # Timeline:
        #   0.0s  KM incoming + CM incoming (both start)
        #   0.3s  KM user request
        #   0.5s  CM user request
        #   0.8s  KM thinking + _filter
        #   0.6s  CM thinking + _filter
        #   0.8s  KM _filter result
        #   0.5s  CM _filter result
        #   0.6s  KM thinking + _insert
        #   0.5s  CM thinking + _update
        #   0.8s  KM _insert result → KM outgoing
        #   0.6s  CM _update result → CM outgoing
        {
            "label": "Inner: KM + CM both incoming (concurrent start)",
            "delay": 1.5,
            "events": [
                mm(
                    ec_kb_cid,
                    ec_kb_h,
                    phase="incoming",
                    manager="KnowledgeManager",
                    method="update",
                    display_label="Updating Knowledge Base",
                    request="Store GCP project unify-prod-2026 credential details.",
                ),
                mm(
                    ec_ct_cid,
                    ec_ct_h,
                    phase="incoming",
                    manager="ContactManager",
                    method="update",
                    display_label="Updating Contacts",
                    request="Update DevOps team contact with Drive credentials.",
                ),
            ],
        },
        {
            "label": "KM inner: user request",
            "delay": 0.3,
            "events": [
                tl(
                    ec_kb_h,
                    {
                        "role": "user",
                        "content": "Store GCP project unify-prod-2026 credential details.",
                    },
                    method="KnowledgeManager.update",
                ),
            ],
        },
        {
            "label": "CM inner: user request",
            "delay": 0.5,
            "events": [
                tl(
                    ec_ct_h,
                    {
                        "role": "user",
                        "content": "Update DevOps team contact with Drive credentials for unify-prod-2026.",
                    },
                    method="ContactManager.update",
                ),
            ],
        },
        {
            "label": "KM inner: thinking + _filter",
            "delay": 0.8,
            "events": [
                tl(
                    ec_kb_h,
                    _thinking(
                        "I need to check if there's already an entry for this GCP project "
                        "before inserting a new row.",
                        tool_calls=[
                            _tc(
                                "tc_km_filter",
                                "_filter",
                                {
                                    "table": "Credentials",
                                    "filter": "project == 'unify-prod-2026'",
                                },
                            ),
                        ],
                    ),
                    method="KnowledgeManager.update",
                    tool_aliases={"_filter": "Searching knowledge base"},
                ),
            ],
        },
        {
            "label": "CM inner: thinking + _filter",
            "delay": 0.6,
            "events": [
                tl(
                    ec_ct_h,
                    _thinking(
                        "I need to find the DevOps team contact to update their record "
                        "with the new credential information.",
                        tool_calls=[
                            _tc(
                                "tc_cm_filter",
                                "_filter",
                                {"table": "Contacts", "filter": "team == 'DevOps'"},
                            ),
                        ],
                    ),
                    method="ContactManager.update",
                    tool_aliases={"_filter": "Searching contacts"},
                ),
            ],
        },
        {
            "label": "KM inner: _filter result",
            "delay": 0.8,
            "events": [
                tl(
                    ec_kb_h,
                    _tool_result("tc_km_filter", "_filter", {"rows": [], "count": 0}),
                    method="KnowledgeManager.update",
                ),
            ],
        },
        {
            "label": "CM inner: _filter result",
            "delay": 0.5,
            "events": [
                tl(
                    ec_ct_h,
                    _tool_result(
                        "tc_cm_filter",
                        "_filter",
                        {
                            "rows": [
                                {
                                    "name": "DevOps Team",
                                    "email": "devops@unify.ai",
                                    "role": "Infrastructure",
                                },
                            ],
                            "count": 1,
                        },
                    ),
                    method="ContactManager.update",
                ),
            ],
        },
        {
            "label": "KM inner: thinking + _insert",
            "delay": 0.6,
            "events": [
                tl(
                    ec_kb_h,
                    _thinking(
                        "No existing entry found. I'll insert a new credentials row.",
                        tool_calls=[
                            _tc(
                                "tc_km_insert",
                                "_insert",
                                {
                                    "table": "Credentials",
                                    "row": {
                                        "project": "unify-prod-2026",
                                        "display_name": "unify-production",
                                        "service_account": "unify-drive-access",
                                        "api": "Google Drive",
                                        "key_format": "JSON",
                                        "status": "active",
                                    },
                                },
                            ),
                        ],
                    ),
                    method="KnowledgeManager.update",
                    tool_aliases={"_insert": "Inserting row"},
                ),
            ],
        },
        {
            "label": "CM inner: thinking + _update",
            "delay": 0.5,
            "events": [
                tl(
                    ec_ct_h,
                    _thinking(
                        "Found the DevOps team contact. I'll update their record with "
                        "the new Google Drive credential details.",
                        tool_calls=[
                            _tc(
                                "tc_cm_update",
                                "_update",
                                {
                                    "table": "Contacts",
                                    "filter": "team == 'DevOps'",
                                    "set": {
                                        "notes": "Google Drive service account (unify-drive-access) "
                                        "configured for project unify-prod-2026. JSON key in secrets manager.",
                                    },
                                },
                            ),
                        ],
                    ),
                    method="ContactManager.update",
                    tool_aliases={"_update": "Updating contact"},
                ),
            ],
        },
        {
            "label": "KM inner: _insert result → KM outgoing",
            "delay": 0.8,
            "events": [
                tl(
                    ec_kb_h,
                    _tool_result("tc_km_insert", "_insert", {"inserted": 1}),
                    method="KnowledgeManager.update",
                ),
                mm(
                    ec_kb_cid,
                    ec_kb_h,
                    phase="outgoing",
                    manager="KnowledgeManager",
                    method="update",
                    display_label="Updating Knowledge Base",
                    answer="Knowledge base updated with credential details.",
                ),
            ],
        },
        {
            "label": "CM inner: _update result → CM outgoing",
            "delay": 0.6,
            "events": [
                tl(
                    ec_ct_h,
                    _tool_result("tc_cm_update", "_update", {"updated": 1}),
                    method="ContactManager.update",
                ),
                mm(
                    ec_ct_cid,
                    ec_ct_h,
                    phase="outgoing",
                    manager="ContactManager",
                    method="update",
                    display_label="Updating Contacts",
                    answer="DevOps team contact updated with new credential details.",
                ),
            ],
        },
        # ── 23. execute_code result (parent ToolLoop) ──
        {
            "label": "execute_code result",
            "delay": 0.5,
            "events": [
                tl(
                    h,
                    _tool_result(
                        "tc_code",
                        "execute_code",
                        {
                            "result": "Knowledge updated and contacts updated successfully.",
                            "notification_sent": True,
                        },
                    ),
                ),
            ],
        },
        # ── 22. LLM thinking (in flight) + response ──
        {
            "label": "LLM thinking (in flight)",
            "delay": 0.5,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
        {
            "label": "Response: step-by-step instructions",
            "delay": 5.5,
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
        # ── 24. LLM thinking (in flight) + response ──
        {
            "label": "LLM thinking (in flight)",
            "delay": 0.5,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
        {
            "label": "Response: grant role and create key",
            "delay": 6.5,
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
        # ── 26. LLM thinking (in flight) → decides to pause ──
        {
            "label": "LLM thinking (in flight)",
            "delay": 0.5,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
        {
            "label": "Steering: pause",
            "delay": 2.0,
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
        # ── 28. LLM thinking (in flight) + response ──
        {
            "label": "LLM thinking (in flight)",
            "delay": 0.5,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
        {
            "label": "Response: welcome back",
            "delay": 2.5,
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
        # ── 32–35. Child: KnowledgeManager.update (with simple tool loop) ──
        # Second parallel tool completes → cancels the in-flight LLM call.
        {
            "label": "Child: KnowledgeManager.update incoming",
            "delay": 1.7,
            "events": [
                mm(
                    kb_cid,
                    kb_h,
                    phase="incoming",
                    manager="KnowledgeManager",
                    method="update",
                    display_label="Updating Knowledge Base",
                    request="Update credential details for unify-prod-2026.",
                ),
            ],
        },
        {
            "label": "KB2 inner: user request",
            "delay": 0.3,
            "events": [
                tl(
                    kb_h,
                    {
                        "role": "user",
                        "content": "Update credential details for unify-prod-2026.",
                    },
                    method="KnowledgeManager.update",
                ),
            ],
        },
        {
            "label": "KB2 inner: thinking + _update",
            "delay": 0.8,
            "events": [
                tl(
                    kb_h,
                    _thinking(
                        "I need to update the existing credentials entry with the "
                        "confirmed key storage details from DevOps.",
                        tool_calls=[
                            _tc(
                                "tc_kb2_update",
                                "_update",
                                {
                                    "table": "Credentials",
                                    "filter": "project == 'unify-prod-2026'",
                                    "set": {
                                        "key_storage": "secrets manager",
                                        "devops_approved": True,
                                    },
                                },
                            ),
                        ],
                    ),
                    method="KnowledgeManager.update",
                    tool_aliases={"_update": "Updating row"},
                ),
            ],
        },
        {
            "label": "KB2 inner: _update result → outgoing",
            "delay": 1.0,
            "events": [
                tl(
                    kb_h,
                    _tool_result("tc_kb2_update", "_update", {"updated": 1}),
                    method="KnowledgeManager.update",
                ),
                mm(
                    kb_cid,
                    kb_h,
                    phase="outgoing",
                    manager="KnowledgeManager",
                    method="update",
                    display_label="Updating Knowledge Base",
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
        # ── 34. LLM thinking (in flight) + final response ──
        {
            "label": "LLM thinking (in flight)",
            "delay": 0.5,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
        {
            "label": "Response: setup complete",
            "delay": 5.5,
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
        # ── 36. LLM thinking (in flight) → decides to stop ──
        {
            "label": "LLM thinking (in flight)",
            "delay": 0.5,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
        {
            "label": "Steering: stop",
            "delay": 0.5,
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
                    display_label="Session",
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


def build_single_action_steps():
    """Single action: act(persist=False) that delegates to a sub-agent.

    Hierarchy structure (5 nesting levels):
      CodeActActor.act(root)
        → execute_function(primitives.actor.act)(ef1)
          → CodeActActor.act(sub)           [sub-agent]
            → execute_function(primitives.contacts.ask)(ef2)
              → ContactManager.ask(cm)      [inner tool loop]
    """
    cid = str(uuid4())
    h = [f"CodeActActor.act({cid[:4]})"]

    # StorageCheck runs after the doing loop completes. TOOL_LOOP_LINEAGE
    # is [] in act()'s context (the inner loop set it in its own task), so
    # the StorageCheck is a root-level node, adjacent to the action.
    sc_suffix = "sc01"
    sc_h = [f"StorageCheck(CodeActActor.act)({sc_suffix})"]
    sc_cid = str(uuid4())
    sc_method = "StorageCheck(CodeActActor.act)"

    # Sub-agent dispatch: execute_function(primitives.actor.act)
    ef1_suffix = "d7e8"
    ef1_h = [*h, f"execute_function(primitives.actor.act)({ef1_suffix})"]
    ef1_cid = str(uuid4())

    # Sub-agent's own CodeActActor.act
    sa_cid = str(uuid4())
    sa_h = [*ef1_h, f"CodeActActor.act({sa_cid[:4]})"]

    # Sub-agent's inner call: execute_function(primitives.contacts.ask)
    ef2_suffix = "f9a0"
    ef2_h = [*sa_h, f"execute_function(primitives.contacts.ask)({ef2_suffix})"]
    ef2_cid = str(uuid4())

    # ContactManager.ask inside the sub-agent
    cm_suffix = "b1c2"
    cm_h = [*ef2_h, f"ContactManager.ask({cm_suffix})"]
    cm_cid = str(uuid4())
    cm_method = "ContactManager.ask"

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
        # ── 1. Root ManagerMethod incoming ──
        {
            "label": "ManagerMethod incoming (persist=False)",
            "delay": 0,
            "events": [
                mm(
                    cid,
                    h,
                    phase="incoming",
                    display_label="Taking Action",
                    request="Find the main contact at Acme Corp and draft a follow-up email about the Q1 partnership review.",
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
                        "content": "Find the main contact at Acme Corp and draft a follow-up email about the Q1 partnership review.",
                    },
                ),
            ],
        },
        # ── 3. Discovery ──
        {
            "label": "LLM thinking (in flight)",
            "delay": 0.5,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
        {
            "label": "Discovery: GuidanceManager_search + FunctionManager_search_functions",
            "delay": 3.0,
            "events": [
                tl(
                    h,
                    _thinking(
                        "The user wants to find a contact at Acme Corp and draft a follow-up email. "
                        "Let me check for any relevant guidance or saved functions first.",
                        tool_calls=[
                            _tc(
                                "tc_gm",
                                "GuidanceManager_search",
                                {"query": "email drafting follow-up partnership"},
                            ),
                            _tc(
                                "tc_fm",
                                "FunctionManager_search_functions",
                                {"query": "draft email contact lookup"},
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
        # ── 4. Discovery results ──
        {
            "label": "GuidanceManager_search result",
            "delay": 1.5,
            "events": [
                tl(
                    h,
                    _tool_result(
                        "tc_gm",
                        "GuidanceManager_search",
                        {
                            "results": [
                                {
                                    "id": 17,
                                    "title": "Professional Email Templates",
                                    "summary": "Standard templates for follow-up, introduction, "
                                    "and partnership review emails. Includes tone guidelines.",
                                },
                            ],
                        },
                    ),
                ),
            ],
        },
        {
            "label": "LLM thinking (in flight, cancelled by next tool)",
            "delay": 0.3,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
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
        # ── 5. Delegate to sub-agent ──
        {
            "label": "LLM thinking (in flight)",
            "delay": 0.5,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
        {
            "label": "Doing: execute_function(primitives.actor.act) — spawn sub-agent",
            "delay": 4.0,
            "events": [
                tl(
                    h,
                    _thinking(
                        "I found email templates guidance. Now I need to look up the Acme Corp "
                        "contact. This is a self-contained research task — let me delegate it to "
                        "a sub-agent so it can focus on the contact lookup while I prepare the "
                        "email structure.",
                        tool_calls=[
                            _tc(
                                "tc_subagent",
                                "execute_function",
                                {
                                    "function_name": "primitives.actor.act",
                                    "call_kwargs": {
                                        "request": "Find the main contact person at Acme Corp. "
                                        "I need their full name, role, and email address.",
                                    },
                                },
                            ),
                        ],
                    ),
                    tool_aliases={"execute_function": "primitives.actor.act"},
                ),
            ],
        },
        # ── 6. execute_function(primitives.actor.act) boundary incoming ──
        {
            "label": "execute_function(primitives.actor.act) — boundary incoming",
            "delay": 0.5,
            "events": [
                mm(
                    ef1_cid,
                    ef1_h,
                    phase="incoming",
                    manager="CodeActActor",
                    method="execute_function",
                    display_label="Running: primitives.actor.act",
                ),
            ],
        },
        # ── 7. Sub-agent CodeActActor.act incoming ──
        {
            "label": "Sub-agent CodeActActor.act — ManagerMethod incoming",
            "delay": 0.5,
            "events": [
                mm(
                    sa_cid,
                    sa_h,
                    phase="incoming",
                    display_label="Taking Action",
                    request="Find the main contact person at Acme Corp. "
                    "I need their full name, role, and email address.",
                ),
            ],
        },
        # ── 8. Sub-agent ToolLoop: user message ──
        {
            "label": "Sub-agent: user message",
            "delay": 0.5,
            "events": [
                tl(
                    sa_h,
                    {
                        "role": "user",
                        "content": "Find the main contact person at Acme Corp. "
                        "I need their full name, role, and email address.",
                    },
                ),
            ],
        },
        # ── 9. Sub-agent: discovery (skipped for brevity — sub-agents
        #       inherit scoped discovery from parent) ──
        # ── 10. Sub-agent: thinking + contacts.ask ──
        {
            "label": "Sub-agent: LLM thinking (in flight)",
            "delay": 0.5,
            "events": [
                tl(sa_h, {"role": "assistant", "_thinking_in_flight": True}),
            ],
        },
        {
            "label": "Sub-agent: thinking + execute_function(primitives.contacts.ask)",
            "delay": 3.0,
            "events": [
                tl(
                    sa_h,
                    _thinking(
                        "I need to look up the main contact at Acme Corp in the contact book. "
                        "Let me search for them.",
                        tool_calls=[
                            _tc(
                                "tc_sa_contacts",
                                "execute_function",
                                {
                                    "function_name": "primitives.contacts.ask",
                                    "call_kwargs": {
                                        "text": "Who is the main contact at Acme Corp?",
                                    },
                                },
                            ),
                        ],
                    ),
                    tool_aliases={"execute_function": "primitives.contacts.ask"},
                ),
            ],
        },
        # ── 11. Sub-agent's contacts.ask boundary incoming ──
        {
            "label": "Sub-agent: contacts.ask — boundary incoming",
            "delay": 0.3,
            "events": [
                mm(
                    ef2_cid,
                    ef2_h,
                    phase="incoming",
                    manager="CodeActActor",
                    method="execute_function",
                    display_label="Running: primitives.contacts.ask",
                ),
            ],
        },
        # ── 12. ContactManager.ask incoming ──
        {
            "label": "ContactManager.ask — ManagerMethod incoming",
            "delay": 0.3,
            "events": [
                mm(
                    cm_cid,
                    cm_h,
                    phase="incoming",
                    manager="ContactManager",
                    method="ask",
                    display_label="Checking Contact Book",
                    request="Who is the main contact at Acme Corp?",
                ),
            ],
        },
        # ── 13. ContactManager.ask inner ToolLoop ──
        {
            "label": "ContactManager.ask: user message",
            "delay": 0.3,
            "events": [
                tl(
                    cm_h,
                    {
                        "role": "user",
                        "content": "Who is the main contact at Acme Corp?",
                    },
                    method=cm_method,
                ),
            ],
        },
        {
            "label": "ContactManager.ask: LLM thinking (in flight)",
            "delay": 0.5,
            "events": [
                tl(
                    cm_h,
                    {"role": "assistant", "_thinking_in_flight": True},
                    method=cm_method,
                ),
            ],
        },
        {
            "label": "ContactManager.ask: thinking + _filter tool call",
            "delay": 2.0,
            "events": [
                tl(
                    cm_h,
                    _thinking(
                        "Let me search the contacts database for anyone associated with Acme Corp.",
                        tool_calls=[
                            _tc(
                                "tc_cm_filter",
                                "_filter",
                                {
                                    "filter_expression": "company == 'Acme Corp'",
                                    "columns": ["name", "role", "email", "company"],
                                },
                            ),
                        ],
                    ),
                    method=cm_method,
                    tool_aliases={"_filter": "Filtering contacts"},
                ),
            ],
        },
        {
            "label": "ContactManager.ask: _filter result",
            "delay": 1.5,
            "events": [
                tl(
                    cm_h,
                    _tool_result(
                        "tc_cm_filter",
                        "_filter",
                        {
                            "rows": [
                                {
                                    "name": "Rachel Torres",
                                    "role": "VP of Partnerships",
                                    "email": "r.torres@acmecorp.com",
                                    "company": "Acme Corp",
                                },
                                {
                                    "name": "David Kim",
                                    "role": "Account Manager",
                                    "email": "d.kim@acmecorp.com",
                                    "company": "Acme Corp",
                                },
                            ],
                            "total": 2,
                        },
                    ),
                    method=cm_method,
                ),
            ],
        },
        {
            "label": "ContactManager.ask: LLM thinking (in flight)",
            "delay": 0.5,
            "events": [
                tl(
                    cm_h,
                    {"role": "assistant", "_thinking_in_flight": True},
                    method=cm_method,
                ),
            ],
        },
        {
            "label": "ContactManager.ask: final response",
            "delay": 2.5,
            "events": [
                tl(
                    cm_h,
                    {
                        "role": "assistant",
                        "content": (
                            "Found 2 contacts at Acme Corp. The main contact for partnerships is "
                            "Rachel Torres (VP of Partnerships, r.torres@acmecorp.com). "
                            "There's also David Kim (Account Manager, d.kim@acmecorp.com)."
                        ),
                    },
                    method=cm_method,
                ),
            ],
        },
        # ── 14. ContactManager.ask outgoing ──
        {
            "label": "ContactManager.ask — ManagerMethod outgoing",
            "delay": 0.3,
            "events": [
                mm(
                    cm_cid,
                    cm_h,
                    phase="outgoing",
                    manager="ContactManager",
                    method="ask",
                    display_label="Checking Contact Book",
                    answer=(
                        "Rachel Torres — VP of Partnerships at Acme Corp (r.torres@acmecorp.com). "
                        "Also David Kim — Account Manager (d.kim@acmecorp.com)."
                    ),
                ),
            ],
        },
        # ── 15. Sub-agent's contacts.ask boundary outgoing ──
        {
            "label": "Sub-agent: contacts.ask — boundary outgoing",
            "delay": 0.3,
            "events": [
                mm(
                    ef2_cid,
                    ef2_h,
                    phase="outgoing",
                    manager="CodeActActor",
                    method="execute_function",
                    display_label="Running: primitives.contacts.ask",
                    answer="Found Rachel Torres (VP of Partnerships) and David Kim (Account Manager) at Acme Corp.",
                ),
            ],
        },
        # ── 16. Sub-agent: tool result ──
        {
            "label": "Sub-agent: contacts.ask tool result",
            "delay": 0.5,
            "events": [
                tl(
                    sa_h,
                    _tool_result(
                        "tc_sa_contacts",
                        "execute_function",
                        {
                            "answer": (
                                "Rachel Torres — VP of Partnerships at Acme Corp (r.torres@acmecorp.com). "
                                "Also David Kim — Account Manager (d.kim@acmecorp.com)."
                            ),
                        },
                    ),
                ),
            ],
        },
        # ── 17. Sub-agent: final response ──
        {
            "label": "Sub-agent: LLM thinking (in flight)",
            "delay": 0.5,
            "events": [
                tl(sa_h, {"role": "assistant", "_thinking_in_flight": True}),
            ],
        },
        {
            "label": "Sub-agent: final response",
            "delay": 3.0,
            "events": [
                tl(
                    sa_h,
                    {
                        "role": "assistant",
                        "content": (
                            "The main contact at Acme Corp is **Rachel Torres**, VP of Partnerships.\n\n"
                            "- **Name:** Rachel Torres\n"
                            "- **Role:** VP of Partnerships\n"
                            "- **Email:** r.torres@acmecorp.com\n\n"
                            "There's also David Kim (Account Manager, d.kim@acmecorp.com) "
                            "as a secondary contact."
                        ),
                    },
                ),
            ],
        },
        # ── 18. Sub-agent CodeActActor.act outgoing ──
        {
            "label": "Sub-agent CodeActActor.act — ManagerMethod outgoing",
            "delay": 0.3,
            "events": [
                mm(
                    sa_cid,
                    sa_h,
                    phase="outgoing",
                    display_label="Taking Action",
                    answer=(
                        "Main contact: Rachel Torres, VP of Partnerships at Acme Corp "
                        "(r.torres@acmecorp.com). Secondary: David Kim, Account Manager "
                        "(d.kim@acmecorp.com)."
                    ),
                ),
            ],
        },
        # ── 19. execute_function(primitives.actor.act) boundary outgoing ──
        {
            "label": "execute_function(primitives.actor.act) — boundary outgoing",
            "delay": 0.3,
            "events": [
                mm(
                    ef1_cid,
                    ef1_h,
                    phase="outgoing",
                    manager="CodeActActor",
                    method="execute_function",
                    display_label="Running: primitives.actor.act",
                    answer="Found contact details for Acme Corp.",
                ),
            ],
        },
        # ── 20. Parent: sub-agent tool result ──
        {
            "label": "Parent: sub-agent tool result",
            "delay": 0.5,
            "events": [
                tl(
                    h,
                    _tool_result(
                        "tc_subagent",
                        "execute_function",
                        {
                            "answer": (
                                "Main contact: Rachel Torres, VP of Partnerships at Acme Corp "
                                "(r.torres@acmecorp.com). Secondary: David Kim, Account Manager "
                                "(d.kim@acmecorp.com)."
                            ),
                        },
                    ),
                ),
            ],
        },
        # ── 21. Parent: draft email via execute_code ──
        {
            "label": "LLM thinking (in flight)",
            "delay": 0.5,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
        {
            "label": "Doing: execute_code — draft follow-up email",
            "delay": 4.0,
            "events": [
                tl(
                    h,
                    _thinking(
                        "The sub-agent found Rachel Torres as the main contact. Now I have the "
                        "email template guidance and the contact info. Let me draft the follow-up "
                        "email about the Q1 partnership review and notify the user.",
                        tool_calls=[
                            _tc(
                                "tc_code",
                                "execute_code",
                                {
                                    "code": (
                                        "email_draft = {\n"
                                        '    "to": "r.torres@acmecorp.com",\n'
                                        '    "subject": "Q1 Partnership Review Follow-Up",\n'
                                        '    "body": (\n'
                                        '        "Hi Rachel,\\n\\n"\n'
                                        '        "Thank you for your time during our Q1 partnership review. "\n'
                                        '        "I wanted to follow up on the key action items we discussed:\\n\\n"\n'
                                        '        "1. Integration timeline for the shared analytics dashboard\\n"\n'
                                        '        "2. Updated SLA terms for the enterprise tier\\n"\n'
                                        '        "3. Joint marketing campaign planning for Q2\\n\\n"\n'
                                        '        "Could we schedule a 30-minute call next week to finalize these items?\\n\\n"\n'
                                        '        "Best regards"\n'
                                        "    ),\n"
                                        "}\n"
                                        'notify({"type": "progress", "message": "Email draft ready for review."})\n'
                                        "email_draft"
                                    ),
                                    "language": "python",
                                },
                            ),
                        ],
                    ),
                ),
            ],
        },
        # ── 22. execute_code result ──
        {
            "label": "execute_code result",
            "delay": 2.0,
            "events": [
                tl(
                    h,
                    _tool_result(
                        "tc_code",
                        "execute_code",
                        {
                            "result": {
                                "to": "r.torres@acmecorp.com",
                                "subject": "Q1 Partnership Review Follow-Up",
                                "body": "Hi Rachel,\n\nThank you for your time...",
                            },
                            "notification_sent": True,
                        },
                    ),
                ),
            ],
        },
        # ── 23. Final response ──
        {
            "label": "LLM thinking (in flight)",
            "delay": 0.5,
            "events": [tl(h, {"role": "assistant", "_thinking_in_flight": True})],
        },
        {
            "label": "Response: email draft ready",
            "delay": 4.0,
            "events": [
                tl(
                    h,
                    {
                        "role": "assistant",
                        "content": (
                            "Here's the draft follow-up email for Rachel Torres at Acme Corp:\n\n"
                            "**To:** r.torres@acmecorp.com\n"
                            "**Subject:** Q1 Partnership Review Follow-Up\n\n"
                            "---\n\n"
                            "Hi Rachel,\n\n"
                            "Thank you for your time during our Q1 partnership review. "
                            "I wanted to follow up on the key action items we discussed:\n\n"
                            "1. Integration timeline for the shared analytics dashboard\n"
                            "2. Updated SLA terms for the enterprise tier\n"
                            "3. Joint marketing campaign planning for Q2\n\n"
                            "Could we schedule a 30-minute call next week to finalize these items?\n\n"
                            "Best regards"
                        ),
                    },
                ),
            ],
        },
        # ── 24–30. StorageCheck: reviews trajectory, decides nothing to store ──
        {
            "label": "StorageCheck incoming",
            "delay": 1.0,
            "events": [
                mm(
                    sc_cid,
                    sc_h,
                    phase="incoming",
                    manager="CodeActActor",
                    method="StorageCheck",
                    display_label="Storing Reusable Skills",
                    request="Review the trajectory and store any reusable functions and compositional guidance.",
                ),
            ],
        },
        {
            "label": "SC inner: user request",
            "delay": 0.3,
            "events": [
                tl(
                    sc_h,
                    {
                        "role": "user",
                        "content": "Review the trajectory and store any reusable functions and compositional guidance.",
                    },
                    method=sc_method,
                ),
            ],
        },
        {
            "label": "SC inner: thinking + search existing functions",
            "delay": 1.5,
            "events": [
                tl(
                    sc_h,
                    _thinking(
                        "Let me check what functions and guidance already exist before "
                        "deciding whether anything from this trajectory is worth storing.",
                        tool_calls=[
                            _tc(
                                "tc_sc_list",
                                "FunctionManager_list_functions",
                                {"include_implementations": False},
                            ),
                        ],
                    ),
                    method=sc_method,
                    tool_aliases={
                        "FunctionManager_list_functions": "Listing existing functions",
                    },
                ),
            ],
        },
        {
            "label": "SC inner: list_functions result",
            "delay": 1.5,
            "events": [
                tl(
                    sc_h,
                    _tool_result(
                        "tc_sc_list",
                        "FunctionManager_list_functions",
                        {"functions": [], "count": 0},
                    ),
                    method=sc_method,
                ),
            ],
        },
        {
            "label": "SC inner: thinking + search guidance",
            "delay": 1.0,
            "events": [
                tl(
                    sc_h,
                    _thinking(
                        "No existing functions. The trajectory involved looking up a contact "
                        "and drafting an email — both are straightforward single-primitive calls "
                        "with no reusable composition pattern. Let me check guidance too.",
                        tool_calls=[
                            _tc(
                                "tc_sc_search_g",
                                "GuidanceManager_search",
                                {"query": "contact lookup email drafting"},
                            ),
                        ],
                    ),
                    method=sc_method,
                    tool_aliases={
                        "GuidanceManager_search": "Searching existing guidance",
                    },
                ),
            ],
        },
        {
            "label": "SC inner: search guidance result",
            "delay": 1.0,
            "events": [
                tl(
                    sc_h,
                    _tool_result(
                        "tc_sc_search_g",
                        "GuidanceManager_search",
                        {"results": [], "count": 0},
                    ),
                    method=sc_method,
                ),
            ],
        },
        {
            "label": "SC inner: final response (nothing to store)",
            "delay": 2.0,
            "events": [
                tl(
                    sc_h,
                    {
                        "role": "assistant",
                        "content": (
                            "Nothing worth storing. The trajectory used two standard primitives "
                            "(primitives.contacts.ask + execute_code for email drafting) in a "
                            "straightforward sequence with no reusable composition pattern or "
                            "non-obvious logic worth persisting as a function or guidance entry."
                        ),
                    },
                    method=sc_method,
                ),
            ],
        },
        {
            "label": "StorageCheck outgoing",
            "delay": 0.5,
            "events": [
                mm(
                    sc_cid,
                    sc_h,
                    phase="outgoing",
                    manager="CodeActActor",
                    method="StorageCheck",
                    display_label="Storing Reusable Skills",
                    answer="Nothing worth storing.",
                ),
            ],
        },
        # ── 31. ManagerMethod outgoing ──
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
                        "Drafted follow-up email to Rachel Torres (VP of Partnerships, Acme Corp) "
                        "regarding the Q1 partnership review action items."
                    ),
                ),
            ],
        },
    ]

    return steps


SCENARIOS = {
    "persistent": build_persistent_steps,
    "single_action": build_single_action_steps,
}


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


def run_stream(
    steps_builder: callable,
    scenario_name: str,
    speed: float,
    persist: bool = False,
) -> None:
    steps = steps_builder()
    total_time = sum(s["delay"] for s in steps) / speed
    mode_label = "stream + save" if persist else "stream"
    print(
        f"\n=== {scenario_name} / {mode_label} (speed={speed}x, ~{total_time:.0f}s total) ===\n",
    )

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


def run_upload(steps_builder: callable, scenario_name: str) -> None:
    print("\n--- Setup ---")
    setup_orchestra()

    steps = steps_builder()
    print(f"\n=== Uploading {len(steps)} steps ({scenario_name}) to Orchestra ===\n")

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
        "--scenario",
        default="persistent",
        choices=list(SCENARIOS.keys()),
        help="Which scenario to simulate (default: persistent).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Push events via SSE to the Console with realistic delays.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Write events to Orchestra for historical access / page refresh.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Delay multiplier for --stream (default: 1.0). "
        "Use 2 for 2x faster, 0.5 for slower.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Wipe existing action events from Orchestra before starting.",
    )
    args = parser.parse_args()

    if not args.stream and not args.save:
        args.stream = True

    if args.clear:
        import subprocess

        clear_script = Path(__file__).parent / "clear_action_events.sh"
        subprocess.run(["bash", str(clear_script)], check=True)
        input("\nRefresh the browser, then press Enter to continue...")

    steps_builder = SCENARIOS[args.scenario]

    if args.stream:
        run_stream(steps_builder, args.scenario, args.speed, persist=args.save)
    else:
        run_upload(steps_builder, args.scenario)


if __name__ == "__main__":
    main()
