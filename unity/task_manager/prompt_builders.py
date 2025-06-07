from __future__ import annotations

import inspect
import json
import textwrap
from datetime import datetime, timezone
from typing import Dict, Callable

from ..task_scheduler.types.task import Task
from ..common.llm_helpers import SteerableToolHandle, class_api_overview

# ───────────────────────────────────── helpers ─────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return a *compact* mapping of {tool-name: '(<argspec>)'}."""
    return {n: str(inspect.signature(fn)) for n, fn in tools.items()}


def _now() -> str:
    """Current UTC timestamp in a friendly format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ───────────────────────────────────── builders ─────────────────────────────────────


def build_ask_prompt(tools: Dict[str, Callable]) -> str:
    """Dynamic **system** prompt for `TaskManager.ask`."""
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    usage_examples = "[Add usage examples here]"  # placeholder

    return textwrap.dedent(
        f"""
        You are an assistant specialising in **read-only questions** about tasks,
        contacts, transcripts and the knowledge-base.  Interact with the tools
        below *step-by-step* until you can answer concisely.

        Tools (name → argspec):
        {sig_json}

        {usage_examples}

        Task schema (for filters):
        {json.dumps(Task.model_json_schema(), indent=4)}

        SteerableToolHandle class:
        {class_api_overview(SteerableToolHandle)}

        Current UTC time is {_now()}.
        """,
    ).strip()


def build_request_prompt(tools: Dict[str, Callable]) -> str:
    """Dynamic **system** prompt for `TaskManager.request`."""
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    usage_examples = "[Add usage examples here]"  # placeholder

    return textwrap.dedent(
        f"""
        You have **full read-write control** over tasks, contacts, transcripts
        and the knowledge-base. Use *only* the tools supplied – never invent
        your own. Call them iteratively until the user's request is completely
        fulfilled, verifying state after each mutation.

        Tools (name → argspec):
        {sig_json}

        {usage_examples}

        Task schema:
        {json.dumps(Task.model_json_schema(), indent=4)}

        SteerableToolHandle class:
        {class_api_overview(SteerableToolHandle)}

        Current UTC time is {_now()}.
        """,
    ).strip()


def build_start_task_prompt(tools: Dict[str, Callable]) -> str:
    """Dynamic **system** prompt for `TaskManager.start_task`."""
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    usage_examples = "[Add usage examples here]"  # placeholder

    return textwrap.dedent(
        f"""
        Your job is to **activate exactly one task** so that it becomes the
        single *active* task in the system.

        Tools (name → argspec):
        {sig_json}

        {usage_examples}

        Activation rules
        • Only one task may be active at any time.
        • Do **not** change other task properties here.
        • After success, confirm activation in natural language.
        • Ask for clarification if the user is ambiguous.

        Task schema:
        {json.dumps(Task.model_json_schema(), indent=4)}

        SteerableToolHandle class:
        {class_api_overview(SteerableToolHandle)}

        Current UTC time is {_now()}.
        """,
    ).strip()
