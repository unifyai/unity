from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from typing import Dict, Callable

from .types.task import Task

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {name: '(<argspec>)', …} for prettier JSON dumps."""
    return {n: str(inspect.signature(fn)) for n, fn in tools.items()}


def _now() -> str:
    """Current UTC timestamp in a compact, human-readable form."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ─────────────────────────────────────────────────────────────────────────────
# Public builders
# ─────────────────────────────────────────────────────────────────────────────


def build_ask_prompt(tools: Dict[str, Callable]) -> str:
    """
    Build the **system** prompt for the `ask` method.

    *Never* hard-codes the number, names or argument-specs of tools – those are
    injected live from the supplied *tools* dict.
    """
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    usage_examples = "[Add usage examples here]"  # placeholder

    return "\n".join(
        [
            "You are an assistant specialising in **answering questions about the task list**.",
            "Interact with the read-only tools provided (see below) to gather whatever",
            "information you need, *step-by-step*.  When you have everything, respond",
            "with a concise, final answer.",
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            usage_examples,
            "",
            "Task schema:",
            json.dumps(Task.model_json_schema(), indent=4),
            "",
            f"Current UTC time is {_now()}.",
        ],
    )


def build_update_prompt(tools: Dict[str, Callable]) -> str:
    """
    Build the **system** prompt for the `update` method.
    """
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    usage_examples = "[Add usage examples here]"  # placeholder

    return "\n".join(
        [
            "You are an assistant responsible for **creating and updating tasks**.",
            "Use the tools supplied *only* – never invent your own – until the task",
            "list fully reflects the user's intent.",
            "If a any tasks were created or updated in the process,"
            "then please *always* include these task id(s) in your final response.",
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            usage_examples,
            "",
            "Task schema:",
            json.dumps(Task.model_json_schema(), indent=4),
            "",
            f"Current UTC time is {_now()}.",
        ],
    )
