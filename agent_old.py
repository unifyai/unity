"""
Agent — plain English ➜ structured Action   (OpenAI‑Python ≥ 1.0.0)
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------- schema
class ActionName(str, Enum):
    click_button = "click_button"  # NEW
    scroll = "scroll"
    start_scroll = "start_scroll"
    stop_scroll = "stop_scroll"
    new_tab = "new_tab"
    close_tab = "close_tab"
    switch_tab = "switch_tab"


class Action(BaseModel):
    action: ActionName
    # click_button
    button_text: Optional[str] = None
    # scrolling
    direction: Optional[Literal["up", "down"]] = None
    pixels: Optional[int] = Field(None, ge=1)
    # tab ops
    tab_text: Optional[str] = None


# ---------------------------------------------------------------------- prompt
_SYSTEM_PROMPT = """\
You convert plain‑English requests into JSON commands.

Allowed *actions*:

• click_button   button_text(str) — click first visible element whose text/label
                 contains this substring, case‑insensitive.
• scroll         direction(up|down) pixels(int>0)
• start_scroll   direction(up|down)
• stop_scroll    –
• new_tab        –
• close_tab      tab_text(optional) – substring of tab title; if omitted close active.
• switch_tab     tab_text(str)

Return ONLY valid JSON, no markdown, exactly matching the schema.
"""

_TOOL_ID = "action_schema"
_TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": _TOOL_ID,
            "description": "Structured browser action",
            "parameters": Action.schema(),  # <- pydantic → JSON schema
        },
    },
]

# single, reusable client
_client = OpenAI()  # uses OPENAI_API_KEY from env / config


# ---------------------------------------------------------------- parse helper
def parse_instruction(text: str, *, debug: bool = False) -> Action | None:
    """
    Convert free‑text `text` into an `Action` or return None on failure.
    If debug=True, exceptions are re‑raised for the caller to handle.
    """
    resp = _client.chat.completions.create(
        model="o3-mini",
        tools=_TOOL_SPEC,
        tool_choice={"type": "function", "function": {"name": _TOOL_ID}},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    args_json = resp.choices[0].message.tool_calls[0].function.arguments
    payload = json.loads(args_json)
    return Action.model_validate(payload)


# ----------------------------------------------------------------
