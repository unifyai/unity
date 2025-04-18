from __future__ import annotations

from typing import List, Optional, Tuple

import unify
from helpers import _pascal, _slug
from pydantic import BaseModel, create_model
from sys_msgs import INTERJECTION_TO_BROWSER_ACTION

client = unify.Unify("gpt-4o-mini@openai")
client.set_system_message(INTERJECTION_TO_BROWSER_ACTION)

SCROLLING_STATE = None


# Schemas #

_response_fields = {
    "rationale": (Optional[str], ...),
    "apply": (bool, ...),
}


class NewTab(BaseModel):
    rationale: Optional[str]
    apply: bool


class ScrollUp(BaseModel):
    rationale: Optional[str]
    apply: bool
    pixels: Optional[int]


class ScrollDown(BaseModel):
    rationale: Optional[str]
    apply: bool
    pixels: Optional[int]


class StartScrollingUp(BaseModel):
    rationale: Optional[str]
    apply: bool


class StartScrollingDown(BaseModel):
    rationale: Optional[str]
    apply: bool


class StopScrollingUp(BaseModel):
    rationale: Optional[str]
    apply: bool


class StopScrollingDown(BaseModel):
    rationale: Optional[str]
    apply: bool


def _construct_tab_actions(tabs: List[str], mode: str):
    if not tabs:
        return {}

    field_prefix = f"{mode.lower()}_tab_"
    model_prefix = f"{mode.capitalize()}Tab"

    return {
        f"{field_prefix}{_slug(title)}": create_model(
            f"{model_prefix}{_pascal(_slug(title))}",
            **_response_fields,
        )
        for title in tabs
    }


def _construct_close_tab_actions(tabs: List[str]):
    return _construct_tab_actions(tabs, "Close")


def _construct_select_tab_actions(tabs: List[str]):
    return _construct_tab_actions(tabs, "Select")


def _construct_select_button_actions(
    buttons: Optional[List[Tuple[int, str]]] = None,
):
    if not buttons:
        return {}

    actions = {}
    for _, raw_text in buttons:
        slug = _slug(raw_text)
        pascal = _pascal(slug)

        actions[f"click_button_{slug}"] = create_model(
            f"ClickButton{pascal}",
            **_response_fields,
        )

    return actions


def _construct_scroll_actions():
    if SCROLLING_STATE is None:
        return {
            "scroll_up": ScrollUp,
            "scroll_down": ScrollDown,
            "start_scrolling_up": StartScrollingUp,
            "start_scrolling_down": StartScrollingDown,
        }
    elif SCROLLING_STATE == "up":
        return {
            "stop_scrolling_up": StopScrollingUp,
            "start_scrolling_down": StartScrollingDown,
        }
    elif SCROLLING_STATE == "down":
        return {
            "stop_scrolling_down": StopScrollingDown,
            "start_scrolling_up": StartScrollingUp,
        }
    else:
        raise Exception(f"Invalid SCROLLING_STATE {SCROLLING_STATE}")


def parse_instruction(
    text: str,
    tabs: List[str],
    screenshot: bytes,
    buttons: Optional[List[Tuple[int, str]]] = None,
) -> Optional[BaseModel]:
    response_format = create_model(
        "ActionSelection",
        tab_actions=create_model(
            "TabActions",
            new_tab=(NewTab, ...),
            **_construct_select_tab_actions(tabs),
            **_construct_close_tab_actions(tabs),
        ),
        scroll_actions=create_model(
            "ScrollActions",
            **_construct_scroll_actions(),
        ),
        button_actions=create_model(
            "ButtonActions",
            **_construct_select_button_actions(buttons),
        ),
    )
    client.set_response_format(response_format)
    ret = client.generate(text)
    ret = response_format.model_validate_json(ret)
    return ret
