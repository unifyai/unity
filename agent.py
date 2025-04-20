from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any

import unify
from helpers import _pascal, _slug
from pydantic import BaseModel, create_model, Field
from actions import ActionHistory, BrowserState
from sys_msgs import PRIMITIVE_TO_BROWSER_ACTION_CANDIDATES, PRIMITIVE_TO_BROWSER_ACTION

client = unify.Unify(traced=True)
client.set_system_message(PRIMITIVE_TO_BROWSER_ACTION_CANDIDATES)

SCROLLING_STATE = None


# Schemas #

_response_fields = {
    "rationale": (
        Optional[str],
        Field(
            None,
            description="Explanation for your decision whether or not to apply this action.",
        ),
    ),
    "apply": (
        bool,
        Field(
            ...,
            description="Decision to apply this action or not.",
        ),
    ),
}


class NewTab(BaseModel):
    """
    Open a new tab.
    """

    rationale: Optional[str] = Field(
        ...,
        description="Explanation for your decision whether or not to apply this action.",
    )
    apply: bool = Field(..., description="Decision to apply this action or not.")


class ScrollUp(BaseModel):
    """
    Scroll up by a certain number of pixels.
    """

    pixels: Optional[int] = Field(
        ...,
        description="Number of pixels to scroll up, if action is applied.",
    )
    rationale: Optional[str] = Field(
        ...,
        description="Explanation for your decision whether or not to apply this action.",
    )
    apply: bool = Field(..., description="Decision to apply this action or not.")


class ScrollDown(BaseModel):
    """
    Scroll down by a certain number of pixels.
    """

    pixels: Optional[int] = Field(
        ...,
        description="Number of pixels to scroll down, if action is applied.",
    )
    rationale: Optional[str] = Field(
        ...,
        description="Explanation for your decision whether or not to apply this action.",
    )
    apply: bool = Field(..., description="Decision to apply this action or not.")


class StartScrollingUp(BaseModel):
    """
    Start gently scrolling upwards, until another stop action is given.
    """

    rationale: Optional[str] = Field(
        ...,
        description="Explanation for your decision whether or not to apply this action.",
    )
    apply: bool = Field(..., description="Decision to apply this action or not.")


class StartScrollingDown(BaseModel):
    """
    Start gently scrolling downwards, until another stop action is given.
    """

    rationale: Optional[str] = Field(
        ...,
        description="Explanation for your decision whether or not to apply this action.",
    )
    apply: bool = Field(..., description="Decision to apply this action or not.")


class StopScrollingUp(BaseModel):
    """
    Stop the upwards scroll motion which is currently occuring.
    """

    rationale: Optional[str] = Field(
        ...,
        description="Explanation for your decision whether or not to apply this action.",
    )
    apply: bool = Field(..., description="Decision to apply this action or not.")


class StopScrollingDown(BaseModel):
    """
    Stop the downwards scroll motion which is currently occuring.
    """

    rationale: Optional[str] = Field(
        ...,
        description="Explanation for your decision whether or not to apply this action.",
    )
    apply: bool = Field(..., description="Decision to apply this action or not.")


class Search(BaseModel):
    """
    Search the web for a the specified query in the topmost search bar of the browser.
    """

    query: str = Field(
        ...,
        description="The search query to type into the search bar, at the top of the browser.",
    )
    rationale: Optional[str] = Field(
        ...,
        description="Explanation for your decision whether or not to apply this action.",
    )
    apply: bool = Field(..., description="Decision to apply this action or not.")


class SearchURL(BaseModel):
    """
    Navigate the browser to a specific URL.
    """

    url: str = Field(..., description="The absolute or bare URL to open.")
    rationale: Optional[str] = Field(
        None,
        description="Explanation for your decision whether or not to apply this action.",
    )
    apply: bool = Field(..., description="Decision to apply this action or not.")


def _construct_tab_actions(tabs: List[str], mode: str):
    if not tabs:
        return {}

    field_prefix = f"{mode.lower()}_tab_"
    model_prefix = f"{mode.capitalize()}Tab"

    return {
        f"{field_prefix}{_slug(title)}": create_model(
            f"{model_prefix}{_pascal(_slug(title))}",
            __doc__=f"{mode} the “{title}” tab.",
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
            __doc__=f"Click the “{raw_text}” button.",
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


def _create_full_response_format(
    tabs: List[str],
    buttons: Optional[List[Tuple[int, str]]] = None,
):
    return create_model(
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
        search_action=(Search, ...),
        search_url_action=(SearchURL, ...),
    )


def _extract_applied_actions(response: BaseModel) -> Tuple[Dict[str, Any], int]:
    applied: Dict[str, Any] = {}
    kept_count = 0

    # ---- grouped (nested) categories --------------------------------------
    for group in ("tab_actions", "scroll_actions", "button_actions"):
        if not hasattr(response, group):
            continue

        subgroup_instance = getattr(response, group)
        kept: Dict[str, BaseModel] = {}

        for field in subgroup_instance.model_fields:
            leaf = getattr(subgroup_instance, field)
            if leaf and getattr(leaf, "apply", False):
                kept[field] = leaf.model_dump()
                kept_count += 1

        if kept:
            applied[group] = kept

    # ---- stand‑alone search action ----------------------------------------
    if hasattr(response, "search_action"):
        sa = getattr(response, "search_action")
        if sa and getattr(sa, "apply", False):
            applied["search_action"] = sa.model_dump()
            kept_count += 1

    if hasattr(response, "search_url_action"):
        sua = getattr(response, "search_url_action")
        if sua and getattr(sua, "apply", False):
            applied["search_url_action"] = sua.model_dump()
            kept_count += 1

    return applied, kept_count


def _get_action_class(action_name: str) -> type[BaseModel]:
    """
    Return a ``pydantic.BaseModel`` subclass whose docstring and field
    descriptions match those used in the *original* full response‑format.

    This works for both the fixed actions (e.g. ``scroll_up``) and the
    dynamically generated tab / button actions.
    """
    # ---- fixed actions ----------------------------------------------------
    fixed = {
        "new_tab": NewTab,
        "scroll_up": ScrollUp,
        "scroll_down": ScrollDown,
        "start_scrolling_up": StartScrollingUp,
        "start_scrolling_down": StartScrollingDown,
        "stop_scrolling_up": StopScrollingUp,
        "stop_scrolling_down": StopScrollingDown,
    }
    if action_name in fixed:
        return fixed[action_name]

    # ---- dynamic tab actions ---------------------------------------------
    if action_name.startswith("select_tab_"):
        slug = action_name[len("select_tab_") :]
        title = slug.replace("_", " ").replace("-", " ").title()
        return create_model(
            f"SelectTab{_pascal(slug)}",
            __doc__=f"Select the “{title}” tab.",
            **_response_fields,
        )

    if action_name.startswith("close_tab_"):
        slug = action_name[len("close_tab_") :]
        title = slug.replace("_", " ").replace("-", " ").title()
        return create_model(
            f"CloseTab{_pascal(slug)}",
            __doc__=f"Close the “{title}” tab.",
            **_response_fields,
        )

    # ---- dynamic button actions ------------------------------------------
    if action_name.startswith("click_button_"):
        slug = action_name[len("click_button_") :]
        text = slug.replace("_", " ").replace("-", " ").title()
        return create_model(
            f"ClickButton{_pascal(slug)}",
            __doc__=f"Click the “{text}” button.",
            **_response_fields,
        )

    raise ValueError(f"Unknown action field: {action_name!r}")


def _build_pruned_response_format(applied: Dict[str, Any]) -> BaseModel:
    """
    Construct a *pruned* response‑format model that preserves every original
    docstring and field description.

    ``applied`` is the mapping returned by ``_extract_applied_actions`` and
    therefore contains **JSON‑serialisable dicts** at the leaves.
    """
    top_level: Dict[str, tuple[type, ...]] = {}

    # ---- nested groups (tab / scroll / button) ---------------------------
    for group, sub in applied.items():
        if group == "search_action":
            continue  # handled separately

        # Rebuild each kept leaf with the correct BaseModel subclass
        fields = {
            name: (_get_action_class(name), ...)
            for name in sub.keys()  # sub values are plain dicts
        }
        SubModel = create_model(f"{_pascal(group)}", **fields)
        top_level[group] = (SubModel, ...)

    # ---- single search action -------------------------------------------
    if "search_action" in applied:
        top_level["search_action"] = (Search, ...)

    if "search_url_action" in applied:
        top_level["search_url_action"] = (SearchURL, ...)

    if not top_level:
        raise ValueError(
            "Cannot build a pruned response‑format — no actions had apply=True.",
        )

    # “ActionSelection” is the same top‑level model name used originally
    return create_model("ActionSelection", **top_level)


# === helper to expose available actions to the GUI =======================
def list_available_actions(  # NEW
    tabs: List[str],
    buttons: Optional[List[Tuple[int, str]]] | None = None,
) -> dict[str, list[str]]:
    """
    Return a mapping {group_name: [field_names,…]} describing every action
    that would appear in the full response‑format schema given the current
    set of browser tabs and visible buttons.

    Groups:
        • "tab_actions"
        • "scroll_actions"
        • "button_actions"
        • "standalone"   (search_action, search_url_action)
    """
    fmt = _create_full_response_format(tabs, buttons)  # reuse existing logic
    return {
        "tab_actions": list(
            fmt.model_fields["tab_actions"].annotation.model_fields,
        ),
        "scroll_actions": list(
            fmt.model_fields["scroll_actions"].annotation.model_fields,
        ),
        "button_actions": list(
            fmt.model_fields["button_actions"].annotation.model_fields,
        ),
        "standalone": ["search_action", "search_url_action"],
    }


@unify.traced
def primitive_to_browser_action(
    text: str,
    screenshot: bytes,
    *,
    tabs: Optional[List[str]],
    buttons: Optional[List[Tuple[int, str]]] = None,
    history: ActionHistory = None,
    state: BrowserState = None,
) -> Optional[BaseModel]:

    response_format = _create_full_response_format(tabs, buttons)
    client.set_endpoint("gpt-4o-mini@openai")

    history_msg = (
        "\n\nThe low-level action history (most recent first) is as follows:\n"
        + "\n".join(f"{r['timestamp']:.0f}: {r['command']}" for r in history[-20:])
    )

    state_msg = f"""\n\nThe current state of the browser is as follows:
    url: {state.get('url')}
    title: {state.get('title')}
    scroll_y: {state.get('scroll_y')}
    auto_scroll: {state.get('auto_scroll')}
    in_textbox: {state.get('in_textbox')}
    """

    client.set_system_message(
        PRIMITIVE_TO_BROWSER_ACTION_CANDIDATES + history_msg + state_msg,
    )
    client.set_response_format(response_format)
    ret = client.generate(text)
    ret = response_format.model_validate_json(ret)
    ret, num_selected = _extract_applied_actions(ret)
    if num_selected == 1:
        # only one candidate, can already return
        response_format = _build_pruned_response_format(ret)
        return response_format.model_validate(ret).model_dump()

    # decide among the candidate actions
    client.set_endpoint("o3-mini@openai")
    client.set_system_message(
        PRIMITIVE_TO_BROWSER_ACTION + history_msg + state_msg,
    )
    while num_selected > 1:
        response_format = _build_pruned_response_format(ret)
        client.set_response_format(response_format)
        ret = client.generate(text)
        ret = response_format.model_validate_json(ret)
        ret, num_selected = _extract_applied_actions(ret)
    response_format = _build_pruned_response_format(ret)
    return response_format.model_validate(ret).model_dump()
