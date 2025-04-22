from __future__ import annotations

from typing import List, Optional, Union, Tuple, Dict, Any

import unify
import base64
from helpers import _pascal, _slug
from constants import *
from pydantic import BaseModel, create_model, Field
from actions import ActionHistory, BrowserState
from action_filter import get_valid_actions
from sys_msgs import PRIMITIVE_TO_BROWSER_ACTION_CANDIDATES, PRIMITIVE_TO_BROWSER_ACTION

client = unify.Unify(traced=True)
client.set_system_message(PRIMITIVE_TO_BROWSER_ACTION_CANDIDATES)

SCROLLING_STATE = None
ADVANCED_MODE = False

# helpers #


def _list_flat_actions(tabs, buttons, state) -> list[str]:
    """
    Return the flat list of valid primitive strings for the current state.
    Uses the same logic as the advanced response‑format builder.
    """
    valid_schemas = get_valid_actions(state, mode="schema")
    valid_actions = get_valid_actions(state, mode="actions")

    flat = sorted(valid_schemas)

    # ---- dynamic tab placeholders --------------------------------------
    if CMD_SELECT_TAB in valid_actions:
        flat.extend(CMD_SELECT_TAB.replace(" *", f"_{_slug(title)}") for title in tabs)
        flat.remove("select_tab_*")

    if CMD_CLOSE_TAB in valid_actions:
        flat.extend(CMD_CLOSE_TAB.replace(" *", f"_{_slug(title)}") for title in tabs)
        flat.remove("close_tab_*")

    # ---- dynamic button placeholders -----------------------------------
    if buttons and CMD_CLICK_BUTTON in valid_actions:
        flat.extend(
            CMD_CLICK_BUTTON.replace(" *", f"_{idx}_{_slug(lbl)}")
            for idx, lbl in buttons
        )
        flat.remove("click_button_*")
    return sorted(set([item.replace("*", "").replace(" ", "") for item in flat]))


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


class StopScrolling(BaseModel):
    """Stop whichever auto‑scroll is currently active (direction‑agnostic)."""

    rationale: Optional[str] = Field(
        ...,
        description="Why you do / don't want to stop scrolling.",
    )
    apply: bool = Field(..., description="Set to true to stop scrolling.")


class ContinueScrolling(BaseModel):
    """Let the current auto‑scroll motion keep running (no‑op)."""

    rationale: Optional[str] = Field(
        ...,
        description="Why you do / don't want to continue scrolling.",
    )
    apply: bool = Field(..., description="Set to true to keep scrolling.")


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


class EnterText(BaseModel):
    """Type the provided text at the current caret position."""

    text: str = Field(..., description="Text to type (may include \\n, \\t, …)")
    rationale: Optional[str] = Field(
        None,
        description="Why you do / don't want to type this text.",
    )
    apply: bool = Field(..., description="Type the text if true.")


class SimpleKeyAction(BaseModel):
    """A single key‑press or modifier action."""

    rationale: Optional[str] = Field(
        None,
        description="Reason for pressing (or not pressing) the key.",
    )
    apply: bool = Field(..., description="Press the key if true.")


_SIMPLE_KEY_ACTIONS = {
    CMD_PRESS_ENTER: "Press the Enter/Return key.",
    CMD_PRESS_BACKSPACE: "Press Backspace (delete character to the left).",
    CMD_PRESS_DELETE: "Press Delete (character to the right).",
    CMD_CURSOR_LEFT: "Move caret one character to the left.",
    CMD_CURSOR_RIGHT: "Move caret one character to the right.",
    CMD_CURSOR_UP: "Move caret up one line.",
    CMD_CURSOR_DOWN: "Move caret down one line.",
    CMD_SELECT_ALL: "Select the entire text.",
    CMD_MOVE_LINE_START: "Move caret to the start of the line.",
    CMD_MOVE_LINE_END: "Move caret to the end of the line.",
    CMD_MOVE_WORD_LEFT: "Move caret one word to the left.",
    CMD_MOVE_WORD_RIGHT: "Move caret one word to the right.",
    CMD_HOLD_SHIFT: "Hold the Shift key down.",
    CMD_RELEASE_SHIFT: "Release the Shift key.",
    CMD_CLICK_OUT: "Click outside the text‑box to blur focus.",
}


def _construct_textbox_actions() -> dict[str, type[BaseModel]]:
    """
    Build {field_name: PydanticModel} for every text‑box‑only primitive.
    """
    actions: dict[str, type[BaseModel]] = {}

    # enter_text *  (wildcard – needs its own model)
    actions[CMD_ENTER_TEXT.replace("*", "").rstrip()] = EnterText

    # simple key / caret actions
    for cmd, doc in _SIMPLE_KEY_ACTIONS.items():
        model_name = _pascal(cmd)
        actions[cmd] = create_model(
            model_name,
            __doc__=doc,
            __base__=SimpleKeyAction,
        )
    return actions


def _construct_tab_actions(tabs: List[str], mode: str):
    if not tabs:
        return {}

    field_prefix = f"{mode.lower()}_tab_"
    model_prefix = f"{mode.capitalize()}Tab"

    actions = {
        f"{field_prefix}{_slug(title)}": create_model(
            f"{model_prefix}{_pascal(_slug(title))}",
            __doc__=f"{mode} the “{title}” tab.",
            **_response_fields,
        )
        for title in tabs
    }
    return actions


def _construct_close_tab_actions(tabs: List[str]):
    return _construct_tab_actions(tabs, "Close")


def _construct_select_tab_actions(tabs: List[str]):
    return _construct_tab_actions(tabs, "Select")


class CloseActiveTab(BaseModel):
    """Close the currently active browser tab."""

    rationale: Optional[str] = Field(
        ...,
        description="Reason for closing / not closing the tab.",
    )
    apply: bool = Field(..., description="Close the active tab if true.")


def _construct_select_button_actions(
    buttons: Optional[List[Tuple[int, str]]] = None,
) -> dict[str, type[BaseModel]]:
    """
    Return a mapping {field_name: PydanticModel} for every visible button.

    Each *field_name* is now "click_button_<idx>_<slug_of_label>" so it carries
    the on‑screen number shown in the coloured overlay.
    """
    if not buttons:
        return {}

    actions: dict[str, type[BaseModel]] = {}

    for idx, raw_text in buttons:
        base_slug = _slug(raw_text)  # "sign_in"
        slug = f"{idx}_{base_slug}"  # "7_sign_in"
        pascal = _pascal(slug)  # "7SignIn"

        field_name = f"click_button_{slug}"
        model_name = f"ClickButton{pascal}"
        doc = f"Click the “{raw_text}” button (element #{idx})."

        actions[field_name] = create_model(
            model_name,
            __doc__=doc,
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
    else:  # already auto‑scrolling (either dir)
        return {
            "stop_scrolling": StopScrolling,
            "continue_scrolling": ContinueScrolling,
        }


class SimpleChoice(BaseModel):
    """Chosen action and your reasoning for it."""

    rationale: str = Field(..., description="Why you chose this action.")
    action: str = Field(
        ...,
        description="Exactly one action from the list you were given.",
    )
    value: Optional[Union[str, int]] = Field(
        ...,
        description="The *optional* str or int value associated with *some* actions.",
    )


def _create_full_response_format(tabs, buttons, state=None):
    # ensure we always work with a BrowserState object
    if state and not isinstance(state, BrowserState):
        state = BrowserState(**state)

    valid = get_valid_actions(state)

    def include(name):
        """
        Return True when *name* corresponds to one of the wildcard patterns
        in `valid`.  Accept three cases:

        1. exact match
        2. `v` ends with '*' and name starts with `v[:-1]`
        3. `v` ends with '*' and name equals `v[:-1].rstrip(" _")`
           (handles bare 'scroll_down' vs pattern 'scroll_down *')
        """
        for v in valid:
            if name == v:
                return True
            if v.endswith("*"):
                prefix = v[:-1]  # drop the '*'
                if name.startswith(prefix):
                    return True
                if name == prefix.rstrip(" _"):
                    return True
        return False

    tab_actions: dict[str, type[BaseModel]] = {}
    # only expose when allowed by action_filter
    if include("new_tab"):
        tab_actions["new_tab"] = NewTab
    if include("close_this_tab"):
        tab_actions["close_this_tab"] = CloseActiveTab
    tab_actions.update(
        {k: v for k, v in _construct_select_tab_actions(tabs).items() if include(k)},
    )
    tab_actions.update(
        {k: v for k, v in _construct_close_tab_actions(tabs).items() if include(k)},
    )

    button_actions = {
        k: v for k, v in _construct_select_button_actions(buttons).items() if include(k)
    }

    scroll_actions = {
        k: v for k, v in _construct_scroll_actions().items() if include(k)
    }

    # text‑box actions (only when we're actually in a text input)
    textbox_actions = {}
    if state and state.in_textbox:
        textbox_actions = {
            k: v for k, v in _construct_textbox_actions().items() if include(k)
        }

    fields = {
        "tab_actions": create_model("TabActions", **tab_actions),
        "scroll_actions": create_model("ScrollActions", **scroll_actions),
        "button_actions": create_model("ButtonActions", **button_actions),
        "textbox_actions": (
            create_model("TextboxActions", **textbox_actions)
            if textbox_actions
            else create_model("TextboxActions", __base__=BaseModel)
        ),
    }

    if include("search"):
        fields["search"] = (Search, ...)
    if include("open_url"):
        fields["open_url"] = (SearchURL, ...)

    return create_model("ActionSelection", **fields)


def _extract_applied_actions(response: BaseModel) -> Tuple[Dict[str, Any], int]:
    applied: Dict[str, Any] = {}
    kept_count = 0

    # ---- grouped (nested) categories --------------------------------------
    for group in (
        "tab_actions",
        "scroll_actions",
        "button_actions",
        "textbox_actions",
    ):
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
    if hasattr(response, "search"):
        sa = getattr(response, "search")
        if sa and getattr(sa, "apply", False):
            applied["search"] = sa.model_dump()
            kept_count += 1

    if hasattr(response, "open_url"):
        sua = getattr(response, "open_url")
        if sua and getattr(sua, "apply", False):
            applied["open_url"] = sua.model_dump()
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
        "stop_scrolling": StopScrolling,
        "continue_scrolling": ContinueScrolling,
        # ---------- simple key / caret actions -------------------------
        **{
            name: create_model(
                f"{_pascal(name)}",
                __doc__=_SIMPLE_KEY_ACTIONS[name],
                __base__=SimpleKeyAction,
            )
            for name in _SIMPLE_KEY_ACTIONS
        },
    }
    if action_name in fixed:
        return fixed[action_name]

    elif action_name == CMD_ENTER_TEXT.replace("*", "").rstrip():
        return EnterText

    # ---- dynamic tab actions ---------------------------------------------
    elif action_name.startswith("select_tab_"):
        slug = action_name[len("select_tab_") :]
        title = slug.replace("_", " ").replace("-", " ").title()
        return create_model(
            f"SelectTab{_pascal(slug)}",
            __doc__=f"Select the “{title}” tab.",
            **_response_fields,
        )

    elif action_name.startswith("close_tab_"):
        slug = action_name[len("close_tab_") :]
        title = slug.replace("_", " ").replace("-", " ").title()
        return create_model(
            f"CloseTab{_pascal(slug)}",
            __doc__=f"Close the “{title}” tab.",
            **_response_fields,
        )

    # ---- dynamic button actions ------------------------------------------
    elif action_name.startswith("click_button_"):
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

    # ---- nested groups (tab / scroll / button / textbox) -----------------
    for group, sub in applied.items():
        if group == "search":
            continue  # handled separately

        # Rebuild each kept leaf with the correct BaseModel subclass
        fields = {
            name: (_get_action_class(name), ...)
            for name in sub.keys()  # sub values are plain dicts
        }
        SubModel = create_model(f"{_pascal(group)}", **fields)
        top_level[group] = (SubModel, ...)

    # ---- single search action -------------------------------------------
    if "search" in applied:
        top_level["search"] = (Search, ...)

    if "open_url" in applied:
        top_level["open_url"] = (SearchURL, ...)

    # nothing special needed for textbox_actions; already handled above

    if not top_level:
        raise ValueError(
            "Cannot build a pruned response‑format — no actions had apply=True.",
        )

    # “ActionSelection” is the same top‑level model name used originally
    return create_model("ActionSelection", **top_level)


# === helper to expose available actions to the GUI =======================
def list_available_actions(
    tabs: List[str],
    buttons: Optional[List[Tuple[int, str]]] | None = None,
    state: BrowserState = None,  # ← add this default
) -> dict[str, list[str]]:
    """
    Return a mapping {group_name: [field_names,…]} describing every action
    that would appear in the full response‑format schema given the current
    set of browser tabs and visible buttons.
    """
    fmt = _create_full_response_format(tabs, buttons, state)
    base = {
        "tab_actions": list(
            fmt.model_fields["tab_actions"].annotation.model_fields,
        ),
        "scroll_actions": list(
            fmt.model_fields["scroll_actions"].annotation.model_fields,
        ),
        "button_actions": list(
            fmt.model_fields["button_actions"].annotation.model_fields,
        ),
        "search_actions": [
            name for name in ["search", "open_url"] if name in fmt.model_fields
        ],
    }

    if state and state.in_textbox:
        base["textbox_actions"] = sorted(TEXTBOX_COMMANDS)

    return base


# @unify.traced
def text_to_browser_action(
    text: str,
    screenshot: bytes,
    *,
    tabs: Optional[List[str]],
    buttons: Optional[List[Tuple[int, str]]] = None,
    history: ActionHistory = None,
    state: BrowserState = None,
) -> Optional[BaseModel]:
    if ADVANCED_MODE:
        response_format = _create_full_response_format(tabs, buttons, state)
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
        screenshot = base64.b64encode(screenshot).decode("utf-8")
        ret = client.generate(
            messages=client.messages
            + [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64," f"{screenshot}",
                            },
                        },
                    ],
                },
            ],
        )
        ret = response_format.model_validate_json(ret)
        ret, num_selected = _extract_applied_actions(ret)
        if num_selected == 0:
            raise Exception(
                f"At least one browser action must be selected, but agent responded with: {ret}",
            )
        if num_selected == 1:
            # only one candidate, can already return
            response_format = _build_pruned_response_format(ret)
            return response_format.model_validate(ret).model_dump()
    else:
        flat_actions = _list_flat_actions(tabs, buttons, state)
        lines = [
            "You control the browser with ONE low‑level action.",
            "Choose the best action‑prototype."
            "For search, open_url, scroll_up, and scroll_down",
            "please also include the query or number of pixels in the `value` field.",
            "",
            "Available prototypes:",
        ]
        lines += [f"- {a}" for a in flat_actions]
        lines += [
            "",
            "Respond ONLY with valid JSON matching:",
            '{"rationale": "...", "action": "<prototype>", "value": <value|null>}',
        ]
        sys_prompt = "\n".join(lines)

        client.set_endpoint("gpt-4o-mini@openai")
        client.set_system_message(sys_prompt)
        client.set_response_format(SimpleChoice)

        try:
            raw = client.generate(text)
            reply = SimpleChoice.model_validate_json(raw)
        except Exception:
            return None

        proto = reply.action
        if proto not in flat_actions:
            return None

        # ----- compose the real primitive string ---------------------------
        if reply.value:
            primitive = f"{proto} {str(reply.value)}"
        else:
            primitive = proto
        return {"rationale": reply.rationale, "action": primitive}

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
