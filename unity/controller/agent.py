from __future__ import annotations

from typing import List, Optional, Union, Tuple, Dict, Any, Type, get_origin
import base64
import time
from datetime import datetime, timezone
from pydantic import BaseModel, create_model, Field
import inspect
import os

from .helpers import _pascal, _slug
from .commands import *
from .states import ActionHistory, BrowserState
from .action_filter import get_valid_actions
from .sys_msgs import (
    PRIMITIVE_TO_BROWSER_ACTION_CANDIDATES,
    PRIMITIVE_TO_BROWSER_ACTION,
    PRIMITIVE_TO_BROWSER_MULTI_STEP,
    PRIMITIVE_TO_BROWSER_ACTION_SIMPLE,
)
from ..constants import LOGGER

import unify
import json

client = unify.Unify(
    cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
    traced=json.loads(os.environ.get("UNIFY_TRACED", "false")),
)
client.set_system_message(PRIMITIVE_TO_BROWSER_ACTION_CANDIDATES)

SCROLLING_STATE = None
ADVANCED_MODE = False

# helpers #


class InvalidActionError(Exception):
    pass


def _list_valid_actions(tabs, buttons, state) -> list[str]:
    """
    Return the flat list of valid primitive strings for the current state.
    Uses the same logic as the advanced response‑format builder.
    """

    if not tabs and not buttons and not state:
        return [CMD_OPEN_URL.replace(" *", "")]

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

    # Normalize any trailing "_*" pattern (e.g. enter_text_* → enter_text)
    norm: list[str] = []
    for a in flat:
        if a.endswith("_*"):
            a = a[:-2]
        # also strip explicit space-star pattern used by enter_text * etc.
        a = a.replace(" *", "").replace("*", "").replace(" ", "")
        norm.append(a)

    norm.append("no_op_element_not_found")
    norm.append("close_browser")
    return sorted(set(norm))


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

    speed: Optional[int] = Field(
        None,
        description="Pixels per second for auto-scroll (default 250 if omitted).",
    )
    rationale: Optional[str] = Field(
        ...,
        description="Explanation for your decision whether or not to apply this action.",
    )
    apply: bool = Field(..., description="Decision to apply this action or not.")


class StartScrollingDown(BaseModel):
    """
    Start gently scrolling downwards, until another stop action is given.
    """

    speed: Optional[int] = Field(
        None,
        description="Pixels per second for auto-scroll (default 250 if omitted).",
    )
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
    CMD_PRESS_KEY: "Press the specified key (e.g. '1', 'a', 'Escape').",
    CMD_HOLD_SHIFT: "Hold the Shift key down.",
    CMD_HOLD_CTRL: "Hold the Control key down.",
    CMD_HOLD_ALT: "Hold the Alt key down.",
    CMD_HOLD_CMD: "Hold the Command (⌘) key down.",
    CMD_RELEASE_SHIFT: "Release the Shift key.",
    CMD_RELEASE_CTRL: "Release the Control key.",
    CMD_RELEASE_ALT: "Release the Alt key.",
    CMD_RELEASE_CMD: "Release the Command (⌘) key.",
    CMD_CLICK_OUT: "Click outside the text-box to blur focus.",
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
            __doc__=f'{mode} the "{title}" tab.',
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
        # Sanitize label – drop any invalid Unicode surrogates that can break
        # class creation when used inside __doc__ strings.
        safe_text = raw_text.encode("utf-8", "ignore").decode("utf-8", "ignore")

        base_slug = _slug(safe_text)  # "sign_in" (ASCII only)
        if not base_slug:
            base_slug = "button"
        slug = f"{idx}_{base_slug}"  # "7_sign_in"
        pascal = _pascal(slug)  # "7SignIn"

        field_name = f"click_button_{slug}"
        model_name = f"ClickButton{pascal}"
        doc = f'Click the "{safe_text}" button (element #{idx}).'

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


class CommandSequence(BaseModel):
    """
    A sequence of low-level browser commands in execution order.
    """

    rationale: str = Field(..., description="Why you chose this action.")
    actions: List[str] = Field(..., description="The sequence of actions to take.")


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

    # dialog actions group (only when a dialog is open)
    dialog_actions = {}
    if state and state.dialog_open:
        if include(CMD_ACCEPT_DIALOG):
            dialog_actions[CMD_ACCEPT_DIALOG] = AcceptDialog
        if include(CMD_DISMISS_DIALOG):
            dialog_actions[CMD_DISMISS_DIALOG] = DismissDialog
        # prompt dialogs need text field
        if state.dialog_type == "prompt" and include(
            CMD_TYPE_DIALOG.replace(" *", "_"),
        ):
            dialog_actions[CMD_TYPE_DIALOG.replace(" *", "")] = TypeDialog

    # popup actions (select/close per title)
    popup_actions = {}
    if state and state.popups:
        popup_actions.update(
            {
                k: v
                for k, v in _construct_select_popup_actions(state.popups).items()
                if include(k)
            },
        )
        popup_actions.update(
            {
                k: v
                for k, v in _construct_close_popup_actions(state.popups).items()
                if include(k)
            },
        )

    # text‑box actions (only when we're actually in a text input)
    textbox_actions = {}
    if state and state.in_textbox:
        textbox_actions = {
            k: v for k, v in _construct_textbox_actions().items() if include(k)
        }

    # Helper to build a Pydantic model from a mapping of field->type
    # ensuring each field is provided as a **(annotation, default)** tuple
    # to satisfy Pydantic v2's stricter requirements.
    def _make_group_model(model_name: str, mapping: dict[str, type[BaseModel]]):
        if mapping:
            return create_model(
                model_name,
                **{k: (cls, ...) for k, cls in mapping.items()},
            )
        # Empty mapping → fall back to a blank BaseModel subclass
        return create_model(model_name, __base__=BaseModel)

    fields = {
        "tab_actions": (_make_group_model("TabActions", tab_actions), ...),
        "scroll_actions": (
            _make_group_model("ScrollActions", scroll_actions),
            ...,
        ),
        "button_actions": (
            _make_group_model("ButtonActions", button_actions),
            ...,
        ),
        "textbox_actions": (
            _make_group_model("TextboxActions", textbox_actions),
            ...,
        ),
        "dialog_actions": (
            _make_group_model("DialogActions", dialog_actions),
            ...,
        ),
        "popup_actions": (
            _make_group_model("PopupActions", popup_actions),
            ...,
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
        "dialog_actions",
        "popup_actions",
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
        CMD_ACCEPT_DIALOG: AcceptDialog,
        CMD_DISMISS_DIALOG: DismissDialog,
        CMD_TYPE_DIALOG.replace(" *", ""): TypeDialog,
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
            __doc__=f'Select the "{title}" tab.',
            **_response_fields,
        )

    elif action_name.startswith("close_tab_"):
        slug = action_name[len("close_tab_") :]
        title = slug.replace("_", " ").replace("-", " ").title()
        return create_model(
            f"CloseTab{_pascal(slug)}",
            __doc__=f'Close the "{title}" tab.',
            **_response_fields,
        )

    # ---- dynamic button actions ------------------------------------------
    elif action_name.startswith("click_button_"):
        slug = action_name[len("click_button_") :]
        text = slug.replace("_", " ").replace("-", " ").title()
        return create_model(
            f"ClickButton{_pascal(slug)}",
            __doc__=f'Click the "{text}" button.',
            **_response_fields,
        )

    elif action_name.startswith("select_popup_"):
        slug = action_name[len("select_popup_") :]
        title = slug.replace("_", " ").replace("-", " ").title()
        return create_model(
            f"SelectPopup{_pascal(slug)}",
            __doc__=f'Select the popup window titled "{title}".',
            **_response_fields,
        )

    elif action_name.startswith("close_popup_"):
        slug = action_name[len("close_popup_") :]
        title = slug.replace("_", " ").replace("-", " ").title()
        return create_model(
            f"ClosePopup{_pascal(slug)}",
            __doc__=f'Close the popup window titled "{title}".',
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
        # Skip search & open_url here; they'll be handled explicitly below
        if group in ("search", "open_url"):
            continue

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

    # "ActionSelection" is the same top‑level model name used originally
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
        "dialog_actions": (
            list(fmt.model_fields["dialog_actions"].annotation.model_fields)
            if "dialog_actions" in fmt.model_fields
            else []
        ),
        "popup_actions": (
            list(fmt.model_fields["popup_actions"].annotation.model_fields)
            if "popup_actions" in fmt.model_fields
            else []
        ),
        "search_actions": [
            name for name in ["search", "open_url"] if name in fmt.model_fields
        ],
    }

    if state and state.in_textbox:
        base["textbox_actions"] = sorted(TEXTBOX_COMMANDS)

    return base


def text_to_browser_action(
    text: str,
    screenshot: base64,
    *,
    tabs: Optional[List[str]],
    buttons: Optional[List[Tuple[int, str]]] = None,
    history: ActionHistory = None,
    state: BrowserState = None,
    multi_step_mode: bool = False,
) -> Optional[BaseModel]:
    t0 = time.perf_counter()
    t = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
    LOGGER.info(
        f"\n🤖 Controller: text command to browser action with text: {text}... ⏳ [⏱️ {t}]\n",
    )
    if ADVANCED_MODE:
        response_format = _create_full_response_format(tabs, buttons, state)
        client.set_endpoint("o4-mini@openai")
        history_msg = (
            "\n\nThe low-level action history (most recent first) is as follows:\n"
            + "\n".join(f"{r['timestamp']:.0f}: {r['command']}" for r in history[-20:])
        )

        state_msg = f"""\n\nThe current state of the browser is as follows:
        url: {state.url if state else ''}
        title: {state.title if state else ''}
        scroll_y: {state.scroll_y if state else 0}
        auto_scroll: {state.auto_scroll if state else None}
        in_textbox: {state.in_textbox if state else False}
        """

        client.set_system_message(
            PRIMITIVE_TO_BROWSER_ACTION_CANDIDATES + history_msg + state_msg,
        )
        client.set_response_format(response_format)
        content = [
            {
                "type": "text",
                "text": text,
            },
        ]
        if screenshot:
            content += [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64," f"{screenshot}",
                    },
                },
            ]
        ret = client.generate(
            messages=client.messages
            + [
                {
                    "role": "user",
                    "content": content,
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
            t = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
            LOGGER.info(
                f"\n🤖 Controller: text command to browser action ✅ [⏱️ {t}] [⏩{(time.perf_counter() - t0):.3g}s]\n",
            )
            return response_format.model_validate(ret).model_dump()

        # decide among the candidate actions
        client.set_endpoint("o4-mini@openai")
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
        t = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
        LOGGER.info(
            f"\n🤖 Controller: text command to browser action ✅ [⏱️ {t}] [⏩{(time.perf_counter() - t0):.3g}s]\n",
        )
        return response_format.model_validate(ret).model_dump()
    else:
        multi_step_mode = multi_step_mode or (
            state and hasattr(state, "in_textbox") and state.in_textbox
        )  # if in textbox, use multi-step mode for enabling key combinations
        valid_actions = _list_valid_actions(tabs, buttons, state)
        print(valid_actions)
        lines = (
            [PRIMITIVE_TO_BROWSER_MULTI_STEP]
            if multi_step_mode
            else [PRIMITIVE_TO_BROWSER_ACTION_SIMPLE]
        )
        response_format = CommandSequence if multi_step_mode else SimpleChoice

        def _format_action(a: str):
            ret = f"- {a}"
            if a in ("search", "open_url"):
                ret += " (please also include the query such that '<search/open_url> <query>')"
            elif a in ("scroll_up", "scroll_down"):
                ret += " (please also include the *non-negative* number of pixels such that '<scroll_up/scroll_down> <pixels>')"
            elif a in ("start_scrolling_up", "start_scrolling_down"):
                ret += " (please also include the *non-negative* speed (pixels/second) such that '<start_scrolling_up/start_scrolling_down> <speed>')"
            elif a in ("press_key"):
                ret += " (please also include a *single* character or digit to press such that '<press_key> <char/digit>')"
            elif a in ("click_button"):
                ret += " (please also include the *number* of the button to click such that '<click_button> <number>'"
            return ret

        lines += [_format_action(a) for a in valid_actions]
        sys_prompt = "\n".join(lines)

        client.set_endpoint("o4-mini@openai")
        client.set_system_message(sys_prompt)
        client.set_response_format(response_format)

        content = [
            {
                "type": "text",
                "text": text,
            },
        ]
        if screenshot:
            content += [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64," f"{screenshot}",
                    },
                },
            ]
        raw = client.generate(
            messages=client.messages
            + [
                {
                    "role": "user",
                    "content": content,
                },
            ],
        )

        reply = response_format.model_validate_json(raw)
        actions = reply.actions if multi_step_mode else [reply.action]

        if any("no_op_element_not_found" in action for action in actions):
            raise InvalidActionError(
                f"Agent could not perform action: '{text}'. Reason: {reply.rationale}",
            )

        if not all(action.split(" ")[0] in valid_actions for action in actions):
            raise InvalidActionError(
                f"Invalid action is present: {reply.model_dump_json()}",
            )

        if not multi_step_mode and reply.value:
            actions = [f"{actions[0]} {str(reply.value)}"]
        t = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
        LOGGER.info(
            f"\n🤖 Controller: text command to browser action ✅ [⏱️ {t}] [⏩{(time.perf_counter() - t0):.3g}s]\n {actions}",
        )
        return {"rationale": reply.rationale, "action": actions}


# ---- Dialog / Popup Schemas (NEW) ----------------------------------


class AcceptDialog(BaseModel):
    """Accept / OK the currently open JavaScript dialog."""

    rationale: Optional[str] = Field(
        None,
        description="Why you want / don't want to accept the dialog.",
    )
    apply: bool = Field(..., description="Accept (click OK) if true.")


class DismissDialog(BaseModel):
    """Dismiss / Cancel the currently open JavaScript dialog."""

    rationale: Optional[str] = Field(
        None,
        description="Why you want / don't want to dismiss the dialog.",
    )
    apply: bool = Field(..., description="Dismiss (click Cancel) if true.")


class TypeDialog(BaseModel):
    """Type text into a JavaScript prompt dialog then accept it."""

    text: str = Field(..., description="Text to send to the prompt()")
    rationale: Optional[str] = Field(
        None,
        description="Why you chose this text or decided not to send it.",
    )
    apply: bool = Field(..., description="Type the text and accept if true.")


# ---------------------------------------------------------------------
def _construct_select_popup_actions(pop_titles: List[str]):
    mapping = {}
    for title in pop_titles:
        slug = _slug(title)
        action_name = f"select_popup_{slug}"
        mapping[action_name] = create_model(
            f"SelectPopup{_pascal(slug)}",
            __doc__=f'Select the popup window titled "{title}".',
            **_response_fields,
        )
    return mapping


def _construct_close_popup_actions(pop_titles: List[str]):
    mapping = {}
    for title in pop_titles:
        slug = _slug(title)
        action_name = f"close_popup_{slug}"
        mapping[action_name] = create_model(
            f"ClosePopup{_pascal(slug)}",
            __doc__=f'Close the popup window titled "{title}".',
            **_response_fields,
        )
    return mapping


# -----------------------------------------------------------------------------
#  Generic "observe" helper – answers questions about the *current* browser view
# -----------------------------------------------------------------------------
_OBSERVE_PROMPT = (
    "You are a browsing assistant. Below is the complete browser context "
    "(state, tab titles, visible elements, action history). Answer the user's "
    "question solely from this information and the screenshot. Return JSON "
    "that matches the response schema and nothing else."
)


def _wrap_type(tp: Any):  # type: ignore[override]
    """Return a Pydantic model appropriate for *tp* (primitive or model)."""
    # Handle string type hints
    if isinstance(tp, str):
        tp_lower = tp.lower()
        if tp_lower == "boolean":
            tp = bool
        elif tp_lower in ("yes or no", "yes/no"):
            tp = bool
        else:
            # For other string hints, use str type
            tp = str

    if tp in {str, bool, int, float}:
        # primitive → wrap in single-field model
        return create_model("AnswerModel", answer=(tp, ...))

    if inspect.isclass(tp) and issubclass(tp, BaseModel):
        return tp  # already a model

    if callable(tp) and not isinstance(tp, type):
        try:
            # Check if it's a function that returns a Pydantic model
            model_instance = tp()
            if isinstance(model_instance, BaseModel):
                return type(model_instance)
        except Exception:
            # If it fails, then it's an unsupported type
            raise TypeError(
                f"Unsupported response_type: {tp!r}. It appears to be a function but not a Pydantic model factory.",
            )

    if get_origin(tp) is not None:  # e.g. Literal, Annotated
        return create_model("AnswerModel", answer=(tp, ...))

    raise TypeError(f"Unsupported response_type: {tp!r}")


def ask_llm(
    question: str,
    *,
    response_format: Type = str,
    context: dict[str, Any] | None = None,
    screenshots: bytes | None = None,
) -> Any:  # noqa: ANN401
    """Call the underlying LLM with the given *question* and browser *context*.

    Parameters
    ----------
    question : str
        Natural-language question to ask.
    response_format : Type, default str
        Desired Python / Pydantic type for the answer.
    context : dict | None
        Rich browser-context payload (state, elements, tabs, history …).
    screenshots : bytes | None
        Base-64 JPEG screenshots (before and after) of current viewport (optional).
    """

    Model = _wrap_type(response_format)

    client.set_endpoint("gemini-2.0-flash@vertex-ai")
    client.set_system_message(_OBSERVE_PROMPT)
    client.set_response_format(Model)

    # 1) build user message content (text + optional image)
    content = [{"type": "text", "text": question}]
    if screenshots:
        for label, png_bytes in screenshots.items():
            content.append({"type": "text", "text": f"## Screenshot: '{label}'"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{png_bytes}"},
                },
            )

    # 2) include browser context as separate system message (JSON string)
    if context:
        import json, textwrap

        ctx_dump = json.dumps(context, ensure_ascii=False, indent=2)
        ctx_txt = textwrap.indent(ctx_dump, "  ")
        client.messages.append(
            {
                "role": "system",
                "content": f"Browser-Context:\n{ctx_txt}",
            },
        )

    raw_json = client.generate(
        messages=client.messages + [{"role": "user", "content": content}],
    )
    parsed = Model.model_validate_json(raw_json)

    # unwrap when we used the primitive wrapper
    return getattr(parsed, "answer", parsed)
