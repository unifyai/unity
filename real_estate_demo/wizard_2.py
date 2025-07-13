from collections import defaultdict
import re
from typing import Callable, Literal, Union, Optional
from textwrap import dedent

from pydantic import BaseModel, Field, create_model


# ---------------------------------------------------------------------------
# Generic action objects
# ---------------------------------------------------------------------------
class GoNext(BaseModel):
    """advance to the next node"""
    next: Literal[True] = True


class GoBack(BaseModel):
    """go back to the previous node"""
    back: Literal[True] = True


class EndCall(BaseModel):
    """
    conclude the session and end the call.

    Can only be used in a terminal node; using EndCall will end the call and
    you won't be able to listen to the user anymore.
    """
    end_call: Literal[True] = True
    closing_message: str


class PromptUser(BaseModel):
    """prompt user (ask a question, provide clarification)"""
    prompt: str


class BaseGoToNode(BaseModel):
    """base class for “jump to a visited node” actions"""
    pass


class BaseDataFieldAction(BaseModel):
    """base class for data-field-manipulation actions"""
    pass


# ---------------------------------------------------------------------------
# Field widgets
# ---------------------------------------------------------------------------
class InputField:
    def __init__(self, id: str, label: str = None,
                 value=None, *, required: bool = True):
        self.id = id
        self.label = label if label is not None else self.id
        # keep an *immutable* copy of the default value
        self.original_value = value[:] if isinstance(value, list) else value
        self.value = value
        self.required = required

    # external API -----------------------------------------------------------
    def set_value(self, value):
        self.value = value

    def render(self):
        return dedent(f"""
{self.label} (Input Field)
{'[...]' if self.value is None else '[' + self.value + ']'}
        """).strip()


class RadioField:
    def __init__(self, id: str, label: str, options: list[str],
                 value=None, *, required: bool = True):
        self.id = id
        self.label = label
        self.options = options
        self.original_value = value[:] if isinstance(value, list) else value
        self.value = value
        self.required = required

    # external API -----------------------------------------------------------
    def set_value(self, value):
        self.value = value

    def render(self):
        str_options = "\n".join(
            "(x) {o} <- currently selected option".format(o=o)
            if self.value == o else
            "( ) {o}".format(o=o)
            for o in self.options
        )
        return dedent(f"""
{self.label} (Radio Field)
{str_options}
        """).strip()


class CheckBoxField:
    def __init__(self, id: str, label: str, options: list[str],
                 value=None, *, required: bool = True):
        self.id = id
        self.label = label
        self.options = options
        self.original_value = value[:] if isinstance(value, list) else value
        self.value = value
        self.required = required

    # external API -----------------------------------------------------------
    def set_value(self, value):
        self.value = value

    def render(self):
        chosen = self.value or []                     # ← guard against None
        str_options = "\n".join(
            "[x] {o} <- checked".format(o=o) if o in chosen else "[ ] {o}".format(o=o)
            for o in self.options
        )
        return dedent(f"""
{self.label} (CheckBox Field)
{str_options}
        """).strip()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class Node:
    """
    A dialog “screen”.  
    `fields` can now be a *static* list *or* a callable(ctx) → list
    so that dynamic screens work.
    """

    def __init__(
        self,
        id: str,
        title: str,
        instructions: str,
        fields: list | Callable[[dict], list],
        next: str | dict | Callable | None = None
    ):
        self.id = id
        self.title = title
        self.instructions = instructions

        # ------------------------------------------------------------------
        #  Make sure the attribute is ALWAYS present from the very start
        # ------------------------------------------------------------------
        self.is_terminal: bool = False   # will be corrected later if needed

        # the user-supplied spec (can be list or callable)
        self._raw_fields = fields
        self.fields: list = []

        # will be filled in by _materialise_fields ↓
        self.data: dict[str, any] = {}

        # placeholder for action ↔ field mapping
        self.action_to_field: dict[type[BaseModel], str] = {}

        # first materialisation – empty ctx for now
        self._materialise_fields({})

        # ----------- next-node handling -----------
        if isinstance(next, str):
            self.next = lambda ctx: next
        elif isinstance(next, dict):
            # assume there is exactly one field on that screen
            self.next = lambda ctx: next[ctx[self.fields[0].id]]
        elif next is None:
            self.next = None
            self.is_terminal = True
        else:
            self.next = next
            self.is_terminal = False

        # ---------- advance control ---------------
        self.can_advance = all(
            self.data.get(f.id) is not None for f in self.fields if f.required
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _materialise_fields(self, ctx: dict):
        """
        (Re)build the node’s field list, cached data, and action model.

        It now protects against missing keys by wrapping `ctx` in a
        defaultdict that returns `None` instead of raising KeyError.
        """
        # ---- make context “forgiving” ------------------------------------
        safe_ctx = defaultdict(lambda: None, ctx)

        # ---- compute the fresh list of fields ----------------------------
        new_fields = (
            self._raw_fields(safe_ctx)
            if callable(self._raw_fields)
            else (self._raw_fields or [])
        )
        new_fields = new_fields or []

        # No change → nothing to do
        if new_fields == self.fields:
            return

        # ---- keep any values that are still relevant ---------------------
        old_data = getattr(self, "data", {})
        self.fields = new_fields
        self.data = {f.id: old_data.get(f.id, f.value) for f in self.fields}

        # ---- rebuild bookkeeping ----------------------------------------
        self.set_up_action_model()
        self.can_advance = all(
            self.data.get(f.id) is not None for f in self.fields if f.required
        )

        # Update terminal flag once self.next exists
        if hasattr(self, "next"):
            self.is_terminal = self.next is None

    # ------------------------------------------------------------------ #
    # Action-model construction                                          #
    # ------------------------------------------------------------------ #
    def set_up_action_model(self):
        """
        Build or refresh the action-classes for the node.

        ── Key points ────────────────────────────────────────────────────
        1.  Preserve entries for action-classes that were created earlier
            and whose field is still present (to avoid KeyError later).
        2.  Guard against unknown field objects so we never reference an
            undefined `action_cls`.
        3.  Always leave `self.action_model` as a valid pydantic model,
            even when the node currently has zero data-fields.
        """
        # 1) start with previously-known mappings for still-present fields
        old_map = getattr(self, "action_to_field", {})
        current_ids = {f.id for f in self.fields}
        mapping: dict[type[BaseModel], str] = {
            cls: fid for cls, fid in old_map.items() if fid in current_ids
        }

        # 2) build fresh action classes for the current field list
        new_action_classes: list[type[BaseModel]] = []

        for field in self.fields:

            # -------- input ------------------------------------------------
            if isinstance(field, InputField):
                cls = create_model(
                    f"Fill{''.join(w.title() for w in field.label.split())}",
                    field_label=(Literal[field.label], field.label),
                    value=(str, Field(..., description="value to input")),
                    __doc__=f"Fill input field '{field.label}'.",
                )
                mapping[cls] = field.id
                new_action_classes.append(cls)

            # -------- radio -------------------------------------------------
            elif isinstance(field, RadioField):
                cls = create_model(
                    f"Select{''.join(w.title() for w in field.label.split())}",
                    field_label=(Literal[field.label], field.label),
                    value=(Literal[*field.options],
                           Field(..., description="option to select")),
                    __doc__=f"Select option for radio field '{field.label}'.",
                )
                mapping[cls] = field.id
                new_action_classes.append(cls)

            # -------- checkbox ---------------------------------------------
            elif isinstance(field, CheckBoxField):
                cls = create_model(
                    f"Check{''.join(w.title() for w in field.label.split())}",
                    field_label=(Literal[field.label], field.label),
                    value=(list[Literal[*field.options]],
                           Field(..., description="options to check")),
                    __doc__=f"Check options for checkbox field '{field.label}'.",
                )
                mapping[cls] = field.id
                new_action_classes.append(cls)

            # -------- unknown field type -----------------------------------
            else:
                # Skip silently – preserves robustness.
                continue

        # 3) commit the updated map
        self.action_to_field = mapping

        # 4) create the union type exposed to the LLM
        if new_action_classes:
            self.action_model = Union[*new_action_classes]
        else:
            class NoFieldAction(BaseModel):
                """This screen currently has no data-field actions."""
                pass

            self.action_model = NoFieldAction


    # ------------------------------------------------------------------ #
    # Public methods                                                     #
    # ------------------------------------------------------------------ #
    def play_actions(self, action: BaseModel):
        """
        Mutate field value according to `action`, refresh bookkeeping
        and re-evaluate whether we can advance.
        """
        action_cls = action.__class__
        field_id = self.action_to_field[action_cls]
        self.data[field_id] = action.value

        # Recompute advance‐ability
        self.can_advance = all(
            self.data.get(f.id) is not None for f in self.fields if f.required
        )

        # Refresh action model so the user can change mind again
        self.set_up_action_model()

    def render(self, ctx: dict = None):
        """Return a textual representation of the current node."""
        ctx = ctx or {}
        body = []
        for field in self.fields:
            field.set_value(self.data[field.id])
            body.append(field.render())

        instruction_block = self.instructions.format(**ctx) if self.instructions else ""
        body_str = "\n".join(body)
        return dedent(f"""
Node: {self.title}
Is Terminal Node?: {self.is_terminal}
Instructions: {instruction_block}
---

{body_str}
""").strip()

    def reset(self):
        """Reset node to its pristine state (used once per run)."""
        self.data = {}
        for field in self.fields:
            field.set_value(field.original_value)
            self.data[field.id] = field.value

        self.can_advance = all(
            self.data.get(f.id) is not None for f in self.fields if f.required
        )

        # Rebuild action map & model
        self.action_to_field = {}
        self.set_up_action_model()


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------
class Flow:
    def __init__(self, screens: list[Node], *, start: str | None = None):
        self.screens = screens

        # make sure everything is fresh
        for s in self.screens:
            s.reset()

        self.current_node: Node = (
            self.screens[0] if start is None
            else next(s for s in self.screens if s.id == start)
        )
        self.root_node = self.current_node

        # data collected so far
        self.ctx = dict(self.current_node.data)

        # history of visited nodes
        self.path: list[Node] = [self.current_node]

        # ensure dynamic fields are ready
        self.current_node._materialise_fields(self.ctx)

        # end-of-flow flag
        self.flow_done = False

    # ------------------------------------------------------------------ #
    # Action handling                                                    #
    # ------------------------------------------------------------------ #
    def play_actions(self, actions):
        """
        Execute a list of actions produced by the agent.  
        Stops once one of them moves the flow forward/backward because
        the agent will be called again afterwards.
        """
        if self.flow_done:
            return

        for action in actions:

            # ---------------- dialog housekeeping ---------------- #
            if isinstance(action, PromptUser):
                continue  # doesn't affect flow state

            # ---------------- backtracking ------------------------ #
            if isinstance(action, GoBack):
                self.path.pop()
                self.current_node = self.path[-1]
                self.current_node._materialise_fields(self.ctx)
                return

            # ---------------- advance to next --------------------- #
            if isinstance(action, GoNext):
                if not self.current_node.can_advance:
                    continue

                next_screen_id = self.current_node.next(self.ctx)
                self.current_node = next(s for s in self.screens if s.id == next_screen_id)
                self.ctx |= self.current_node.data
                self.path.append(self.current_node)
                # dynamic field refresh
                self.current_node._materialise_fields(self.ctx)
                return

            # ---------------- jump to visited node ---------------- #
            if isinstance(action, BaseGoToNode):
                self.current_node = next(s for s in self.screens if s.id == action.node_id)
                # ctx unchanged → still refresh fields (in case callable depends on ctx only)
                self.current_node._materialise_fields(self.ctx)
                return

            # ---------------- terminate --------------------------- #
            if isinstance(action, EndCall):
                self.flow_done = True
                return

            # ---------------- field manipulation ------------------ #
            self.current_node.play_actions(action)
            self.ctx |= self.current_node.data
            self.current_node._materialise_fields(self.ctx)

    # ------------------------------------------------------------------ #
    # Action-model presented to the LLM                                 #
    # ------------------------------------------------------------------ #
    def current_action_model(self) -> BaseModel:
        """
        Returns a *Union* pydantic model describing all actions
        currently available to the agent.
        """
        # dynamic extra actions
        extra_actions: list[type[BaseModel]] = []

        GoToNode = create_model(
            "GoToNode",
            node_id=(Literal[*[n.id for n in self.path]],
                     Field(..., description="Node ID to go to")),
            __doc__=(
                "Goes to the chosen node that you have already visited in your "
                "path. Useful when you need to go back to a specific node to "
                "modify data or start-over."
            ),
            __base__=BaseGoToNode,
        )

        if not self.current_node.is_terminal and self.current_node.can_advance:
            extra_actions.append(GoNext)

        if len(self.path) > 1:
            extra_actions.append(GoToNode)

        if self.current_node.is_terminal:
            extra_actions.append(EndCall)

        # wrap field actions so we can attach a short "update" string
        DataFieldAction = create_model(
            "DataFieldAction",
            update=(str, Field(default=None,
                               description="optional short (~3-5 words) friendly update for the user")),
            fields_actions=(list[Union[self.current_node.action_model]],
                            Field(..., description="data-field action(s) to take")),
            __base__=BaseDataFieldAction
        )

        # Union of everything
        return Union[DataFieldAction, *extra_actions]  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # Debug / display                                                    #
    # ------------------------------------------------------------------ #
    def render(self) -> str:
        """Return a pretty representation (for debugging/UI)."""
        return dedent(f"""
Current Path: {' > '.join(n.title for n in self.path)}

{self.current_node.render(self.ctx)}
        """).strip()
