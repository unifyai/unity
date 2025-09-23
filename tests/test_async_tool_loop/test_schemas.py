"""
pytest tests for the helper utilities:

* annotation_to_schema           – all supported annotation kinds
* method_to_schema               – schema structure & enum handling
"""

from __future__ import annotations

from enum import Enum

import unify
import pytest
from pydantic import BaseModel

import unity.common.llm_helpers as llmh


# --------------------------------------------------------------------------- #
#  TEST DATA TYPES FOR SCHEMA TESTS                                           #
# --------------------------------------------------------------------------- #
class ColumnType(str, Enum):
    str = "str"
    int = "int"


class Person(BaseModel):
    name: str
    age: int


# Helper function defined at module scope to stabilise type-hint resolution
def _tool_with_optional_mapping(
    references: dict[str, str] | None = None,
    k: int = 10,
) -> None:  # pragma: no cover - schema only
    return None


# --------------------------------------------------------------------------- #
#  annotation_to_schema                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "t, checker",
    [
        (str, lambda s: s == {"type": "string"}),
        (int, lambda s: s == {"type": "integer"}),
        (
            ColumnType,
            lambda s: s["type"] == "string" and set(s["enum"]) == {"str", "int"},
        ),
        (
            Person,
            lambda s: s["type"] == "object" and {"name", "age"} <= set(s["properties"]),
        ),
        (
            dict[str, int],
            lambda s: s["type"] == "object"
            and s["additionalProperties"]["type"] == "integer",
        ),
        (
            list[Person],
            lambda s: s["type"] == "array" and s["items"]["type"] == "object",
        ),
    ],
)
def test_annotation_to_schema_variants(t, checker):
    """Every major annotation flavour is converted correctly."""
    assert checker(llmh.annotation_to_schema(t))


# --------------------------------------------------------------------------- #
#  method_to_schema – enum round-trip                                         #
# --------------------------------------------------------------------------- #
def _demo_func(a: str, col: ColumnType):
    """Docstring for unit test."""
    return None


def test_method_to_schema_includes_enum():
    schema = llmh.method_to_schema(_demo_func)
    params = schema["function"]["parameters"]["properties"]
    assert params["a"]["type"] == "string"
    # Enum must appear with *exact* allowed literals
    assert params["col"]["enum"] == ["str", "int"]


# --------------------------------------------------------------------------- #
#  PRIVATE OPTIONAL ARGUMENTS ARE NOT EXPOSED                                 #
# --------------------------------------------------------------------------- #
def test_private_optional_parameters_are_hidden_from_tool_schema() -> None:
    """
    *Optional* parameters whose names begin with an underscore (``_``)
    must **not** appear in the schema that is presented to the LLM.
    Required private parameters, however, *must* stay visible or the
    tool would become impossible to call – and their docs should stay too.
    """

    # ── 1. optional private argument should be hidden ─────────────────────
    @unify.traced
    def sample_tool(a: int, b: int = 0, _secret: str = "x") -> int:
        """
        Sample calculator.

        Args:
            a: first addend.
            b: second addend, defaults to 0.
            _secret: **internal** flag, never shown to the LLM.
        """
        return a + b

    schema = llmh.method_to_schema(sample_tool)
    props = schema["function"]["parameters"]["properties"]
    required = schema["function"]["parameters"]["required"]
    desc = schema["function"]["description"]

    # public arguments are present …
    assert "a" in props and "b" in props
    # … while the optional private one is not
    assert "_secret" not in props
    # and its doc-line has been pruned
    assert "_secret" not in desc

    # required list unchanged
    assert "a" in required and "b" not in required

    # ── 2. required private argument should be kept ───────────────────────
    @unify.traced
    def tool_with_required_private(x: int, _hidden: str) -> str:
        """
        Echo tool.

        Parameters
        ----------
        x : int
            Multiplier.
        _hidden : str
            Mandatory private value (must stay visible).
        """
        return _hidden * x

    schema2 = llmh.method_to_schema(tool_with_required_private)
    props2 = schema2["function"]["parameters"]["properties"]
    required2 = schema2["function"]["parameters"]["required"]
    desc2 = schema2["function"]["description"]

    # the *required* private parameter is still exposed …
    assert "_hidden" in props2 and "_hidden" in required2
    # … and its doc-line is still present
    assert "_hidden" in desc2


# --------------------------------------------------------------------------- #
#  `parent_chat_context` MUST NEVER BE EXPOSED                                #
# --------------------------------------------------------------------------- #
def test_parent_chat_context_parameter_is_always_hidden() -> None:
    """
    The special ``parent_chat_context`` argument is injected automatically by
    the tool-loop.  It must be hidden from both the schema **and** the
    docstring that is sent to the LLM.
    """

    @unify.traced
    def tool_with_ctx(a: int, parent_chat_context: list[dict]):
        """
        Dummy tool.

        Parameters
        ----------
        a : int
            Some value.
        parent_chat_context : list[dict]
            Internal plumbing, never surfaced.
        """
        return a

    @unify.traced
    def tool_with_ctx_optional(
        a: int,
        parent_chat_context: list[dict] | None = None,
    ):
        """
        Dummy tool (optional ctx).

        Args:
            a: Some value.
            parent_chat_context: Internal plumbing, never surfaced.
        """
        return a

    for fn in (tool_with_ctx, tool_with_ctx_optional):
        schema = llmh.method_to_schema(fn)
        props = schema["function"]["parameters"]["properties"]
        required = schema["function"]["parameters"]["required"]
        desc = schema["function"]["description"]

        assert "parent_chat_context" not in props
        assert "parent_chat_context" not in required
        # docstring has been scrubbed
        assert "parent_chat_context" not in desc


# --------------------------------------------------------------------------- #
#  OPTIONAL[Dict[str, str]] COLLAPSES TO OBJECT (NO STRING ALTERNATIVE)       #
# --------------------------------------------------------------------------- #
def test_optional_dict_parameter_collapses_without_string() -> None:
    """
    Optional[Dict[str, str]] should collapse to a plain object schema.
    Prior to the fix, NoneType was treated as "string", producing
    anyOf [object, string]. This test ensures only the object form remains.
    """

    schema = llmh.method_to_schema(_tool_with_optional_mapping)
    params = schema["function"]["parameters"]["properties"]
    refs_schema = params["references"]

    # Must be a plain object with string values
    assert "anyOf" not in refs_schema
    assert refs_schema["type"] == "object"
    assert refs_schema["additionalProperties"]["type"] == "string"
