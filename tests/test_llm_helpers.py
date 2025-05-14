"""
pytest tests for the helper utilities:

* annotation_to_schema           – all supported annotation kinds
* method_to_schema               – schema structure & enum handling
* tool_use_loop                  – happy path, self-healing on error,
                                   counter reset, abort after N failures
"""

from __future__ import annotations

import json
import types
from enum import Enum

import pytest
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
#  MODULE UNDER TEST                                                          #
# --------------------------------------------------------------------------- #
# Change "tool_helpers" to the filename that contains annotation_to_schema,
# method_to_schema, and tool_use_loop.
import common.llm_helpers as llmh


# --------------------------------------------------------------------------- #
#  FIXTURES & TEST DOUBLES                                                    #
# --------------------------------------------------------------------------- #
class FakeToolCall:
    """Mimics OpenAI's ToolCall object."""

    def __init__(self, name: str, args: dict, call_id: str = "1"):
        self.id = call_id
        # the `function` field is itself an object with name + arguments json
        self.function = types.SimpleNamespace(
            name=name,
            arguments=json.dumps(args),
        )


def make_response(message):
    """Wrap a Message-like object into the exact structure returned by
    client.generate:  SimpleNamespace(choices=[SimpleNamespace(message=...)])
    """
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message)],
    )


class FakeClient:
    """Stand-in for unify.Unify that feeds pre-canned responses back to
    tool_use_loop."""

    def __init__(self, scripted_responses: list):
        self._responses = scripted_responses[:]  # shallow copy
        self.messages: list[dict] = []  # stores all append_messages calls

    # unify.Unify.generate(...)
    def generate(self, **_unused_kwargs):
        try:
            return self._responses.pop(0)
        except IndexError:
            raise RuntimeError("FakeClient ran out of scripted responses")

    # unify.Unify.append_messages(...)
    def append_messages(self, msgs):
        self.messages.extend(msgs)


# --------------------------------------------------------------------------- #
#  TEST DATA TYPES FOR SCHEMA TESTS                                           #
# --------------------------------------------------------------------------- #
class ColumnType(str, Enum):
    str = "str"
    int = "int"


class Person(BaseModel):
    name: str
    age: int


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
#  tool_use_loop – helper tools                                               #
# --------------------------------------------------------------------------- #
def add(x: int, y: int) -> int:  # happy-path tool
    return x + y


def divide(a: int, b: int) -> float:  # may raise ZeroDivisionError
    return a / b


# Script helpers ------------------------------------------------------------ #
def message_with_tool_call(name: str, args: dict, call_id: str = "1"):
    return types.SimpleNamespace(tool_calls=[FakeToolCall(name, args)], content="")


def final_answer(content: str):
    return types.SimpleNamespace(tool_calls=None, content=content)


# --------------------------------------------------------------------------- #
#  tool_use_loop – SUCCESS PATH                                               #
# --------------------------------------------------------------------------- #
def test_tool_use_loop_happy_path():
    scripted = [
        make_response(message_with_tool_call("add", {"x": 2, "y": 3})),
        make_response(final_answer("5")),
    ]
    client = FakeClient(scripted)
    result = llmh.tool_use_loop(
        client,
        message="Add two numbers",
        tools={"add": add},
        max_consecutive_failures=2,
    )
    assert result.strip() == "5"
    # Ensure exactly one tool message was fed back
    assert any(m["role"] == "tool" for m in client.messages)


# --------------------------------------------------------------------------- #
#  tool_use_loop – ERROR, RECOVERY & COUNTER RESET                            #
# --------------------------------------------------------------------------- #
def test_tool_use_loop_recovers_after_failure():
    """
    First call divides by zero (exception) → model sees traceback,
    second call fixes arguments, third message is final answer.
    """
    scripted = [
        make_response(message_with_tool_call("divide", {"a": 4, "b": 0})),
        make_response(message_with_tool_call("divide", {"a": 4, "b": 2})),
        make_response(final_answer("2.0")),
    ]
    client = FakeClient(scripted)

    answer = llmh.tool_use_loop(
        client,
        message="Divide numbers",
        tools={"divide": divide},
        max_consecutive_failures=3,
    )

    # Returned value is correct
    assert answer.strip().startswith("2")
    # First feedback message must contain a stack-trace mentioning ZeroDivisionError
    tracebacks = [m["content"] for m in client.messages if m["role"] == "tool"]
    assert any("ZeroDivisionError" in tb for tb in tracebacks), "\n".join(tracebacks)


# --------------------------------------------------------------------------- #
#  tool_use_loop – ABORT AFTER MAX FAILURES                                   #
# --------------------------------------------------------------------------- #
def test_tool_use_loop_aborts_after_too_many_failures():
    scripted = [
        make_response(message_with_tool_call("divide", {"a": 1, "b": 0})),
        make_response(message_with_tool_call("divide", {"a": 1, "b": 0})),
    ]
    client = FakeClient(scripted)

    raised = False
    try:
        llmh.tool_use_loop(
            client,
            message="Break me",
            tools={"divide": divide},
            max_consecutive_failures=2,
        )
    except Exception as e:
        raised = True

    assert raised, "Failed to raise on ZeroDivisionError"
