# tests/test_hidden_parameters.py
from __future__ import annotations

import asyncio
import pytest
import unify

import unity.common.async_tool_loop as llmh


# --------------------------------------------------------------------------- #
# Shared assertion helper                                                     #
# --------------------------------------------------------------------------- #
def _assert_internal_queues_hidden(fn) -> None:
    """
    Verify that *clarification_up_q* and *clarification_down_q* never show up
    in the generated schema nor in the docstring description that is forwarded
    to the LLM.
    """
    schema = llmh.method_to_schema(fn)
    props = schema["function"]["parameters"]["properties"]
    required = schema["function"]["parameters"]["required"]
    desc = schema["function"]["description"]

    for name in ("clarification_up_q", "clarification_down_q"):
        # ––– must be absent from JSON schema –––
        assert name not in props
        assert name not in required
        # ––– and also stripped from the docstring forwarded to the model –––
        assert name not in desc


# --------------------------------------------------------------------------- #
# 1. Original “Google style” example (colon after each name)                  #
# --------------------------------------------------------------------------- #
@unify.traced
def tool_google_style(
    a: int,
    clarification_up_q: asyncio.Queue[str],
    clarification_down_q: asyncio.Queue[str],
) -> int:
    """
    Google-style.

    Args:
        a: Some value.
        clarification_up_q: internal.
        clarification_down_q: internal.
    """
    return a


# --------------------------------------------------------------------------- #
# 2. New *slash-separated* NumPy variant: “name_a / name_b : type”            #
# --------------------------------------------------------------------------- #
@unify.traced
def tool_numpy_slash(
    a: int,
    clarification_up_q: asyncio.Queue[str] | None = None,
    clarification_down_q: asyncio.Queue[str] | None = None,
) -> int:
    """
    NumPy style — slash-separated synonyms.

    Parameters
    ----------
    a : int
        Some value.
    clarification_up_q / clarification_down_q : asyncio.Queue[str] | None
        Internal queues.
    """
    return a


# --------------------------------------------------------------------------- #
# 3. New *comma-separated* variant: “name_a, name_b, …” (no type, no colon)   #
# --------------------------------------------------------------------------- #
@unify.traced
def tool_numpy_commas(
    a: int,
    clarification_up_q: asyncio.Queue[str] | None = None,
    clarification_down_q: asyncio.Queue[str] | None = None,
) -> int:
    """
    NumPy style — comma-separated, no type / no colon.

    Parameters
    ----------
    a : int
        Some value.

    clarification_up_q, clarification_down_q
        Internal queues.
    """
    return a


# --------------------------------------------------------------------------- #
# Parametrised test that covers all three doc-string flavours                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fn",
    [tool_google_style, tool_numpy_slash, tool_numpy_commas],
    ids=["google", "numpy-slash", "numpy-commas"],
)
def test_hidden_parameters_are_removed_from_schema_and_docs(fn) -> None:
    """
    Any internal queues must be invisible to the API schema and the LLM
    description, regardless of how they are documented in the original
    docstring (Google, NumPy-slash, or NumPy-commas).
    """
    _assert_internal_queues_hidden(fn)
