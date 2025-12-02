from __future__ import annotations

import asyncio
import pytest

pytestmark = pytest.mark.eval

from tests.helpers import _handle_project
from unity.function_manager.function_manager import FunctionManager
from unity.skill_manager.skill_manager import SkillManager


def _tool_names_from_messages(msgs: list[dict]) -> list[str]:
    names: list[str] = []
    for m in msgs:
        if m.get("role") == "tool":
            name = str(m.get("name") or "")
            if name and not name.startswith("check_status_"):
                names.append(name)
    return names


ALLOWED_FM_TOOLS = {
    "list_functions",
    "search_functions",
    "search_functions_by_similarity",
    "get_precondition",
    "request_clarification",
}


@pytest.mark.asyncio
@_handle_project
async def test_ask_lists_seeded_functions_and_calls_fm_tools():
    fm = FunctionManager()
    src1 = (
        "def add(a: int, b: int) -> int:\n"
        '    """Add two numbers"""\n'
        "    return a + b\n"
    )
    src2 = (
        "def price_total(p: float, tax: float) -> float:\n"
        '    """Return total price including tax"""\n'
        "    return p + tax\n"
    )
    fm.add_functions(implementations=[src1, src2])

    sk = SkillManager()
    handle = await sk.ask(
        "List your available skills. Include the exact underlying function names in parentheses and show signatures.",
        _return_reasoning_steps=True,
    )
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    assert isinstance(answer, str) and answer.strip()

    # Validate that at least one stored function name appears in the answer
    stored_names = set(FunctionManager().list_functions().keys())
    assert any(name in answer for name in stored_names)

    # Only FunctionManager read-only tools should be invoked
    executed = _tool_names_from_messages(messages)
    assert executed, "Expected at least one tool call"
    assert (
        set(executed) <= ALLOWED_FM_TOOLS
    ), f"Unexpected tools executed: {sorted(set(executed) - ALLOWED_FM_TOOLS)}"


@pytest.mark.asyncio
@_handle_project
async def test_ask_keyword_search_finds_price_skill():
    fm = FunctionManager()
    src1 = (
        "def add(a: int, b: int) -> int:\n"
        '    """Add two numbers"""\n'
        "    return a + b\n"
    )
    src2 = (
        "def price_total(p: float, tax: float) -> float:\n"
        '    """Return total price including tax"""\n'
        "    return p + tax\n"
    )
    fm.add_functions(implementations=[src1, src2])

    sk = SkillManager()
    handle = await sk.ask(
        "Which skill mentions 'price' in its description? Please include the underlying function name exactly.",
        _return_reasoning_steps=True,
    )
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    assert isinstance(answer, str) and answer.strip()
    assert "price_total" in answer or "price total" in answer.lower()

    executed = _tool_names_from_messages(messages)
    assert executed
    # Prefer search by keyword or semantic similarity; be permissive across both
    assert any(
        n in ("search_functions", "search_functions_by_similarity") for n in executed
    )


@pytest.mark.asyncio
@_handle_project
async def test_ask_precondition_fetch_is_used():
    fm = FunctionManager()
    src = (
        "def price_total(p: float, tax: float) -> float:\n"
        '    """Return total price including tax"""\n'
        "    return p + tax\n"
    )
    fm.add_functions(
        implementations=src,
        preconditions={
            "price_total": {"requires": ["tax_rate"], "note": "needs config"},
        },
    )

    sk = SkillManager()
    handle = await sk.ask(
        "Does the 'price_total' function require any configuration or preconditions? Please check and summarise.",
        _return_reasoning_steps=True,
    )
    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)

    assert isinstance(answer, str) and answer.strip()
    executed = _tool_names_from_messages(messages)
    assert "get_precondition" in executed
