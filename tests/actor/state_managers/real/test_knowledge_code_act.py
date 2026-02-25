"""Real KnowledgeManager routing tests for CodeActActor.

Validates that CodeActActor uses ``execute_function`` for simple single-primitive
knowledge operations, both with and without FunctionManager discovery tools.
"""

import pytest

from tests.helpers import _handle_project
from tests.actor.state_managers.utils import (
    assert_used_execute_function,
    make_code_act_actor,
)
from unity.knowledge_manager.types import ColumnType
from unity.manager_registry import ManagerRegistry


def _office_hours_fact_present() -> bool:
    km = ManagerRegistry.get_knowledge_manager()
    found = False
    tables = list(km._tables_overview(include_column_info=False).keys())
    for table_name in tables:
        try:
            rows_by_table = km._filter(tables=table_name, filter=None, limit=1000)
            rows = (
                rows_by_table.get(table_name, [])
                if isinstance(rows_by_table, dict)
                else []
            )
            for row in rows:
                row_str = str(row).lower()
                if (
                    "office" in row_str
                    and "hours" in row_str
                    and "9" in row_str
                    and "5" in row_str
                    and (
                        "pt" in row_str or ("pacific" in row_str and "time" in row_str)
                    )
                ):
                    found = True
                    break
            if found:
                break
        except Exception:
            continue
    return found


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager():
    """CodeAct routes read-only knowledge question via execute_function."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):
        km = ManagerRegistry.get_knowledge_manager()
        km._create_table(
            name="Policies",
            description="Company policies and procedures",
            columns={"title": ColumnType.str, "content": ColumnType.str},
        )
        km._add_rows(
            table="Policies",
            rows=[{"title": "Office Hours", "content": "Office hours are 9–5 PT."}],
        )

        handle = await actor.act(
            "What are our office hours?",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert result and len(str(result)) > 0
        assert_used_execute_function(handle)
        assert "primitives.knowledge.ask" in calls
        assert all(c.startswith("primitives.knowledge.") for c in calls)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_ask_calls_manager_with_fm_tools():
    """CodeAct routes knowledge query via execute_function even with FM discovery tools present."""
    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
    ) as (actor, _primitives, calls):
        km = ManagerRegistry.get_knowledge_manager()
        km._create_table(
            name="Policies",
            description="Company policies and procedures",
            columns={"title": ColumnType.str, "content": ColumnType.str},
        )
        km._add_rows(
            table="Policies",
            rows=[{"title": "Office Hours", "content": "Office hours are 9–5 PT."}],
        )

        handle = await actor.act(
            "What are our office hours?",
            clarification_enabled=False,
        )
        result = await handle.result()

        assert result and len(str(result)) > 0
        assert_used_execute_function(handle)
        assert "primitives.knowledge.ask" in calls
        assert all(c.startswith("primitives.knowledge.") for c in calls)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_update_calls_manager():
    """CodeAct routes knowledge mutation via execute_function."""
    async with make_code_act_actor(impl="real") as (actor, _primitives, calls):
        _km = ManagerRegistry.get_knowledge_manager()

        handle = await actor.act(
            "Store: Office hours are 9–5 PT.",
            clarification_enabled=False,
        )
        await handle.result()

        assert_used_execute_function(handle)
        assert "primitives.knowledge.update" in calls
        assert all(c.startswith("primitives.knowledge.") for c in calls)

        assert (
            _office_hours_fact_present()
        ), "Expected fact 'Office hours are 9–5 PT.' was not found in any knowledge table"


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.eval
@_handle_project
async def test_update_calls_manager_with_fm_tools():
    """CodeAct routes knowledge mutation via execute_function even with FM discovery tools present."""
    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
    ) as (actor, _primitives, calls):
        _km = ManagerRegistry.get_knowledge_manager()

        handle = await actor.act(
            "Store: Office hours are 9–5 PT.",
            clarification_enabled=False,
        )
        await handle.result()

        assert_used_execute_function(handle)
        assert "primitives.knowledge.update" in calls
        assert all(c.startswith("primitives.knowledge.") for c in calls)

        assert (
            _office_hours_fact_present()
        ), "Expected fact 'Office hours are 9–5 PT.' was not found in any knowledge table"
