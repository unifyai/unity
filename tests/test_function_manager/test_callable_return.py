from __future__ import annotations

import shutil

import pytest

from tests.helpers import _handle_project
from unity.function_manager.execution_env import create_base_globals
from unity.function_manager.function_manager import FunctionManager


# Keep this minimal to make venv prep fast (no third-party deps).
MINIMAL_VENV_CONTENT = """
[project]
name = "test-venv"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
""".strip()


@_handle_project
def test_invalid_parameter_combinations_raise():
    fm = FunctionManager()

    with pytest.raises(ValueError, match="also_return_metadata"):
        fm.filter_functions(filter=None, also_return_metadata=True)

    with pytest.raises(ValueError, match="namespace"):
        fm.filter_functions(filter=None, return_callable=True)

    with pytest.raises(ValueError, match="also_return_metadata"):
        fm.search_functions(query="anything", also_return_metadata=True)

    with pytest.raises(ValueError, match="namespace"):
        fm.search_functions(query="anything", return_callable=True)


@_handle_project
@pytest.mark.asyncio
async def test_filter_return_callable_injects_dependency_chain():
    fm = FunctionManager()

    c_src = "async def c(x: int) -> int:\n    return x + 1\n"
    b_src = "async def b(x: int) -> int:\n    return (await c(x=x)) + 10\n"
    a_src = "async def a(x: int) -> int:\n    return (await b(x=x)) + 100\n"

    fm.add_functions(implementations=[a_src, b_src, c_src])

    ns = create_base_globals()
    callables = fm.filter_functions(
        filter="name == 'a'",
        limit=1,
        return_callable=True,
        namespace=ns,
    )

    assert len(callables) == 1
    assert "a" in ns and callable(ns["a"])
    assert "b" in ns and callable(ns["b"])
    assert "c" in ns and callable(ns["c"])

    result = await ns["a"](x=1)
    assert result == 112


@_handle_project
@pytest.mark.asyncio
async def test_dependency_injection_supports_indirect_calls_and_returned_functions():
    """
    Stress cases that require dependency tracking beyond direct `foo()` calls:
    - alias assignment: `fn = dep; await fn(...)`
    - returning function references: `return dep`
    """
    fm = FunctionManager()

    leaf_src = "async def leaf(x: int) -> int:\n    return x + 1\n"
    factory_src = "async def factory():\n    return leaf\n"
    # Use alias assignment + call the returned function
    use_src = (
        "async def use(x: int) -> int:\n"
        "    f = factory\n"
        "    fn = await f()\n"
        "    return await fn(x=x)\n"
    )

    fm.add_functions(implementations=[leaf_src, factory_src, use_src])

    ns = create_base_globals()
    callables = fm.filter_functions(
        filter="name == 'use'",
        limit=1,
        return_callable=True,
        namespace=ns,
    )
    assert len(callables) == 1
    assert "leaf" in ns and callable(ns["leaf"])
    assert "factory" in ns and callable(ns["factory"])
    assert "use" in ns and callable(ns["use"])

    assert await ns["use"](x=10) == 11


@_handle_project
def test_search_return_callable_also_returns_metadata():
    fm = FunctionManager()
    fm.add_functions(
        implementations="def add_numbers(a: int, b: int) -> int:\n    return a + b\n",
    )

    ns = create_base_globals()
    res = fm.filter_functions(
        filter="name == 'add_numbers'",
        limit=1,
        return_callable=True,
        namespace=ns,
        also_return_metadata=True,
    )

    assert isinstance(res, dict)
    assert set(res.keys()) == {"callables", "metadata"}
    assert isinstance(res["callables"], list)
    assert isinstance(res["metadata"], list)
    assert len(res["callables"]) == 1
    assert len(res["metadata"]) == 1


@_handle_project
def test_circular_dependency_injection_does_not_loop():
    fm = FunctionManager()

    a_src = "async def a() -> int:\n    return await b()\n"
    b_src = "async def b() -> int:\n    return await a()\n"

    fm.add_functions(implementations=[a_src, b_src])

    ns = create_base_globals()
    callables = fm.filter_functions(
        filter="name == 'a'",
        limit=1,
        return_callable=True,
        namespace=ns,
    )

    assert len(callables) == 1
    assert "a" in ns and callable(ns["a"])
    assert "b" in ns and callable(ns["b"])


@_handle_project
@pytest.mark.asyncio
async def test_search_return_callable_venv_proxy_executes():
    fm = FunctionManager()

    venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)
    venv_dir = fm._get_venv_dir(venv_id)

    try:
        fm.add_functions(
            implementations="def add_numbers(a: int, b: int) -> int:\n    return a + b\n",
        )
        function_id = fm.list_functions()["add_numbers"]["function_id"]
        assert fm.set_function_venv(function_id=function_id, venv_id=venv_id) is True

        ns = create_base_globals()
        callables = fm.filter_functions(
            filter="name == 'add_numbers'",
            limit=1,
            return_callable=True,
            namespace=ns,
        )

        assert len(callables) == 1
        proxy = callables[0]
        assert getattr(proxy, "__name__", None) == "add_numbers"

        result = await proxy(a=3, b=5)
        assert result == 8
    finally:
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)


@_handle_project
@pytest.mark.asyncio
async def test_similarity_search_return_callable_monkeypatched(monkeypatch):
    fm = FunctionManager()

    fake_record = {
        "name": "foo",
        "argspec": "(x: int) -> int",
        "docstring": "Add one.",
        "implementation": "async def foo(x: int) -> int:\n    return x + 1\n",
        "calls": [],
        "venv_id": None,
        "is_primitive": False,
    }

    def _fake_table_search_top_k(*args, **kwargs):
        return [dict(fake_record)]

    monkeypatch.setattr(
        "unity.function_manager.function_manager.table_search_top_k",
        _fake_table_search_top_k,
    )

    ns = create_base_globals()
    res = fm.search_functions(
        query="irrelevant",
        n=1,
        return_callable=True,
        namespace=ns,
        also_return_metadata=True,
        include_primitives=False,
    )

    assert isinstance(res, dict)
    assert set(res.keys()) == {"callables", "metadata"}
    assert len(res["callables"]) == 1
    assert len(res["metadata"]) == 1

    fn = res["callables"][0]
    assert callable(fn)
    assert "foo" in ns
    assert await fn(x=1) == 2


@_handle_project
@pytest.mark.asyncio
async def test_dependency_injection_handles_custom_decorators_and_annotations(
    tmp_path,
    monkeypatch,
):
    """
    Advanced dependency injection cases:
    - decorator dependency: `@my_decorator` must exist at exec-time
    - typing.Annotated metadata dependency: `typing.Annotated[int, validator]`
      must resolve at exec-time (and can be reflected via get_type_hints)
    """
    # Create a temporary "custom functions" folder and point collector at it.
    custom_dir = tmp_path / "custom_functions"
    custom_dir.mkdir(parents=True, exist_ok=True)
    mod_path = custom_dir / "advanced_custom.py"
    mod_path.write_text(
        (
            "import typing\n"
            "from unity.function_manager.custom import custom_function\n"
            "\n"
            "@custom_function()\n"
            "def my_decorator(fn):\n"
            "    fn._decorated_marker = True\n"
            "    return fn\n"
            "\n"
            "@custom_function()\n"
            "@my_decorator\n"
            "async def decorated(x: int) -> int:\n"
            "    return x\n"
            "\n"
            "@custom_function()\n"
            "def validator(x: int) -> bool:\n"
            "    return x > 0\n"
            "\n"
            "@custom_function()\n"
            "async def annotated(x: typing.Annotated[int, validator]) -> int:\n"
            "    # Force annotation evaluation to ensure validator is resolvable.\n"
            "    hints = typing.get_type_hints(annotated, include_extras=True)\n"
            "    _ = hints['x'].__metadata__[0]\n"
            "    return x\n"
            "\n"
            "@custom_function()\n"
            "async def leaf(x: int) -> int:\n"
            "    return x + 1\n"
            "\n"
            "@custom_function()\n"
            "async def inner(x: int) -> int:\n"
            "    return await leaf(x=x)\n"
            "\n"
            "@custom_function()\n"
            "async def outer(x: int) -> int:\n"
            "    fn = inner\n"
            "    return await fn(x=x)\n"
        ),
        encoding="utf-8",
    )

    from unity.function_manager import custom_functions as _cf

    monkeypatch.setattr(_cf, "_get_custom_functions_folder", lambda: custom_dir)

    fm = FunctionManager()
    fm.sync_custom_functions()

    ns = create_base_globals()

    # (1) Decorator dependency resolution
    callables = fm.filter_functions(
        filter="name == 'decorated'",
        limit=1,
        return_callable=True,
        namespace=ns,
    )
    assert len(callables) == 1
    assert "my_decorator" in ns and callable(ns["my_decorator"])
    assert await ns["decorated"](x=1) == 1
    assert getattr(ns["decorated"], "_decorated_marker", False) is True

    # (2) Annotation dependency resolution
    callables = fm.filter_functions(
        filter="name == 'annotated'",
        limit=1,
        return_callable=True,
        namespace=ns,
    )
    assert len(callables) == 1
    assert "validator" in ns and callable(ns["validator"])
    assert await ns["annotated"](x=5) == 5

    # (3) Nested injection for custom functions (outer -> inner -> leaf, via alias)
    callables = fm.filter_functions(
        filter="name == 'outer'",
        limit=1,
        return_callable=True,
        namespace=ns,
    )
    assert len(callables) == 1
    assert "inner" in ns and callable(ns["inner"])
    assert "leaf" in ns and callable(ns["leaf"])
    assert await ns["outer"](x=10) == 11


@_handle_project
@pytest.mark.asyncio
async def test_dependency_injection_supports_user_defined_forward_ref_string_annotations():
    """
    Validate "user-defined" annotation names expressed as forward-ref strings.

    This mirrors patterns like:
      group_by: 'Optional[GroupBy | str]' = None
      time_period: 'TimePeriod' = 'day'
      -> 'MetricResult'

    The key behavior we care about:
    - exec() succeeds (because annotations are strings)
    - if code later resolves hints (e.g. via typing.get_type_hints), it works
      provided the caller injected the referenced types into the namespace.
    """
    fm = FunctionManager()

    metric_src = (
        "async def metric(\n"
        "    group_by: 'Optional[GroupBy | str]' = None,\n"
        "    start_date: 'Optional[str]' = None,\n"
        "    end_date: 'Optional[str]' = None,\n"
        "    time_period: 'TimePeriod' = 'day',\n"
        "    include_plots: 'bool' = False,\n"
        ") -> 'MetricResult':\n"
        "    hints = typing.get_type_hints(metric, include_extras=True)\n"
        "    group_by_str = str(hints['group_by'])\n"
        "    tp = hints['time_period']\n"
        "    ret = hints['return']\n"
        "    return (group_by_str, tp.__name__, ret.__name__)\n"
    )

    fm.add_functions(implementations=[metric_src])

    class GroupBy:  # user-defined type
        pass

    class TimePeriod:  # user-defined type
        pass

    class MetricResult:  # user-defined type
        pass

    ns = create_base_globals()
    ns["GroupBy"] = GroupBy
    ns["TimePeriod"] = TimePeriod
    ns["MetricResult"] = MetricResult

    callables = fm.filter_functions(
        filter="name == 'metric'",
        limit=1,
        return_callable=True,
        namespace=ns,
    )
    assert len(callables) == 1
    assert "metric" in ns and callable(ns["metric"])

    group_by_str, time_period_name, return_name = await ns["metric"]()
    assert "GroupBy" in group_by_str
    assert "str" in group_by_str
    assert "NoneType" in group_by_str
    assert time_period_name == "TimePeriod"
    assert return_name == "MetricResult"


@_handle_project
@pytest.mark.asyncio
async def test_similarity_search_return_callable_forward_ref_annotations_just_work(
    monkeypatch,
):
    """
    Validate the CodeActActor-style flow:
      fm.search_functions(..., return_callable=True)

    Caller provides only `create_base_globals()` (no manual injection of
    GroupBy/TimePeriod/MetricResult), and the returned callable should still
    be able to resolve forward-ref annotations without NameError.
    """
    fm = FunctionManager()

    fake_record = {
        "name": "metric",
        "argspec": "(...)",
        "docstring": "Metric query.",
        "implementation": (
            "async def metric(\n"
            "    group_by: 'Optional[GroupBy | str]' = None,\n"
            "    start_date: 'Optional[str]' = None,\n"
            "    end_date: 'Optional[str]' = None,\n"
            "    time_period: 'TimePeriod' = 'day',\n"
            "    include_plots: 'bool' = False,\n"
            ") -> 'MetricResult':\n"
            "    hints = typing.get_type_hints(metric, include_extras=True)\n"
            "    group_by_str = str(hints['group_by'])\n"
            "    tp = hints['time_period']\n"
            "    ret = hints['return']\n"
            "    return (group_by_str, tp.__name__, ret.__name__)\n"
        ),
        # No dependency graph info available from similarity search in this test.
        "calls": [],
        "venv_id": None,
        "is_primitive": False,
    }

    def _fake_table_search_top_k(*args, **kwargs):
        return [dict(fake_record)]

    monkeypatch.setattr(
        "unity.function_manager.function_manager.table_search_top_k",
        _fake_table_search_top_k,
    )

    ns = create_base_globals()
    res = fm.search_functions(
        query="irrelevant",
        n=1,
        return_callable=True,
        namespace=ns,
        also_return_metadata=True,
        include_primitives=False,
    )

    assert isinstance(res, dict)
    assert set(res.keys()) == {"callables", "metadata"}
    assert len(res["callables"]) == 1

    fn = res["callables"][0]
    group_by_str, time_period_name, return_name = await fn()

    # "Just works" = no NameError, and forward-ref symbols are resolvable.
    assert "GroupBy" in group_by_str
    assert "str" in group_by_str
    assert "NoneType" in group_by_str
    assert time_period_name == "TimePeriod"
    assert return_name == "MetricResult"
