"""
Tests for dependency_analysis module.
"""

from unity.function_manager.dependency_analysis import (
    collect_dependencies_from_source,
    detect_third_party_imports_from_source,
)


class TestCallableDependencies:
    """Tests for tracking callables passed as function arguments."""

    def test_callable_as_positional_arg(self):
        """Callable passed as positional argument is tracked."""
        source = """
def main():
    return bar(callback_fn)
"""
        known = {"callback_fn", "bar"}
        deps = collect_dependencies_from_source(source, known)
        assert "callback_fn" in deps
        assert "bar" in deps

    def test_callable_as_keyword_arg(self):
        """Callable passed as keyword argument is tracked."""
        source = """
def main():
    return bar(callback=callback_fn)
"""
        known = {"callback_fn", "bar"}
        deps = collect_dependencies_from_source(source, known)
        assert "callback_fn" in deps
        assert "bar" in deps

    def test_multiple_callable_args(self):
        """Multiple callables passed as arguments are all tracked."""
        source = """
def main():
    return executor(func_a, func_b, on_error=error_handler)
"""
        known = {"func_a", "func_b", "error_handler", "executor"}
        deps = collect_dependencies_from_source(source, known)
        assert deps == {"func_a", "func_b", "error_handler", "executor"}

    def test_assigned_callable_as_arg(self):
        """Callable assigned to variable then passed as arg is tracked."""
        source = """
def main():
    f = my_callback
    return executor(callback=f)
"""
        known = {"my_callback", "executor"}
        deps = collect_dependencies_from_source(source, known)
        assert "my_callback" in deps
        assert "executor" in deps

    def test_non_callable_arg_not_tracked(self):
        """Regular variables passed as args are not tracked as dependencies."""
        source = """
def main():
    x = 42
    return bar(x, y=100)
"""
        known = {"bar", "other_fn"}
        deps = collect_dependencies_from_source(source, known)
        assert deps == {"bar"}


class TestDirectCallDependencies:
    """Tests for direct function call dependency tracking."""

    def test_simple_call(self):
        """Direct function call is tracked."""
        source = """
def main():
    return helper()
"""
        known = {"helper"}
        deps = collect_dependencies_from_source(source, known)
        assert deps == {"helper"}

    def test_nested_calls(self):
        """Nested function calls are all tracked."""
        source = """
def main():
    return outer(inner())
"""
        known = {"outer", "inner"}
        deps = collect_dependencies_from_source(source, known)
        assert deps == {"outer", "inner"}

    def test_assigned_then_called(self):
        """Function assigned to variable then called is tracked."""
        source = """
def main():
    f = helper
    return f()
"""
        known = {"helper"}
        deps = collect_dependencies_from_source(source, known)
        assert deps == {"helper"}


class TestReturnDependencies:
    """Tests for returned function reference tracking."""

    def test_return_function(self):
        """Returned function reference is tracked."""
        source = """
def main():
    return helper
"""
        known = {"helper"}
        deps = collect_dependencies_from_source(source, known)
        assert deps == {"helper"}

    def test_return_assigned_function(self):
        """Returned variable holding function reference is tracked."""
        source = """
def main():
    f = helper
    return f
"""
        known = {"helper"}
        deps = collect_dependencies_from_source(source, known)
        assert deps == {"helper"}


class TestEnvironmentDependencies:
    """Tests for dotted environment call dependency tracking."""

    def test_primitive_call_detected(self):
        """Awaited primitives.contacts.ask() is captured as a dependency."""
        source = """
async def main():
    result = await primitives.contacts.ask("find John")
    return result
"""
        deps = collect_dependencies_from_source(
            source,
            set(),
            environment_namespaces=frozenset({"primitives"}),
        )
        assert "primitives.contacts.ask" in deps

    def test_computer_primitive_detected(self):
        """Awaited primitives.computer.screenshot() is captured."""
        source = """
async def main():
    img = await primitives.computer.screenshot()
    return img
"""
        deps = collect_dependencies_from_source(
            source,
            set(),
            environment_namespaces=frozenset({"primitives"}),
        )
        assert "primitives.computer.screenshot" in deps

    def test_actor_act_detected(self):
        """Awaited primitives.actor.act() is captured."""
        source = """
async def main():
    handle = await primitives.actor.act("do the thing")
    return await handle.result()
"""
        deps = collect_dependencies_from_source(
            source,
            set(),
            environment_namespaces=frozenset({"primitives"}),
        )
        assert "primitives.actor.act" in deps

    def test_mixed_dependencies(self):
        """Both bare compositional and dotted environment deps are captured."""
        source = """
async def main():
    data = helper()
    result = await primitives.knowledge.ask(data)
    return result
"""
        deps = collect_dependencies_from_source(
            source,
            {"helper"},
            environment_namespaces=frozenset({"primitives"}),
        )
        assert deps == {"helper", "primitives.knowledge.ask"}

    def test_unknown_dotted_name_not_detected(self):
        """Dotted calls whose root is not in environment_namespaces are ignored."""
        source = """
async def main():
    return await random_thing.foo()
"""
        deps = collect_dependencies_from_source(
            source,
            set(),
            environment_namespaces=frozenset({"primitives"}),
        )
        assert deps == set()

    def test_no_environment_namespaces_backward_compat(self):
        """Omitting environment_namespaces preserves old behavior (no dotted deps)."""
        source = """
async def main():
    await primitives.contacts.ask("query")
    return helper()
"""
        deps = collect_dependencies_from_source(source, {"helper"})
        assert deps == {"helper"}

    def test_multiple_primitives_in_one_function(self):
        """Multiple distinct primitive calls are all captured."""
        source = """
async def main():
    contacts = await primitives.contacts.ask("list all")
    await primitives.tasks.update("create task")
    handle = await primitives.actor.act("subtask")
    return contacts
"""
        deps = collect_dependencies_from_source(
            source,
            set(),
            environment_namespaces=frozenset({"primitives"}),
        )
        assert deps == {
            "primitives.contacts.ask",
            "primitives.tasks.update",
            "primitives.actor.act",
        }

    def test_non_awaited_environment_call(self):
        """Environment calls without await are also captured (sync or handle-only)."""
        source = """
def main():
    primitives.contacts.ask("query")
"""
        deps = collect_dependencies_from_source(
            source,
            set(),
            environment_namespaces=frozenset({"primitives"}),
        )
        assert "primitives.contacts.ask" in deps


class TestDetectThirdPartyImports:
    """Tests for detect_third_party_imports / detect_third_party_imports_from_source."""

    def test_no_imports(self):
        source = """
def compute(a, b):
    return a + b
"""
        assert detect_third_party_imports_from_source(source) == set()

    def test_stdlib_only(self):
        source = """
def do_stuff():
    import os
    import json
    from pathlib import Path
    return os.getcwd()
"""
        assert detect_third_party_imports_from_source(source) == set()

    def test_third_party_import(self):
        source = """
def process():
    import pandas
    return pandas.DataFrame()
"""
        result = detect_third_party_imports_from_source(source)
        assert result == {"pandas"}

    def test_third_party_from_import(self):
        source = """
def upload():
    from google.cloud import storage
    client = storage.Client()
    return client
"""
        result = detect_third_party_imports_from_source(source)
        assert result == {"google"}

    def test_mixed_imports(self):
        source = """
def analyze():
    import json
    import os
    import numpy as np
    from scipy.stats import norm
    data = json.loads('{}')
    return np.array([norm.rvs()])
"""
        result = detect_third_party_imports_from_source(source)
        assert result == {"numpy", "scipy"}

    def test_environment_module_excluded(self):
        source = """
async def my_func():
    import pydantic
    from pydantic import BaseModel
    class M(BaseModel):
        x: int
    return M(x=1)
"""
        result = detect_third_party_imports_from_source(
            source,
            environment_modules=frozenset({"pydantic", "primitives"}),
        )
        assert result == set()

    def test_primitives_namespace_excluded(self):
        """The 'primitives' namespace is accessed as a name, not imported,
        but if someone writes 'import primitives' it should still be excluded."""
        source = """
async def my_func():
    import primitives
    return await primitives.contacts.ask("hi")
"""
        result = detect_third_party_imports_from_source(
            source,
            environment_modules=frozenset({"primitives"}),
        )
        assert result == set()

    def test_multiple_third_party(self):
        source = """
async def pipeline():
    import requests
    import boto3
    from google.cloud import bigquery
    resp = requests.get("https://example.com")
    return resp.text
"""
        result = detect_third_party_imports_from_source(source)
        assert result == {"requests", "boto3", "google"}

    def test_nested_import_in_conditional(self):
        source = """
def conditional_import():
    if True:
        import pandas
    return None
"""
        result = detect_third_party_imports_from_source(source)
        assert result == {"pandas"}

    def test_async_function(self):
        source = """
async def fetch_data():
    import aiohttp
    async with aiohttp.ClientSession() as session:
        return await session.get("https://example.com")
"""
        result = detect_third_party_imports_from_source(source)
        assert result == {"aiohttp"}

    def test_invalid_source_returns_empty(self):
        assert detect_third_party_imports_from_source("not valid python {{") == set()

    def test_non_function_source_returns_empty(self):
        assert detect_third_party_imports_from_source("x = 1") == set()
