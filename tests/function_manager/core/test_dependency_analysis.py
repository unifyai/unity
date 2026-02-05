"""
Tests for dependency_analysis module.
"""

from unity.function_manager.dependency_analysis import (
    collect_dependencies_from_source,
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
