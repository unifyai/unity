import inspect
import pytest
import unity.common.llm_helpers as llmh


# Helper: mimic the docstring assignment logic from our updated `_reg_tool`
def apply_doc(fn, fallback_doc):
    existing = inspect.getdoc(fn)
    new_doc = existing.strip() if existing else fallback_doc
    # if this is a bound method, set on its underlying function object
    target = getattr(fn, "__func__", fn)
    target.__doc__ = new_doc
    return fn


def test_custom_method_docstring_extraction():
    class DummyHandle:
        def ask(self):
            """Query elapsed progress."""
            return "progress"

    bound = DummyHandle().ask
    fallback = "Fallback doc."
    helper = apply_doc(bound, fallback)
    # Should pick up the original docstring, not the fallback
    assert inspect.getdoc(helper) == "Query elapsed progress."


def test_no_docstring_uses_fallback():
    def no_doc_fn():
        pass

    fallback = "Cancel pending call foo()."
    helper = apply_doc(no_doc_fn, fallback)
    # No original docstring → uses fallback
    assert helper.__doc__ == fallback


def test_multiline_docstring_stripped():
    class Dummy:
        def multi(self):
            """
            Line one.
            Line two.
            """
            return None

    bound = Dummy().multi
    helper = apply_doc(bound, "Should not fallback")
    # Leading/trailing whitespace stripped, inner lines preserved
    assert inspect.getdoc(helper) == "Line one.\nLine two."


def test_builtin_dynamic_docstring_fallback_for_stop():
    # Simulate a generated stop helper with no original docstring
    async def stop_fn():
        pass

    fallback = "Cancel pending call inner_tool()."
    helper = apply_doc(stop_fn, fallback)
    # The docstring should start with our fallback text
    assert helper.__doc__.startswith("Cancel pending call")


def test_builtin_dynamic_docstring_extracted_if_present():
    # Simulate a generated helper that actually has its own docstring
    async def interject_fn():
        """Inject guidance into the running tool."""

    fallback = "Fallback doc."
    helper = apply_doc(interject_fn, fallback)
    # Should use the real docstring, not the fallback
    assert inspect.getdoc(helper) == "Inject guidance into the running tool."


def test_mixed_scenario_multiple_helpers():
    class Handle:
        def ask(self):
            """Ask current status."""
            return "ok"

        def stop(self):
            pass

    h = Handle()
    bound_ask = h.ask
    bound_stop = h.stop
    ask_helper = apply_doc(bound_ask, "Fallback ask doc.")
    stop_helper = apply_doc(bound_stop, "Fallback stop doc.")

    assert inspect.getdoc(ask_helper) == "Ask current status."
    assert stop_helper.__doc__ == "Fallback stop doc."


def test_mro_fallback_for_bound_method_in_schema():
    """
    Ensure centralized schema doc resolver falls back to an ancestor method's
    docstring when the concrete method lacks one.
    """

    class _Base:
        def status(self) -> None:
            """Base status doc: returns current state."""
            return None

    class _Child(_Base):
        def status(self) -> None:
            # no docstring; should inherit via MRO
            return None

    schema = llmh.method_to_schema(_Child().status)
    desc = schema["function"]["description"]
    assert "Base status doc: returns current state." in desc


if __name__ == "__main__":
    pytest.main()
