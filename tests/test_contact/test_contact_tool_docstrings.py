from __future__ import annotations


from unity.contact_manager.contact_manager import ContactManager


def _unwrap_callable(tool):
    """Return the underlying callable from either a ToolSpec or a function."""
    return getattr(tool, "fn", tool)


def test_all_ask_tools_have_sufficient_docstrings():
    cm = ContactManager()
    tools = cm.get_tools("ask")

    assert tools, "ContactManager.ask should expose at least one tool"

    for name, value in tools.items():
        fn = _unwrap_callable(value)
        doc = (getattr(fn, "__doc__", None) or "").strip()
        assert doc, f"Tool '{name}' is missing a docstring"
        assert (
            len(doc) >= 100
        ), f"Docstring for tool '{name}' is too short (len={len(doc)})"


def test_all_update_tools_have_sufficient_docstrings():
    cm = ContactManager()
    tools = cm.get_tools("update")

    assert tools, "ContactManager.update should expose at least one tool"

    for name, value in tools.items():
        fn = _unwrap_callable(value)
        doc = (getattr(fn, "__doc__", None) or "").strip()
        assert doc, f"Tool '{name}' is missing a docstring"
        assert (
            len(doc) >= 100
        ), f"Docstring for tool '{name}' is too short (len={len(doc)})"
