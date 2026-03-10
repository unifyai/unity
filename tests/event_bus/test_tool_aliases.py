"""Unit tests for tool display alias propagation.

All tests are synchronous, zero-IO, and test the data-flow from ToolSpec
through normalise_tools, methods_to_tool_dict, LoopConfig, and
ToolLoopPayload without touching any EventBus, Pub/Sub, or project setup.
"""

from unity.common.tool_spec import ToolSpec, normalise_tools
from unity.common.llm_helpers import methods_to_tool_dict
from unity.common._async_tool.loop_config import LoopConfig
from unity.events.types.tool_loop import ToolLoopPayload

# ============================================================================
#  ToolSpec.display_label basics
# ============================================================================


def test_toolspec_display_label_defaults_to_none():
    spec = ToolSpec(fn=lambda: None)
    assert spec.display_label is None


def test_toolspec_display_label_stored():
    spec = ToolSpec(fn=lambda: None, display_label="Running code")
    assert spec.display_label == "Running code"


def test_toolspec_display_label_callable():
    label_fn = lambda tc: tc.get("function", {}).get("name", "?")
    spec = ToolSpec(fn=lambda: None, display_label=label_fn)
    assert callable(spec.display_label)


# ============================================================================
#  normalise_tools preserves display_label
# ============================================================================


def test_normalise_tools_preserves_display_label():
    fn = lambda: None
    raw = {
        "my_tool": ToolSpec(fn=fn, display_label="Doing something"),
    }
    result = normalise_tools(raw)
    assert result["my_tool"].display_label == "Doing something"


def test_normalise_tools_bare_callable_has_none_label():
    fn = lambda: None
    result = normalise_tools({"plain": fn})
    assert result["plain"].display_label is None


# ============================================================================
#  methods_to_tool_dict propagates display_label
# ============================================================================


class _FakeManager:
    class __class__:
        __name__ = "FakeManager"
        __mro__ = (object,)

    def some_tool(self):
        """A tool."""


def test_methods_to_tool_dict_propagates_display_label():
    mgr = _FakeManager()
    result = methods_to_tool_dict(
        ToolSpec(fn=mgr.some_tool, display_label="Doing fake things"),
        include_class_name=False,
    )
    spec = result["some_tool"]
    assert isinstance(spec, ToolSpec)
    assert spec.display_label == "Doing fake things"


def test_methods_to_tool_dict_bare_method_has_no_label():
    mgr = _FakeManager()
    result = methods_to_tool_dict(
        mgr.some_tool,
        include_class_name=False,
    )
    assert not isinstance(result["some_tool"], ToolSpec)


def test_methods_to_tool_dict_preserves_all_spec_fields():
    mgr = _FakeManager()
    result = methods_to_tool_dict(
        ToolSpec(
            fn=mgr.some_tool,
            max_concurrent=3,
            max_total_calls=10,
            read_only=True,
            manager_tool=True,
            display_label="Testing all fields",
        ),
        include_class_name=False,
    )
    spec = result["some_tool"]
    assert isinstance(spec, ToolSpec)
    assert spec.max_concurrent == 3
    assert spec.max_total_calls == 10
    assert spec.read_only is True
    assert spec.manager_tool is True
    assert spec.display_label == "Testing all fields"


# ============================================================================
#  LoopConfig.tool_alias_lookup
# ============================================================================


def test_loop_config_tool_alias_lookup_defaults_to_none():
    cfg = LoopConfig("test_loop", None, [])
    assert cfg.tool_alias_lookup is None


def test_loop_config_tool_alias_lookup_setter():
    cfg = LoopConfig("test_loop", None, [])
    cfg.tool_alias_lookup = {"my_tool": "Doing something"}
    assert cfg.tool_alias_lookup == {"my_tool": "Doing something"}


def test_loop_config_tool_alias_lookup_set_to_none():
    cfg = LoopConfig("test_loop", None, [])
    cfg.tool_alias_lookup = {"x": "y"}
    cfg.tool_alias_lookup = None
    assert cfg.tool_alias_lookup is None


# ============================================================================
#  ToolLoopPayload.tool_aliases
# ============================================================================


def test_payload_tool_aliases_defaults_to_none():
    payload = ToolLoopPayload(
        message={"role": "user", "content": "hi"},
        method="test",
    )
    assert payload.tool_aliases is None


def test_payload_tool_aliases_serializes():
    payload = ToolLoopPayload(
        message={"role": "assistant", "content": None},
        method="test",
        tool_aliases={"my_tool": "Doing something"},
    )
    data = payload.model_dump()
    assert data["tool_aliases"] == {"my_tool": "Doing something"}


def test_payload_tool_aliases_null_when_none():
    payload = ToolLoopPayload(
        message={"role": "assistant", "content": None},
        method="test",
        tool_aliases=None,
    )
    data = payload.model_dump()
    assert data["tool_aliases"] is None


# ============================================================================
#  Sparse alias dict building (mirrors event_bus_util logic)
# ============================================================================


def _build_sparse_aliases(lookup, message):
    """Replicate the sparse alias logic from event_bus_util."""
    if not lookup or not isinstance(message, dict):
        return None
    tool_calls = message.get("tool_calls") or []
    sparse = {}
    for tc in tool_calls:
        name = (tc.get("function") or {}).get("name", "")
        if name in lookup:
            val = lookup[name]
            sparse[name] = val(tc) if callable(val) else val
    return sparse or None


def test_sparse_aliases_single_tool_call():
    lookup = {"tool_a": "Doing A", "tool_b": "Doing B", "tool_c": "Doing C"}
    message = {
        "role": "assistant",
        "tool_calls": [{"function": {"name": "tool_a"}, "id": "call_1"}],
    }
    assert _build_sparse_aliases(lookup, message) == {"tool_a": "Doing A"}


def test_sparse_aliases_multiple_tool_calls():
    lookup = {"tool_a": "Doing A", "tool_b": "Doing B", "tool_c": "Doing C"}
    message = {
        "role": "assistant",
        "tool_calls": [
            {"function": {"name": "tool_a"}, "id": "call_1"},
            {"function": {"name": "tool_c"}, "id": "call_2"},
        ],
    }
    assert _build_sparse_aliases(lookup, message) == {
        "tool_a": "Doing A",
        "tool_c": "Doing C",
    }


def test_sparse_aliases_tool_without_alias():
    lookup = {"tool_a": "Doing A"}
    message = {
        "role": "assistant",
        "tool_calls": [{"function": {"name": "tool_b"}, "id": "call_1"}],
    }
    assert _build_sparse_aliases(lookup, message) is None


def test_sparse_aliases_no_tool_calls():
    assert (
        _build_sparse_aliases(
            {"tool_a": "Doing A"},
            {"role": "user", "content": "hello"},
        )
        is None
    )


def test_sparse_aliases_tool_result_message():
    assert (
        _build_sparse_aliases(
            {"tool_a": "Doing A"},
            {"role": "tool", "name": "tool_a", "content": "result"},
        )
        is None
    )


def test_sparse_aliases_empty_lookup():
    message = {
        "role": "assistant",
        "tool_calls": [{"function": {"name": "x"}}],
    }
    assert _build_sparse_aliases({}, message) is None


def test_sparse_aliases_none_lookup():
    message = {
        "role": "assistant",
        "tool_calls": [{"function": {"name": "x"}}],
    }
    assert _build_sparse_aliases(None, message) is None


def test_sparse_aliases_callable_label():
    import json

    def dynamic_label(tc):
        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
        return args.get("function_name", "unknown")

    lookup = {"execute_function": dynamic_label}
    message = {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "execute_function",
                    "arguments": json.dumps({"function_name": "primitives.web.ask"}),
                },
                "id": "call_1",
            },
        ],
    }
    assert _build_sparse_aliases(lookup, message) == {
        "execute_function": "primitives.web.ask",
    }


def test_sparse_aliases_mixed_known_and_unknown():
    lookup = {"tool_a": "Doing A", "tool_c": "Doing C"}
    message = {
        "role": "assistant",
        "tool_calls": [
            {"function": {"name": "tool_a"}, "id": "call_1"},
            {"function": {"name": "tool_b"}, "id": "call_2"},
            {"function": {"name": "tool_c"}, "id": "call_3"},
        ],
    }
    assert _build_sparse_aliases(lookup, message) == {
        "tool_a": "Doing A",
        "tool_c": "Doing C",
    }


# ============================================================================
#  End-to-end: ToolSpec -> normalise_tools -> alias lookup -> sparse dict
# ============================================================================


def test_end_to_end_alias_flow():
    fn_a = lambda: None
    fn_b = lambda: None
    fn_c = lambda: None

    raw_tools = {
        "tool_a": ToolSpec(fn=fn_a, display_label="Doing A"),
        "tool_b": fn_b,
        "tool_c": ToolSpec(fn=fn_c, display_label="Doing C"),
    }
    normalized = normalise_tools(raw_tools)

    lookup = {
        name: spec.display_label
        for name, spec in normalized.items()
        if spec.display_label
    }
    assert lookup == {"tool_a": "Doing A", "tool_c": "Doing C"}

    message_with_a = {
        "role": "assistant",
        "tool_calls": [{"function": {"name": "tool_a"}, "id": "call_1"}],
    }
    sparse = _build_sparse_aliases(lookup, message_with_a)
    assert sparse == {"tool_a": "Doing A"}

    message_with_b = {
        "role": "assistant",
        "tool_calls": [{"function": {"name": "tool_b"}, "id": "call_2"}],
    }
    assert _build_sparse_aliases(lookup, message_with_b) is None

    payload = ToolLoopPayload(
        message=message_with_a,
        method="TestLoop",
        tool_aliases=sparse,
    )
    assert payload.tool_aliases == {"tool_a": "Doing A"}
