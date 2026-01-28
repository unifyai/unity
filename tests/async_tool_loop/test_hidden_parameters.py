# tests/async_tool_loop/test_hidden_parameters.py
from __future__ import annotations

import asyncio
import pytest

import unity.common.llm_helpers as llmh


# --------------------------------------------------------------------------- #
# Shared assertion helper                                                     #
# --------------------------------------------------------------------------- #
def _assert_internal_queues_hidden(fn) -> None:
    """
    Verify that *_clarification_up_q*, *_clarification_down_q*, and *_notification_up_q* never show up
    in the generated schema nor in the docstring description that is forwarded
    to the LLM.
    """
    schema = llmh.method_to_schema(fn)
    props = schema["function"]["parameters"]["properties"]
    required = schema["function"]["parameters"]["required"]
    desc = schema["function"]["description"]

    for name in ("_clarification_up_q", "_clarification_down_q", "_notification_up_q"):
        # ––– must be absent from JSON schema –––
        assert name not in props
        assert name not in required
        # ––– and also stripped from the docstring forwarded to the model –––
        assert name not in desc


# --------------------------------------------------------------------------- #
# 1. Original “Google style” example (colon after each name)                  #
# --------------------------------------------------------------------------- #
def tool_google_style(
    a: int,
    _clarification_up_q: asyncio.Queue[str],
    _clarification_down_q: asyncio.Queue[str],
    _notification_up_q: asyncio.Queue | None = None,
) -> int:
    """
    Google-style.

    Args:
        a: Some value.
        _clarification_up_q: internal.
        _clarification_down_q: internal.
        _notification_up_q: internal.
    """
    return a


# --------------------------------------------------------------------------- #
# 2. New *slash-separated* NumPy variant: “name_a / name_b : type”            #
# --------------------------------------------------------------------------- #
def tool_numpy_slash(
    a: int,
    _clarification_up_q: asyncio.Queue[str] | None = None,
    _clarification_down_q: asyncio.Queue[str] | None = None,
    _notification_up_q: asyncio.Queue | None = None,
) -> int:
    """
    NumPy style — slash-separated synonyms.

    Parameters
    ----------
    a : int
        Some value.
    _clarification_up_q / _clarification_down_q / _notification_up_q : asyncio.Queue | None
        Internal queues.
    """
    return a


# --------------------------------------------------------------------------- #
# 3. New *comma-separated* variant: “name_a, name_b, …” (no type, no colon)   #
# --------------------------------------------------------------------------- #
def tool_numpy_commas(
    a: int,
    _clarification_up_q: asyncio.Queue[str] | None = None,
    _clarification_down_q: asyncio.Queue[str] | None = None,
    _notification_up_q: asyncio.Queue | None = None,
) -> int:
    """
    NumPy style — comma-separated, no type / no colon.

    Parameters
    ----------
    a : int
        Some value.

    _clarification_up_q, _clarification_down_q, _notification_up_q
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


# --------------------------------------------------------------------------- #
# 4. Test that parent_chat_context_cont is hidden from steering methods       #
# --------------------------------------------------------------------------- #
def test_parent_chat_context_cont_hidden_from_steering_methods() -> None:
    """
    The parent_chat_context_cont parameter (plumbing for context propagation)
    must be hidden from LLM tool schemas because it's automatically injected
    by the orchestrating code layer, not supplied by the LLM.

    This parameter is hidden via an explicit list in method_to_schema, not by
    underscore prefix convention.
    """
    from typing import Optional
    from unity.image_manager.types.image_refs import ImageRefs

    # Simulate a steering method with parent_chat_context_cont
    async def interject(
        message: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
        images: Optional[ImageRefs] = None,
    ) -> str:
        """Provide additional information or instructions to the running task.

        Parameters
        ----------
        message : str
            The user interjection to inject into the loop.
        images : ImageRefs | None, optional
            Live image references to make available during this interjection.
        """
        return "ok"

    schema = llmh.method_to_schema(interject)
    props = schema["function"]["parameters"]["properties"]
    required = schema["function"]["parameters"]["required"]
    desc = schema["function"]["description"]

    # parent_chat_context_cont MUST be hidden (via explicit list)
    assert "parent_chat_context_cont" not in props
    assert "parent_chat_context_cont" not in required
    assert "parent_chat_context_cont" not in desc

    # message and images SHOULD be visible
    assert "message" in props
    assert "message" in required
    assert "images" in props


def test_steering_method_images_type_is_schema_safe() -> None:
    """
    The images parameter should use Optional[ImageRefs] which produces a valid
    strict JSON schema, not list | dict | None which fails strict validation.
    """
    from typing import Optional
    from unity.image_manager.types.image_refs import ImageRefs

    async def interject(
        message: str,
        *,
        images: Optional[ImageRefs] = None,
    ) -> str:
        """Interject with images."""
        return "ok"

    schema = llmh.method_to_schema(interject)
    props = schema["function"]["parameters"]["properties"]

    # The images property should exist and have a valid schema
    assert "images" in props
    images_schema = props["images"]

    # Should NOT be a bare {"type": "array"} without items
    # Should NOT be an anyOf union (which strict mode rejects)
    # ImageRefs produces a proper JSON schema with $defs and items
    assert images_schema.get("type") != "array" or "items" in images_schema
    assert "anyOf" not in images_schema


# --------------------------------------------------------------------------- #
# 5. Test DynamicToolFactory adopts handle signatures without hardcoding      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_dynamic_factory_adopts_custom_interject_args() -> None:
    """
    Verify that DynamicToolFactory._create_interject_tool dynamically adopts
    the handle's interject signature, including any custom parameters the
    handle defines, rather than hardcoding a fixed set of params.
    """
    from contextlib import suppress
    from typing import Optional

    from unity.common.async_tool_loop import SteerableToolHandle
    from unity.common._async_tool.tools_data import ToolsData
    from unity.common._async_tool.tools_utils import ToolCallMetadata
    from unity.common._async_tool.dynamic_tools_factory import DynamicToolFactory
    from unity.image_manager.types.image_refs import ImageRefs

    class CustomInterjectHandle(SteerableToolHandle):
        """Handle with custom interject parameters."""

        def __init__(self) -> None:
            pass

        async def interject(
            self,
            message: str,
            *,
            priority: int = 1,
            category: str = "general",
            parent_chat_context_cont: list[dict] | None = None,
            images: Optional[ImageRefs] = None,
        ) -> str:
            """Custom interject with priority and category.

            Parameters
            ----------
            message : str
                The interjection message.
            priority : int
                Priority level (1-5).
            category : str
                Category for routing.
            images : ImageRefs | None
                Optional images.
            """
            return "ok"

        async def ask(self, q: str, **kw):
            return self

        def stop(self, r=None, **kw):
            return "stopped"

        async def pause(self):
            return "paused"

        async def resume(self):
            return "resumed"

        def done(self):
            return True

        async def result(self):
            return "ok"

        async def next_clarification(self):
            return {}

        async def next_notification(self):
            return {}

        async def answer_clarification(self, cid, ans):
            pass

    # Setup minimal environment
    class _DummyLogger:
        log_steps = False

        def info(self, *a, **kw): ...
        def error(self, *a, **kw): ...

    class _DummyClient:
        def __init__(self):
            self.messages = []

    tools_data = ToolsData({}, client=_DummyClient(), logger=_DummyLogger())
    pending_task = asyncio.create_task(asyncio.sleep(10))

    meta = ToolCallMetadata(
        name="custom_tool",
        call_id="call_456",
        call_dict={"id": "call_456", "function": {"name": "custom", "arguments": "{}"}},
        call_idx=0,
        chat_context=None,
        assistant_msg={},
        is_interjectable=True,
        tool_schema={},
        llm_arguments={},
        raw_arguments_json="{}",
        handle=CustomInterjectHandle(),
        interject_queue=None,
        clar_up_queue=None,
        clar_down_queue=None,
        notification_queue=None,
        pause_event=None,
    )
    tools_data.save_task(pending_task, meta)

    # Generate tools
    factory = DynamicToolFactory(tools_data)
    factory.generate()

    # Find the interject tool
    interject_keys = [
        k for k in factory.dynamic_tools.keys() if k.startswith("interject_")
    ]
    assert interject_keys, "Expected interject helper to be generated"

    helper = factory.dynamic_tools[interject_keys[0]]
    schema = llmh.method_to_schema(helper, include_class_name=False)
    props = schema["function"]["parameters"]["properties"]

    # Custom params (priority, category) SHOULD be visible
    assert (
        "priority" in props
    ), f"Expected 'priority' in schema, got: {list(props.keys())}"
    assert (
        "category" in props
    ), f"Expected 'category' in schema, got: {list(props.keys())}"

    # message SHOULD be visible (either from handle or content alias)
    assert "message" in props or "content" in props

    # parent_chat_context_cont MUST be hidden (via explicit list)
    assert "parent_chat_context_cont" not in props

    # images SHOULD be visible
    assert "images" in props

    with suppress(BaseException):
        pending_task.cancel()
        await pending_task


# --------------------------------------------------------------------------- #
# 6. Test parity between context and images across steering methods           #
# --------------------------------------------------------------------------- #
def test_steering_methods_have_parity_for_plumbing_and_images() -> None:
    """
    Verify that ask and interject have both context parameters (hidden plumbing)
    and images (visible) parameters for full parity.

    - ask() uses parent_chat_context (initial context for fresh inspection loop)
    - interject() uses parent_chat_context_cont (continuation for ongoing loop)
    - stop() only has images (stop is symbolic cancellation, no context propagation)
    """
    from unity.common.async_tool_loop import SteerableHandle, SteerableToolHandle
    import inspect

    # Check SteerableHandle.ask - uses _parent_chat_context (not _cont) since it
    # spawns a fresh inspection loop that needs initial context
    # Note: underscore prefix makes it hidden from LLM schema
    ask_sig = inspect.signature(SteerableHandle.ask)
    ask_params = set(ask_sig.parameters.keys()) - {"self"}
    assert "_parent_chat_context" in ask_params, "ask should have _parent_chat_context"
    assert "images" in ask_params, "ask should have images"

    # Check SteerableHandle.interject - uses parent_chat_context_cont since it
    # adds to an ongoing conversation
    interject_sig = inspect.signature(SteerableHandle.interject)
    interject_params = set(interject_sig.parameters.keys()) - {"self"}
    assert (
        "parent_chat_context_cont" in interject_params
    ), "interject should have parent_chat_context_cont"
    assert "images" in interject_params, "interject should have images"

    # Check SteerableToolHandle.stop - only has images (stop is symbolic, no context propagation)
    stop_sig = inspect.signature(SteerableToolHandle.stop)
    stop_params = set(stop_sig.parameters.keys()) - {"self"}
    assert "images" in stop_params, "stop should have images"
    assert (
        "parent_chat_context_cont" not in stop_params
    ), "stop should NOT have parent_chat_context_cont (stop is symbolic cancellation)"


@pytest.mark.asyncio
async def test_dynamic_factory_stop_tool_has_images_from_handle() -> None:
    """
    Verify that the stop tool generated by DynamicToolFactory has the images
    parameter from the handle's native signature (not from post-processing).
    """
    from contextlib import suppress
    from typing import Optional

    from unity.common.async_tool_loop import SteerableToolHandle
    from unity.common._async_tool.tools_data import ToolsData
    from unity.common._async_tool.tools_utils import ToolCallMetadata
    from unity.common._async_tool.dynamic_tools_factory import DynamicToolFactory
    from unity.image_manager.types.image_refs import ImageRefs

    class HandleWithStopImages(SteerableToolHandle):
        """Handle with stop() that includes images natively."""

        def __init__(self) -> None:
            pass

        def stop(
            self,
            reason: Optional[str] = None,
            *,
            parent_chat_context_cont: list[dict] | None = None,
            images: Optional[ImageRefs] = None,
        ) -> str:
            """Stop with images.

            Parameters
            ----------
            reason : str | None
                Reason for stopping.
            images : ImageRefs | None
                Images to attach.
            """
            return "stopped"

        async def ask(self, q: str, **kw):
            return self

        async def interject(self, m: str, **kw):
            return None

        async def pause(self):
            return "paused"

        async def resume(self):
            return "resumed"

        def done(self):
            return True

        async def result(self):
            return "ok"

        async def next_clarification(self):
            return {}

        async def next_notification(self):
            return {}

        async def answer_clarification(self, cid, ans):
            pass

    class _DummyLogger:
        log_steps = False

        def info(self, *a, **kw): ...
        def error(self, *a, **kw): ...

    class _DummyClient:
        def __init__(self):
            self.messages = []

    tools_data = ToolsData({}, client=_DummyClient(), logger=_DummyLogger())
    pending_task = asyncio.create_task(asyncio.sleep(10))

    meta = ToolCallMetadata(
        name="stop_test_tool",
        call_id="call_789",
        call_dict={
            "id": "call_789",
            "function": {"name": "stop_test", "arguments": "{}"},
        },
        call_idx=0,
        chat_context=None,
        assistant_msg={},
        is_interjectable=False,
        tool_schema={},
        llm_arguments={},
        raw_arguments_json="{}",
        handle=HandleWithStopImages(),
        interject_queue=None,
        clar_up_queue=None,
        clar_down_queue=None,
        notification_queue=None,
        pause_event=None,
    )
    tools_data.save_task(pending_task, meta)

    factory = DynamicToolFactory(tools_data)
    factory.generate()

    # Find the stop tool
    stop_keys = [k for k in factory.dynamic_tools.keys() if k.startswith("stop_")]
    assert stop_keys, "Expected stop helper to be generated"

    helper = factory.dynamic_tools[stop_keys[0]]
    schema = llmh.method_to_schema(helper, include_class_name=False)
    props = schema["function"]["parameters"]["properties"]

    # images SHOULD be visible (from handle's native signature, not post-processing)
    assert (
        "images" in props
    ), f"Expected 'images' in stop schema, got: {list(props.keys())}"

    # reason SHOULD be visible
    assert (
        "reason" in props
    ), f"Expected 'reason' in stop schema, got: {list(props.keys())}"

    # parent_chat_context_cont MUST be hidden (via explicit list)
    assert "parent_chat_context_cont" not in props

    with suppress(BaseException):
        pending_task.cancel()
        await pending_task
