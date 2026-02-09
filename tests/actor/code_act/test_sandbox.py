import pytest
from pydantic import TypeAdapter

from unity.actor.code_act_actor import CodeActActor
from unity.actor.execution import (
    PythonExecutionSession,
    ExecutionResult,
    parts_to_text,
    parts_to_llm_content,
    TextPart,
    ImagePart,
    OutputPart,
)
from unity.common._async_tool.formatting import (
    serialize_tool_content,
    FormattedToolResult,
)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_stateful_variable_execution():
    """Tests that the sandbox maintains simple variable state between calls."""
    sandbox = PythonExecutionSession()

    result1 = await sandbox.execute("x = 100")
    assert result1["error"] is None
    assert "x" in sandbox.global_state
    assert sandbox.global_state["x"] == 100

    result2 = await sandbox.execute("y = x * 2\nprint(y)")
    assert result2["error"] is None
    assert "y" in sandbox.global_state
    assert sandbox.global_state["y"] == 200
    assert parts_to_text(result2["stdout"]) == "200\n"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_stateful_import_execution():
    """Tests that the sandbox maintains imported modules between calls."""
    sandbox = PythonExecutionSession()

    result1 = await sandbox.execute("import json")
    assert result1["error"] is None
    assert "json" in sandbox.global_state

    result2 = await sandbox.execute(
        "my_dict = {'key': 'value'}\nprint(json.dumps(my_dict))",
    )
    assert result2["error"] is None
    assert parts_to_text(result2["stdout"]).strip() == '{"key": "value"}'


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_stateful_function_definition():
    """Tests that the sandbox maintains function definitions between calls."""
    sandbox = PythonExecutionSession()

    func_def_code = "def my_adder(a, b):\n    return a + b"
    result1 = await sandbox.execute(func_def_code)
    assert result1["error"] is None
    assert "my_adder" in sandbox.global_state

    func_call_code = "result = my_adder(10, 5)\nprint(result)"
    result2 = await sandbox.execute(func_call_code)
    assert result2["error"] is None
    assert parts_to_text(result2["stdout"]).strip() == "15"
    assert sandbox.global_state["result"] == 15


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_stateful_class_definition():
    """Tests that the sandbox maintains class definitions between calls."""
    sandbox = PythonExecutionSession()

    class_def_code = (
        "class Greeter:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
        "    def greet(self):\n"
        "        return f'Hello, {self.name}!'\n"
    )
    result1 = await sandbox.execute(class_def_code)
    assert result1["error"] is None
    assert "Greeter" in sandbox.global_state

    class_use_code = "g = Greeter('World')\nprint(g.greet())"
    result2 = await sandbox.execute(class_use_code)
    assert result2["error"] is None
    assert parts_to_text(result2["stdout"]).strip() == "Hello, World!"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_computer_tool_execution(mock_computer_primitives):
    """
    Tests that the sandbox can execute code that calls computer tools via
    the injected computer_primitives.
    """
    sandbox = PythonExecutionSession(computer_primitives=mock_computer_primitives)

    nav_code = "await computer_primitives.navigate('https://example.com')"
    nav_result = await sandbox.execute(nav_code)
    assert nav_result["error"] is None
    mock_computer_primitives.navigate.assert_awaited_once_with("https://example.com")

    act_code = "await computer_primitives.act('Click login button')"
    act_result = await sandbox.execute(act_code)
    assert act_result["error"] is None
    mock_computer_primitives.act.assert_awaited_once_with("Click login button")

    observe_code = """
from pydantic import BaseModel

class MyData(BaseModel):
    data: str

result = await computer_primitives.observe('get data', response_format=MyData)
print(result['data'])
"""
    observe_result = await sandbox.execute(observe_code)
    assert observe_result["error"] is None
    assert parts_to_text(observe_result["stdout"]).strip() == "observed_data"
    mock_computer_primitives.observe.assert_awaited_once()
    assert mock_computer_primitives.observe.call_args[0][0] == "get data"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_error_handling():
    """Tests that the sandbox correctly captures and reports exceptions."""
    sandbox = PythonExecutionSession()
    result = await sandbox.execute("x = 1 / 0")

    assert parts_to_text(result["stdout"]) == ""
    assert parts_to_text(result["stderr"]) == ""
    assert result["result"] is None
    assert result["error"] is not None
    assert "ZeroDivisionError" in result["error"]


# ---------------------------------------------------------------------------
# Tests for Pydantic OutputPart types and order-preserving serialization
# ---------------------------------------------------------------------------


def test_output_part_pydantic_discriminated_union():
    """Tests that OutputPart is a proper Pydantic discriminated union."""
    # TextPart
    text = TextPart(text="hello")
    assert text.type == "text"
    assert text.text == "hello"

    # ImagePart
    image = ImagePart(data="base64data", mime="image/png")
    assert image.type == "image"
    assert image.data == "base64data"
    assert image.mime == "image/png"

    # Discriminated union parsing via TypeAdapter
    adapter = TypeAdapter(OutputPart)

    parsed_text = adapter.validate_python({"type": "text", "text": "world"})
    assert isinstance(parsed_text, TextPart)
    assert parsed_text.text == "world"

    parsed_image = adapter.validate_python(
        {"type": "image", "data": "abc123", "mime": "image/jpeg"},
    )
    assert isinstance(parsed_image, ImagePart)
    assert parsed_image.data == "abc123"


def test_parts_to_llm_content_preserves_order():
    """Tests that parts_to_llm_content preserves text/image interleaving."""
    parts = [
        TextPart(text="before image\n"),
        ImagePart(data="img1_base64", mime="image/png"),
        TextPart(text="between images\n"),
        ImagePart(data="img2_base64", mime="image/jpeg"),
        TextPart(text="after images\n"),
    ]

    llm_content = parts_to_llm_content(parts)

    # Should produce 5 blocks in exact order
    assert len(llm_content) == 5

    assert llm_content[0] == {"type": "text", "text": "before image\n"}
    assert llm_content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,img1_base64"},
    }
    assert llm_content[2] == {"type": "text", "text": "between images\n"}
    assert llm_content[3] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,img2_base64"},
    }
    assert llm_content[4] == {"type": "text", "text": "after images\n"}


def test_parts_to_llm_content_merges_adjacent_text():
    """Tests that adjacent TextParts are merged into a single text block."""
    parts = [
        TextPart(text="line1\n"),
        TextPart(text="line2\n"),
        ImagePart(data="img_base64", mime="image/png"),
        TextPart(text="line3\n"),
        TextPart(text="line4\n"),
    ]

    llm_content = parts_to_llm_content(parts)

    # Adjacent text parts should be merged: 2 text blocks + 1 image
    assert len(llm_content) == 3

    assert llm_content[0] == {"type": "text", "text": "line1\nline2\n"}
    assert llm_content[1]["type"] == "image_url"
    assert llm_content[2] == {"type": "text", "text": "line3\nline4\n"}


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_display_produces_image_parts():
    """Tests that display() produces ImagePart for PIL images."""
    sandbox = PythonExecutionSession()

    code = """
from PIL import Image
img = Image.new("RGB", (10, 10), color="red")
print("before")
display(img)
print("after")
"""
    result = await sandbox.execute(code)
    assert result["error"] is None

    stdout = result["stdout"]

    # Should have TextPart, ImagePart, TextPart in order
    assert len(stdout) == 3
    assert isinstance(stdout[0], TextPart)
    assert isinstance(stdout[1], ImagePart)
    assert isinstance(stdout[2], TextPart)

    # Verify text content
    assert "before" in stdout[0].text
    assert "after" in stdout[2].text

    # Verify image part has valid base64 data
    assert stdout[1].mime == "image/png"
    assert len(stdout[1].data) > 0

    # Verify LLM content preserves order
    llm_content = parts_to_llm_content(stdout)
    assert llm_content[0]["type"] == "text"
    assert llm_content[1]["type"] == "image_url"
    assert llm_content[2]["type"] == "text"


def test_execution_result_implements_formatted_tool_result():
    """Tests that ExecutionResult implements the FormattedToolResult protocol."""
    # ExecutionResult should be recognized as a FormattedToolResult
    result = ExecutionResult(
        stdout=[TextPart(text="hello\n")],
        stderr=[],
        result=42,
        error=None,
    )

    # Protocol check
    assert isinstance(result, FormattedToolResult)

    # to_llm_content should return list of dicts
    llm_content = result.to_llm_content()
    assert isinstance(llm_content, list)
    assert all(isinstance(block, dict) for block in llm_content)
    assert all("type" in block for block in llm_content)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_serialize_tool_content_delegates_to_execution_result():
    """Tests that serialize_tool_content delegates to ExecutionResult.to_llm_content().

    This verifies the protocol-based design: the tool result (ExecutionResult)
    controls its own formatting, not serialize_tool_content.
    """
    sandbox = PythonExecutionSession()

    code = """
from PIL import Image
img = Image.new("RGB", (10, 10), color="blue")
print("text before image")
display(img)
print("text after image")
"""
    result_dict = await sandbox.execute(code)
    assert result_dict["error"] is None

    # Wrap in ExecutionResult (simulating what execute_code tool does)
    result = ExecutionResult(**result_dict)

    # Verify result is an ExecutionResult implementing the protocol
    assert isinstance(result, ExecutionResult)
    assert isinstance(result, FormattedToolResult)

    # Pass through serialize_tool_content (simulating what happens in tool loop)
    serialized = serialize_tool_content(
        tool_name="execute_code",
        payload=result,
        is_final=True,
    )

    # Should be a list of content blocks
    assert isinstance(serialized, list)

    # Extract block types
    block_types = [b.get("type") for b in serialized]

    # Find image and text indices
    image_indices = [i for i, t in enumerate(block_types) if t == "image_url"]
    text_indices = [i for i, t in enumerate(block_types) if t == "text"]

    # There should be at least one image
    assert len(image_indices) >= 1, "Expected at least one image_url block"

    # Critical: image should NOT be at the very end if there's text after it
    # The last text block should come AFTER the image (preserving print order)
    last_image_idx = max(image_indices)
    last_text_idx = max(text_indices)

    assert last_text_idx > last_image_idx, (
        f"Image at index {last_image_idx} should not be after all text. "
        f"Text indices: {text_indices}, Image indices: {image_indices}. "
        "This indicates images are being collected at the end instead of "
        "preserving their original interleaved positions."
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_execute_code_surfaces_computer_state_when_computer_used():
    """
    When sandbox execution invokes computer primitives, the tool result should
    include the current computer state so the model can reason about the page
    without requiring an additional perception step.
    """
    actor = CodeActActor(headless=True, computer_mode="mock", timeout=30)
    try:
        tools = actor._build_tools()
        execute_code = tools["execute_code"]

        res = await execute_code(
            thought="Navigate so computer tools are exercised",
            language="python",
            state_mode="stateful",
            code="await computer_primitives.navigate('https://example.com')",
        )

        assert isinstance(res, ExecutionResult)
        assert res.computer_used is True

        llm_content = serialize_tool_content(
            tool_name="execute_code",
            payload=res,
            is_final=True,
        )
        assert isinstance(llm_content, list)

        text_blocks = [
            b for b in llm_content if isinstance(b, dict) and b.get("type") == "text"
        ]
        image_blocks = [
            b
            for b in llm_content
            if isinstance(b, dict) and b.get("type") == "image_url"
        ]

        assert text_blocks, "Expected at least one text block with metadata"
        meta_text = str(text_blocks[0].get("text") or "")
        assert "computer_state" in meta_text
        assert "url" in meta_text
        assert '"screenshot"' not in meta_text
        assert "data:image" not in meta_text
        assert ";base64," not in meta_text

        assert image_blocks, "Expected a computer screenshot image_url block"
        assert all(
            isinstance(b.get("image_url"), dict)
            and isinstance(b["image_url"].get("url"), str)
            for b in image_blocks
        )
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_builtin_open_available(tmp_path):
    """Python builtins like open() and FileNotFoundError must be available.

    The sandbox restricts the global namespace, but standard builtins that
    are essential for basic file I/O should not be stripped.
    """
    sandbox = PythonExecutionSession()

    # Write a small file for the test to read.
    test_file = tmp_path / "hello.txt"
    test_file.write_text("hello world")

    code = f'with open("{test_file}", "r") as f:\n    print(f.read())'
    result = await sandbox.execute(code)
    assert result["error"] is None, (
        f"open() should be available in the sandbox but got: {result['error']}"
    )
    assert "hello world" in parts_to_text(result["stdout"])
