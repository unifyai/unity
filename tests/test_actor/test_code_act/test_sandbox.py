import pytest

from unity.actor.code_act_actor import CodeExecutionSandbox


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_stateful_variable_execution():
    """Tests that the sandbox maintains simple variable state between calls."""
    sandbox = CodeExecutionSandbox()

    result1 = await sandbox.execute("x = 100")
    assert result1["error"] is None
    assert "x" in sandbox.global_state
    assert sandbox.global_state["x"] == 100

    result2 = await sandbox.execute("y = x * 2\nprint(y)")
    assert result2["error"] is None
    assert "y" in sandbox.global_state
    assert sandbox.global_state["y"] == 200
    assert result2["stdout"] == "200\n"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_stateful_import_execution():
    """Tests that the sandbox maintains imported modules between calls."""
    sandbox = CodeExecutionSandbox()

    result1 = await sandbox.execute("import json")
    assert result1["error"] is None
    assert "json" in sandbox.global_state

    result2 = await sandbox.execute(
        "my_dict = {'key': 'value'}\nprint(json.dumps(my_dict))",
    )
    assert result2["error"] is None
    assert result2["stdout"].strip() == '{"key": "value"}'


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_stateful_function_definition():
    """Tests that the sandbox maintains function definitions between calls."""
    sandbox = CodeExecutionSandbox()

    func_def_code = "def my_adder(a, b):\n    return a + b"
    result1 = await sandbox.execute(func_def_code)
    assert result1["error"] is None
    assert "my_adder" in sandbox.global_state

    func_call_code = "result = my_adder(10, 5)\nprint(result)"
    result2 = await sandbox.execute(func_call_code)
    assert result2["error"] is None
    assert result2["stdout"].strip() == "15"
    assert sandbox.global_state["result"] == 15


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_stateful_class_definition():
    """Tests that the sandbox maintains class definitions between calls."""
    sandbox = CodeExecutionSandbox()

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
    assert result2["stdout"].strip() == "Hello, World!"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_browser_tool_execution(mock_computer_primitives):
    """
    Tests that the sandbox can execute code that calls browser tools via
    the injected computer_primitives.
    """
    sandbox = CodeExecutionSandbox(computer_primitives=mock_computer_primitives)

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
    assert observe_result["stdout"].strip() == "observed_data"
    mock_computer_primitives.observe.assert_awaited_once()
    assert mock_computer_primitives.observe.call_args[0][0] == "get data"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_sandbox_error_handling():
    """Tests that the sandbox correctly captures and reports exceptions."""
    sandbox = CodeExecutionSandbox()
    result = await sandbox.execute("x = 1 / 0")

    assert result["stdout"] == ""
    assert result["stderr"] == ""
    assert result["result"] is None
    assert result["error"] is not None
    assert "ZeroDivisionError" in result["error"]
