"""Code merge / AST-based patching tests for HierarchicalActor."""

import ast
import textwrap
from unittest.mock import MagicMock

import pytest

from unity.actor.hierarchical_actor import HierarchicalActor, HierarchicalActorHandle

# NOTE: These tests intentionally avoid running the full actor loop; they focus on
# AST-based plan editing helpers used during recovery/reimplementation flows.


class MockHierarchicalActorHandle:
    def __init__(self):
        self.plan_source_code = ""
        self.clean_function_source_map = {}
        self.action_log = []
        self.actor = MagicMock()
        self.actor._sanitize_code = MagicMock(side_effect=lambda code, plan: code)
        self.actor._load_plan_module = MagicMock()


def _update_plan_with_new_code(plan, function_name, new_code):
    """
    (Test version) Updates the plan's source code by surgically replacing a
    function's AST node, preserving all nested structures.
    """

    class FunctionReplacer(ast.NodeTransformer):
        def __init__(self, target_name, new_node):
            self.target_name = target_name
            self.new_node = new_node
            self.replaced = False

        def visit_FunctionDef(self, node):
            if node.name == self.target_name:
                self.replaced = True
                return self.new_node
            return self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            if node.name == self.target_name:
                self.replaced = True
                return self.new_node
            return self.generic_visit(node)

    plan.action_log.append(f"Updating implementation of '{function_name}'.")
    original_tree = ast.parse(plan.plan_source_code or "pass")
    new_function_tree = ast.parse(textwrap.dedent(new_code))
    new_function_node = new_function_tree.body[0]

    replacer = FunctionReplacer(function_name, new_function_node)
    modified_tree = replacer.visit(original_tree)

    if not replacer.replaced:
        modified_tree.body.append(new_function_node)

    ast.fix_missing_locations(modified_tree)
    unsanitized_code = ast.unparse(modified_tree)
    plan.plan_source_code = plan.actor._sanitize_code(unsanitized_code, plan)
    plan.actor._load_plan_module(plan)


COMPLEX_NESTED_PLAN = textwrap.dedent(
    """
# Module-level constant
API_ENDPOINT = "https://api.example.com"

# Module-level helper (not async)
def format_data(data):
    return {"payload": data}

@verify
async def main_orchestrator():
    \"\"\"
    This function contains nested functions and decorators,
    simulating the structure of the Midland Heart skill.
    \"\"\"

    # A nested decorator
    def run_until_success(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            print("Wrapper is running")
            return await fn(*args, **kwargs)
        return wrapper

    # A nested function using the decorator
    @run_until_success
    async def nested_step_one():
        print("Executing nested step one.")
        return True

    # The function we are going to replace (stubbed)
    async def function_to_replace(param1: str):
        \"\"\"This is a stub that will be dynamically implemented.\"\"\"
        raise NotImplementedError("Implement me!")

    # Main logic
    await nested_step_one()
    result = await function_to_replace("test_param")
    return result
""",
)

NEW_IMPLEMENTATION_CODE = textwrap.dedent(
    """
async def function_to_replace(param1: str):
    \"\"\"This is the new, full implementation.\"\"\"
    import json

    print(f"New implementation running with: {param1}")
    class ResultProcessor:
        def process(self, data):
            return json.dumps(data)

    processor = ResultProcessor()
    return processor.process({"status": "success"})
""",
)


def test_ast_merge_replaces_function_without_corrupting_nested_structure():
    plan = MockHierarchicalActorHandle()
    plan.plan_source_code = COMPLEX_NESTED_PLAN

    _update_plan_with_new_code(plan, "function_to_replace", NEW_IMPLEMENTATION_CODE)
    final_code = plan.plan_source_code

    ast.parse(final_code)  # still valid python
    assert "def run_until_success(fn):" in final_code
    assert "@functools.wraps(fn)" in final_code
    assert "@run_until_success" in final_code
    assert "async def nested_step_one():" in final_code
    assert "\n@functools.wraps(fn)" not in final_code
    assert "API_ENDPOINT = 'https://api.example.com'" in final_code
    assert "def format_data(data):" in final_code
    assert "This is the new, full implementation." in final_code
    assert "raise NotImplementedError" not in final_code


# --- Nested replacement tests against HierarchicalActorHandle._update_plan_with_new_code ---

CANNED_PLAN_WITH_NESTING_NESTED_FUNCTION_REPLACEMENT = textwrap.dedent(
    """
    # This is a top-level comment that should be preserved.

    async def top_level_function_one():
        '''This is the first top-level function.'''
        print("Executing top_level_function_one")
        await computer_primitives.act("First action")

    async def parent_function():
        '''This function contains a nested function that will be replaced.'''
        print("Entering parent_function")

        async def nested_function(param: str):
            '''This is the ORIGINAL nested function.'''
            print(f"Original nested_function called with: {param}")
            # This line will be replaced
            await computer_primitives.act(f"Original action: {param}")

        await nested_function("initial_call")
        print("Exiting parent_function")

    async def main_plan():
        '''The main entry point.'''
        await top_level_function_one()
        await parent_function()
        return "Plan finished."
    """,
)

NEW_NESTED_CODE = textwrap.dedent(
    """
    async def nested_function(param: str, new_param: int = 42):
        '''This is the REPLACED nested function with a new signature.'''
        print(f"Replaced nested_function called with: {param} and {new_param}")
        for i in range(new_param):
            await computer_primitives.act(f"New repeated action {i+1}: {param}")
        print("New nested logic finished.")
    """,
).strip()


async def _run_nested_function_replacement_test():
    mock_actor = MagicMock(spec=HierarchicalActor)
    mock_actor._sanitize_code.side_effect = lambda code, plan: code

    active_task = HierarchicalActorHandle(
        actor=mock_actor,
        goal="Test nested replacement",
    )
    if active_task._execution_task:
        active_task._execution_task.cancel()

    initial_sanitized_code = CANNED_PLAN_WITH_NESTING_NESTED_FUNCTION_REPLACEMENT
    active_task.plan_source_code = initial_sanitized_code

    tree = ast.parse(initial_sanitized_code)
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef):
            func_name = node.name
            active_task.top_level_function_names.add(func_name)
            active_task.clean_function_source_map[func_name] = ast.unparse(node)

    assert "Original action" in active_task.clean_function_source_map["parent_function"]

    active_task._update_plan_with_new_code("nested_function", NEW_NESTED_CODE)

    updated_parent_source = active_task.clean_function_source_map.get("parent_function")
    assert updated_parent_source is not None
    assert "Original action" not in updated_parent_source
    assert "New repeated action" in updated_parent_source

    parent_tree = ast.parse(updated_parent_source)
    nested_func_node = None
    for node in ast.walk(parent_tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "nested_function":
            nested_func_node = node
            break
    assert nested_func_node is not None

    args = nested_func_node.args
    param_names = [a.arg for a in args.args]
    assert param_names == ["param", "new_param"]
    assert len(args.defaults) == 1
    assert isinstance(args.defaults[0], ast.Constant) and args.defaults[0].value == 42

    reconstructed_parts = [
        ast.unparse(node)
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for func_name in sorted(list(active_task.top_level_function_names)):
        if func_name in active_task.clean_function_source_map:
            reconstructed_parts.append(active_task.clean_function_source_map[func_name])
    final_source_code = "\n\n".join(reconstructed_parts)

    assert "top_level_function_one" in final_source_code
    assert "main_plan" in final_source_code
    assert "Original action" not in final_source_code
    assert "New repeated action" in final_source_code
    ast.parse(final_source_code)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_deeply_nested_function_replaced_without_corrupting_plan():
    await _run_nested_function_replacement_test()
