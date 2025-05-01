import ast
import importlib.util
import os
import re
import sys
import uuid
import textwrap
from typing import Dict, Any, List, Optional, Tuple, Union, cast, Callable
from types import ModuleType

from .model import FunctionNode, Plan, Primitive
from . import primitives
from .unify_client import set_system_message, generate_prompt
from . import sandbox


PREAMBLE = (
    "from planner.verifier import verify\n" "from planner.primitives import *\n\n"
)


# TODO remove this helper after migrate to sandbox.exec_plan
def _write_and_load_module(code: str) -> Tuple[str, Dict[str, Callable]]:
    """
    Writes the generated code to a uniquely named file and loads it as a module.

    Args:
        code: The Python code string to write and load.

    Returns:
        A tuple containing the module path and a dictionary mapping function names to callables.
    """
    # Create a unique module name
    module_name = f"generated_{uuid.uuid4().hex}"

    # Determine the directory path (planner package directory)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dir_path, f"{module_name}.py")

    # Write the code to the file
    with open(file_path, "w") as f:
        # Add imports for verify decorator and primitives
        f.write("from unity.planner.verifier import verify\n")
        f.write("from unity.planner.primitives import *\n\n")
        f.write(code)

    # Load the module using importlib
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Collect all callables from the module
    callables = {}
    for name in dir(module):
        if not name.startswith("_"):
            attr = getattr(module, name)
            if callable(attr):
                callables[name] = attr

    return file_path, callables


# Template for the prompt to generate the initial plan
PROMPT_TEMPLATE = """
You are an expert Python programmer tasked with creating execution plans.
Your plans must be valid Python code that can be parsed and executed.
Always define a root function named 'root_plan' and include primitive helper calls.

You are given a task: {task_str}

Use ONLY the following primitive helper functions (no others):
{helper_list}

For each primitive, here's its purpose:
{helper_descriptions}

Define a root Python function named `root_plan`.
Include at least one call to a primitive helper and stub any other logic
with `raise NotImplementedError` statements.

Return ONLY valid Python code without any explanations or markdown formatting.
"""


def create_initial_plan(task_str: str) -> Tuple[ModuleType, Callable]:
    """
    Generates an initial execution plan for the given task string by invoking the
    Unify agent in a zero-shot fashion. The agent is instructed to use only the
    available primitive helpers and must define a root_plan function.

    Args:
        task_str: A natural-language description of the task.

    Returns:
        A tuple containing (module, root_fn) where:
        - module: The module containing the executed plan
        - root_fn: The callable root_plan function
    """
    # Introspect available primitive helpers
    helper_names: List[str] = [
        name
        for name in dir(primitives)
        if not name.startswith("_") and callable(getattr(primitives, name))
    ]
    # Sort helper names for stable ordering in prompts
    helper_names = sorted(helper_names)

    # Create helper descriptions
    helper_descriptions = ""
    for name in helper_names:
        func = getattr(primitives, name)
        doc = func.__doc__ or f"Helper function to {name.replace('_', ' ')}"
        helper_descriptions += f"- {name}: {doc.strip()}\n"

    # Build the prompt using the template
    prompt = PROMPT_TEMPLATE.format(
        task_str=task_str,
        helper_list=", ".join(helper_names),
        helper_descriptions=helper_descriptions,
    )

    # Invoke LLM to generate the plan code
    generated_code: str = generate_prompt(prompt)

    # Strip markdown fences if present
    generated_code = _strip_markdown_fences(generated_code)

    # Ensure the code has a root_plan function
    if "def root_plan" not in generated_code:
        # Add a stub root_plan function if missing
        generated_code += "\n\ndef root_plan():\n    raise NotImplementedError('Root plan function not implemented')\n"

    # Create directory for plans if it doesn't exist
    os.makedirs("/tmp/plans", exist_ok=True)

    # Generate a unique filename
    plan_id = uuid.uuid4().hex
    file_path = f"/tmp/plans/plan_{plan_id}.py"

    # Prepend essential imports for generated code
    full_code = PREAMBLE + textwrap.dedent(generated_code)
    with open(file_path, "w") as f:
        f.write(full_code)

    # Execute the plan in the sandbox, passing the filepath so inspect.getsource can locate the source
    try:
        module = sandbox.exec_plan(full_code, filename=file_path)
    except Exception as e:
        raise ValueError(f"Failed to execute plan: {str(e)}")

    # Get the root_plan function
    if not hasattr(module, "root_plan"):
        raise ValueError("Generated code does not contain a root_plan function")

    root_fn = module.root_plan

    # Validate that root_fn is callable
    if not callable(root_fn):
        raise ValueError("root_plan is not callable")

    return module, root_fn


def _parse_generated_code(code: str) -> Plan:
    """
    Parses Python code into a Plan using AST parsing.
    Extracts function definitions and builds a hierarchy of FunctionNodes.

    Args:
        code: The Python code string returned by the Unify agent.

    Returns:
        A Plan with a hierarchical structure of FunctionNodes and Primitives.
    """
    # Step 1: Strip Markdown fences if present
    code = _strip_markdown_fences(code)

    # Step 2: Parse the code using AST
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Failed to parse generated code: {str(e)}")

    # Find all function definitions
    function_defs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_defs[node.name] = node

    # Find the root function (should be decorated with @verify)
    root_func = None
    for name, func_def in function_defs.items():
        if func_def.decorator_list and any(
            getattr(decorator, "id", "") == "verify"
            for decorator in func_def.decorator_list
        ):
            root_func = func_def
            break

    # If no decorated function found, look for root_plan
    if not root_func and "root_plan" in function_defs:
        root_func = function_defs["root_plan"]

    if not root_func:
        raise ValueError(
            "No root function found in generated code. Expected a function decorated with @verify or named 'root_plan'."
        )

    # Build the function node tree starting from the root
    root_node = _build_function_node(root_func, function_defs)

    # Validate we have at least one primitive call
    if not _contains_primitive(root_node):
        raise ValueError("No valid primitive calls found in the generated plan.")

    # Construct the Plan
    return Plan(root_node)


def _strip_markdown_fences(code: str) -> str:
    """
    Removes markdown code fences if present.

    Args:
        code: The code string that might contain markdown fences.

    Returns:
        The code string with markdown fences removed.
    """
    # Extract code block if wrapped in markdown
    code_block_match = re.search(r"```(?:python)?\s*(.*?)\s*```", code, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1)
    return code


def _build_function_node(
    func_def: ast.FunctionDef, function_defs: Dict[str, ast.FunctionDef]
) -> FunctionNode:
    """
    Recursively builds a FunctionNode from an AST FunctionDef node.

    Args:
        func_def: The AST FunctionDef node.
        function_defs: Dictionary of all function definitions.

    Returns:
        A FunctionNode representing the function and its body.
    """
    body_nodes = []

    for stmt in func_def.body:
        # Skip docstrings
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            continue

        # Skip NotImplemented raises
        if (
            isinstance(stmt, ast.Raise)
            and isinstance(stmt.exc, ast.Name)
            and stmt.exc.id == "NotImplemented"
        ):
            continue

        # Process function calls
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call_node = stmt.value
            func_name = _get_call_name(call_node)

            if hasattr(primitives, func_name):
                # This is a primitive call
                args_dict = _extract_call_args(call_node)
                primitive = _create_primitive(func_name, args_dict)
                body_nodes.append(primitive)
            elif func_name in function_defs:
                # This is a call to another defined function
                nested_func_def = function_defs[func_name]
                nested_node = _build_function_node(nested_func_def, function_defs)
                body_nodes.append(nested_node)

        # Handle loops and conditionals as function node placeholders
        elif isinstance(stmt, (ast.For, ast.While, ast.If)):
            placeholder_name = f"{func_def.name}_{type(stmt).__name__.lower()}"
            loop_body = _process_compound_statement(stmt, function_defs)
            body_nodes.append(FunctionNode(placeholder_name, loop_body))

    return FunctionNode(func_def.name, body_nodes)


def _process_compound_statement(
    stmt: Union[ast.For, ast.While, ast.If], function_defs: Dict[str, ast.FunctionDef]
) -> List[Union[FunctionNode, Primitive]]:
    """
    Processes compound statements like loops and conditionals.

    Args:
        stmt: The AST node representing a compound statement.
        function_defs: Dictionary of all function definitions.

    Returns:
        A list of nodes representing the body of the compound statement.
    """
    body_nodes = []

    # Process the body of the compound statement
    for body_stmt in stmt.body:
        if isinstance(body_stmt, ast.Expr) and isinstance(body_stmt.value, ast.Call):
            call_node = body_stmt.value
            func_name = _get_call_name(call_node)

            if hasattr(primitives, func_name):
                args_dict = _extract_call_args(call_node)
                primitive = _create_primitive(func_name, args_dict)
                body_nodes.append(primitive)
            elif func_name in function_defs:
                nested_func_def = function_defs[func_name]
                nested_node = _build_function_node(nested_func_def, function_defs)
                body_nodes.append(nested_node)

        elif isinstance(body_stmt, (ast.For, ast.While, ast.If)):
            # Recursively process nested compound statements
            nested_body = _process_compound_statement(body_stmt, function_defs)
            placeholder_name = f"nested_{type(body_stmt).__name__.lower()}"
            body_nodes.append(FunctionNode(placeholder_name, nested_body))

    # Also process the orelse part for if statements
    if isinstance(stmt, ast.If) and stmt.orelse:
        orelse_nodes = []
        for orelse_stmt in stmt.orelse:
            if isinstance(orelse_stmt, ast.Expr) and isinstance(
                orelse_stmt.value, ast.Call
            ):
                call_node = orelse_stmt.value
                func_name = _get_call_name(call_node)

                if hasattr(primitives, func_name):
                    args_dict = _extract_call_args(call_node)
                    primitive = _create_primitive(func_name, args_dict)
                    orelse_nodes.append(primitive)

        if orelse_nodes:
            body_nodes.append(FunctionNode("else_branch", orelse_nodes))

    return body_nodes


def _get_call_name(call_node: ast.Call) -> str:
    """
    Extracts the function name from a Call node.

    Args:
        call_node: The AST Call node.

    Returns:
        The name of the function being called.
    """
    if isinstance(call_node.func, ast.Name):
        return call_node.func.id
    elif isinstance(call_node.func, ast.Attribute):
        return call_node.func.attr
    return ""


def _extract_call_args(call_node: ast.Call) -> Dict[str, Any]:
    """
    Extracts arguments from a Call node into a dictionary.

    Args:
        call_node: The AST Call node.

    Returns:
        A dictionary of argument names and values.
    """
    args_dict = {}

    # Process positional arguments
    for i, arg in enumerate(call_node.args):
        if isinstance(arg, ast.Constant):
            args_dict[f"arg{i}"] = arg.value
        elif isinstance(arg, ast.Name):
            args_dict[f"arg{i}"] = arg.id
        elif isinstance(arg, ast.Str):  # For Python 3.7 compatibility
            args_dict[f"arg{i}"] = arg.s

    # Process keyword arguments
    for keyword in call_node.keywords:
        if keyword.arg is not None:
            if isinstance(keyword.value, ast.Constant):
                args_dict[keyword.arg] = keyword.value.value
            elif isinstance(keyword.value, ast.Name):
                args_dict[keyword.arg] = keyword.value.id
            elif isinstance(keyword.value, ast.Str):  # For Python 3.7 compatibility
                args_dict[keyword.arg] = keyword.value.s

    return args_dict


def _create_primitive(func_name: str, args_dict: Dict[str, Any]) -> Primitive:
    """
    Creates a Primitive instance with the correct call_literal.

    Args:
        func_name: The name of the primitive function.
        args_dict: Dictionary of arguments for the primitive.

    Returns:
        A Primitive instance.
    """
    # Get the primitive function
    primitive_func = getattr(primitives, func_name)

    # Create a call literal based on the primitive's expected format
    # This will use the primitive's implementation to generate the correct command string
    try:
        # Create a sample primitive to get its command format
        if not args_dict:
            # No arguments
            sample_primitive = primitive_func()
        elif len(args_dict) == 1 and "arg0" in args_dict:
            # Single positional argument
            sample_primitive = primitive_func(args_dict["arg0"])
        else:
            # Multiple or keyword arguments
            # For simplicity, we'll just use the call_literal directly
            args_str = ", ".join(
                [
                    f"{k}={repr(v)}" if not k.startswith("arg") else repr(v)
                    for k, v in args_dict.items()
                ]
            )
            call_literal = f"{func_name}({args_str})"
            return Primitive(func_name, args_dict, call_literal)

        # Use the sample primitive's call_literal
        return Primitive(func_name, args_dict, sample_primitive.call_literal)
    except Exception:
        # Fallback: construct a basic call_literal
        args_str = ", ".join(
            [
                f"{k}={repr(v)}" if not k.startswith("arg") else repr(v)
                for k, v in args_dict.items()
            ]
        )
        call_literal = f"{func_name}({args_str})"
        return Primitive(func_name, args_dict, call_literal)


def _contains_primitive(node: FunctionNode) -> bool:
    """
    Checks if a FunctionNode contains at least one Primitive.

    Args:
        node: The FunctionNode to check.

    Returns:
        True if the node contains at least one Primitive, False otherwise.
    """
    for item in node.body:
        if isinstance(item, Primitive):
            return True
        elif isinstance(item, FunctionNode):
            if _contains_primitive(item):
                return True
    return False
