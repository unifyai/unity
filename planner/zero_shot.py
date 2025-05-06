import importlib.util
import os
import re
import sys
import uuid
import textwrap
import ast
import inspect
import linecache
from typing import Dict, Any, List, Tuple, Callable
from types import ModuleType

from .model import Primitive
from . import primitives
from .unify_client import set_system_message, generate_prompt
from . import sandbox


PREAMBLE = (
    "from planner.verifier import verify\n" "from planner.primitives import *\n\n"
)


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

Only the FIRST primitive call will be executed; subsequent helper calls **must** be stubbed with `raise NotImplementedError` and decorated with `@verify`.

Return ONLY valid Python code without any explanations or markdown formatting.
"""


def _ensure_verify(decorator_list):
    """
    Ensures that @verify is in the decorator list of a function.

    Args:
        decorator_list: List of AST decorator nodes

    Returns:
        Updated decorator list with @verify added if not present
    """
    # Check if @verify is already in the decorator list
    has_verify = False
    for dec in decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "verify":
            has_verify = True
            break

    # Add @verify if not present
    if not has_verify:
        decorator_list.append(ast.Name(id="verify", ctx=ast.Load()))

    return decorator_list


def _stubify_tree(tree):
    """
    Process an AST to:
    1. Keep only the first primitive call in each function
    2. Add @verify decorator to all helper functions
    3. Add NotImplementedError to functions without concrete code

    Args:
        tree: The AST to process

    Returns:
        The processed source code as a string
    """
    # Get all primitive names from the primitives module
    from . import primitives

    primitive_names = [
        name
        for name in dir(primitives)
        if not name.startswith("_") and callable(getattr(primitives, name))
    ]

    # Track which functions have been processed
    processed_functions = set()

    class PlanTransformer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            # Skip if already processed
            if node.name in processed_functions:
                return node

            processed_functions.add(node.name)

            # Ensure @verify decorator is present for all functions
            node.decorator_list = _ensure_verify(node.decorator_list)

            # Process the function body
            has_primitive = False
            new_body = []

            for stmt in node.body:
                # Check if this statement contains a primitive call
                primitive_call = False

                class PrimitiveVisitor(ast.NodeVisitor):
                    def visit_Call(self, call_node):
                        nonlocal primitive_call
                        if (
                            isinstance(call_node.func, ast.Name)
                            and call_node.func.id in primitive_names
                        ):
                            primitive_call = True
                        self.generic_visit(call_node)

                visitor = PrimitiveVisitor()
                visitor.visit(stmt)

                # Add the statement to the new body
                new_body.append(stmt)

                # If this statement had a primitive call, mark it
                if primitive_call:
                    has_primitive = True
                    # Add NotImplementedError after the first primitive call
                    if len(new_body) < len(node.body):
                        error_stmt = ast.Call(
                            func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                            args=[
                                ast.Constant(value="Subsequent primitive calls are stubbed")
                            ],
                            keywords=[],
                        )
                        new_body.append(ast.Raise(exc=error_stmt))
                        break

            # If the function has no body or no primitive call, add NotImplementedError
            if not new_body or not has_primitive:
                # Add a NotImplementedError statement
                error_stmt = ast.Call(
                    func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                    args=[
                        ast.Constant(value=f"Function {node.name} not implemented")
                    ],
                    keywords=[],
                )
                new_body.append(ast.Raise(exc=error_stmt))

            # Update the function body
            node.body = new_body
            return node

    # Apply the transformation
    transformed = PlanTransformer().visit(tree)
    ast.fix_missing_locations(transformed)

    # Convert back to source code
    return ast.unparse(transformed)


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

    # Process the generated code with AST to enforce single primitive and add stubs
    try:
        processed_code = textwrap.dedent(generated_code).strip()
        processed_code = _stubify_tree(ast.parse(processed_code))
        generated_code = processed_code
    except Exception:
        # Fall back to raw code if parsing fails
        pass

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

        # Ensure inspect.getsource returns stubified code without PREAMBLE
        processed_src = textwrap.dedent(generated_code)
        lines = processed_src.splitlines(keepends=True)
        linecache.cache[file_path] = (len(processed_src), None, lines, file_path)
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
