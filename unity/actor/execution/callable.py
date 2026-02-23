"""Execute a Python callable with structured stdout/stderr capture.

Provides the same output shape as ``PythonExecutionSession.execute()``
but for direct callable invocation (no code-string eval).
"""

from __future__ import annotations

import inspect
import traceback
from typing import Any, Callable

from .capture import capture_sandbox_output


async def execute_callable(
    fn: Callable[..., Any],
    **call_kwargs: Any,
) -> dict:
    """Execute *fn* with stdout/stderr capture, returning a structured dict.

    Returns a dict matching the shape of ``PythonExecutionSession.execute()``::

        {
            "stdout": List[TextPart | ImagePart],
            "stderr": List[TextPart | ImagePart],
            "result": <return value of fn>,
            "error":  <traceback string or None>,
        }
    """
    with capture_sandbox_output() as (stdout_parts, stderr_parts, display_fn):
        try:
            if inspect.iscoroutinefunction(fn):
                result = await fn(**call_kwargs)
            else:
                result = fn(**call_kwargs)
            error = None
        except Exception:
            result = None
            error = traceback.format_exc()

    return {
        "stdout": stdout_parts,
        "stderr": stderr_parts,
        "result": result,
        "error": error,
    }
