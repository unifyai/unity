"""Code execution infrastructure for actors.

Re-exports the public surface so consumers can write::

    from unity.actor.execution import (
        ExecutionResult, TextPart, ImagePart, parts_to_text, ...
    )
"""

from .types import (
    ExecutionResult,
    ImagePart,
    OutputPart,
    TextPart,
    _detect_image_mime_from_b64,
    parts_to_llm_content,
    parts_to_text,
)

from .capture import (
    StreamLike,
    StreamRouter,
    _make_display,
    _stderr_parts,
    _stdout_parts,
    capture_sandbox_output,
)

from .callable import execute_callable

from .package_overlay import PackageOverlay, _CURRENT_PACKAGE_OVERLAY

from .session import (
    PythonExecutionSession,
    SessionExecutor,
    SessionKey,
    StateMode,
    SupportedLanguage,
    SupportedShellLanguage,
    _CURRENT_SANDBOX,
    _PARENT_CHAT_CONTEXT,
    _execute_shell_stateless,
    _validate_execution_params,
    _wrap_code_as_async_function,
)

__all__ = [
    # types
    "TextPart",
    "ImagePart",
    "OutputPart",
    "ExecutionResult",
    "parts_to_text",
    "parts_to_llm_content",
    "_detect_image_mime_from_b64",
    # capture
    "StreamLike",
    "StreamRouter",
    "_stdout_parts",
    "_stderr_parts",
    "_make_display",
    "capture_sandbox_output",
    # callable
    "execute_callable",
    # package_overlay
    "PackageOverlay",
    "_CURRENT_PACKAGE_OVERLAY",
    # session
    "SupportedShellLanguage",
    "SupportedLanguage",
    "StateMode",
    "SessionKey",
    "_CURRENT_SANDBOX",
    "_PARENT_CHAT_CONTEXT",
    "_validate_execution_params",
    "PythonExecutionSession",
    "SessionExecutor",
    "_wrap_code_as_async_function",
    "_execute_shell_stateless",
]
