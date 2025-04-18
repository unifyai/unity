"""
primitive.py  – decorator that turns a docstring into an English instruction
                routed through the existing GUI → LLM → Worker pipeline.

Usage:

    from primitive import primitive

    @primitive
    def open_gmail():
        \"\"\"Open gmail.com\"\"\"
        pass
"""

from typing import Callable, Optional
from functools import wraps

# The GUI will register a callable here at runtime
_command_sender: Optional[Callable[[str], None]] = None


def init(sender: Callable[[str], None]) -> None:
    """
    Called once from main.py to connect the decorator to the GUI's
    high‑level text‑command handler.
    """
    global _command_sender
    _command_sender = sender


def primitive(fn: Callable) -> Callable:
    """
    Decorator: when the wrapped function is invoked, its docstring is sent
    as a plain‑English command to the browser controller.  The original
    function body is never executed.
    """
    doc = (fn.__doc__ or "").strip()

    if not doc:
        raise ValueError(f"@primitive function {fn.__name__} must have a docstring")

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if _command_sender is None:
            raise RuntimeError(
                "primitive.init() has not been called – cannot send command",
            )
        _command_sender(doc)  # hand the instruction to the GUI/LLM path

    return wrapper
