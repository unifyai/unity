import functools
import inspect
import queue
from typing import Any, Optional

from .context import get_snapshot
import planner.code_rewriter as code_rewriter


class BubbleUp(Exception):
    """Raised when the intent should be pushed up the call stack."""

    pass


class Verifier:
    # Queue for reimplement requests
    _reimplement_queue = queue.Queue()

    @staticmethod
    def get_reimplement_queue() -> queue.Queue:
        """
        Get the queue for reimplement requests.

        Returns:
            Queue containing functions that need to be reimplemented
        """
        return Verifier._reimplement_queue

    @staticmethod
    def check(src: str, before: Any, after: Any, args: Any, kwargs: Any) -> str:
        """
        Check whether the function execution matched the intent
        by comparing before and after state snapshots.
        Should return one of: "ok", "reimplement", "push_up_stack".

        Args:
            src: The source code of the function being verified
            before: Browser state snapshot before function execution
            after: Browser state snapshot after function execution
            args: Positional arguments passed to the function
            kwargs: Keyword arguments passed to the function

        Returns:
            Verdict string: "ok", "reimplement", or "push_up_stack"
        """
        # TODO: Implement a more sophisticated verifier using LLM

        # Simple heuristic based on URL presence or search terms
        if before is None or after is None:
            return "ok"  # Can't verify without snapshots

        before_url = before.get("url") if isinstance(before, dict) else None
        after_url = after.get("url") if isinstance(after, dict) else None

        if before_url == after_url:
            return "reimplement"
        return "ok"


def verify(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        src = inspect.getsource(fn)
        while True:
            before = get_snapshot()
            try:
                result = fn(*args, **kwargs)
            except NotImplementedError:
                code_rewriter.rewrite_function(fn)
                continue

            after = get_snapshot()
            verdict = Verifier.check(src, before, after, args, kwargs)

            if verdict == "ok":
                return result
            elif verdict == "reimplement":
                code_rewriter.rewrite_function(fn)
                continue
            elif verdict == "push_up_stack":
                raise BubbleUp(f"Intent '{fn.__name__}' requires pushing up")
            else:
                raise RuntimeError(f"Unknown verdict {verdict!r}")

    wrapper._wrapped_fn = fn
    return wrapper
