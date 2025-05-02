import functools
import inspect
import queue
import hashlib
import json
import time
import logging
from typing import Any, Optional

from . import context
from .unify_client import set_system_message, generate_prompt, set_stateful


_HEURISTIC_TIMEOUT_S = 15
_MAX_REWRITES_PER_FN = 3


def _fingerprint(state: dict) -> str:
    """
    Generate a stable fingerprint of the browser state.

    Args:
        state: Browser state snapshot

    Returns:
        A hash string representing the state fingerprint
    """
    if not state:
        return ""

    # Extract key fields for fingerprinting
    fingerprint_data = (
        state.get("url"),
        state.get("title"),
        state.get("scroll_y"),
        state.get("in_textbox"),
        state.get("active_tab"),
        state.get("dom_sha"),
    )

    # Convert to JSON string and hash
    json_data = json.dumps(fingerprint_data, sort_keys=True)
    return hashlib.sha256(json_data.encode()).hexdigest()


def _cheap_heuristic(fn_name: str, before: dict, after: dict) -> Optional[str]:
    """
    Apply cheap heuristics to determine if a function succeeded or failed.

    Args:
        fn_name: Name of the function being verified
        before: Browser state snapshot before function execution
        after: Browser state snapshot after function execution

    Returns:
        Verdict string or None if heuristic can't determine
    """
    if before is None or after is None:
        return "ok"  # Can't verify without snapshots

    # Get fingerprints
    before_fp = _fingerprint(before)
    after_fp = _fingerprint(after)

    # Check for common patterns
    if fn_name.startswith("open_url") or fn_name.startswith("navigate_to"):
        before_url = before.get("url")
        after_url = after.get("url")

        if before_url == after_url:
            return "reimplement"  # URL should change for navigation functions

    elif (
        fn_name.startswith("click")
        or fn_name.startswith("press")
        or fn_name.startswith("select")
    ):
        # For interactive functions, something should change
        if before_fp == after_fp:
            return "reimplement"  # State should change for interactive functions

    elif fn_name.startswith("search") or fn_name.startswith("fill_form"):
        # For search/form functions, either URL or DOM should change
        if before_fp == after_fp:
            return "reimplement"

    # No clear determination from heuristics
    return None


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
        # Get function name from the source code
        fn_name = src.strip().split("def ")[1].split("(")[0].strip()

        # Try cheap heuristics first
        heuristic_result = _cheap_heuristic(fn_name, before, after)
        if heuristic_result is not None:
            return heuristic_result

        # Calculate fingerprints for before and after states
        before_fp = _fingerprint(before) if before else ""
        after_fp = _fingerprint(after) if after else ""

        # Prepare payload for LLM
        try:
            # Convert args and kwargs to strings to avoid serialization issues
            args_str = str(args)
            kwargs_str = str(kwargs)

            # Build a payload for the LLM
            payload = {
                "fn_name": fn_name,
                "args": args_str,
                "kwargs": kwargs_str,
                "before": before,
                "after": after,
                "before_fp": before_fp,
                "after_fp": after_fp,
                "src": src,
            }

            # Convert to JSON string for the prompt
            payload_json = json.dumps(payload, default=str, indent=2)

            # Set up the LLM for verification
            set_stateful(False)
            set_system_message(
                """You are a verification expert for browser automation tasks.
                Analyze the function execution and determine if it succeeded.
                Reply with EXACTLY ONE of these verdicts:
                - "ok" if the function achieved its intent
                - "reimplement" if the function failed and should be rewritten
                - "push_up_stack" if the intent should be handled at a higher level
                Base your decision on the function name, code, arguments, and browser state changes."""
            )

            # Generate the verdict using LLM
            prompt = f"""
            Verify if this browser automation function succeeded:
            
            Function information:
            {payload_json}
            
            Respond with exactly one of: "ok", "reimplement", or "push_up_stack"
            """

            verdict = generate_prompt(prompt).strip().lower()

            # Validate the verdict
            if verdict in ["ok", "reimplement", "push_up_stack"]:
                return verdict
            else:
                # Invalid response, default to reimplement
                logging.warning(
                    f"Invalid verification verdict: {verdict}. Defaulting to 'reimplement'"
                )
                return "reimplement"

        except Exception as e:
            # On any error, default to reimplement
            logging.warning(
                f"Error during verification: {e}. Defaulting to 'reimplement'"
            )
            return "reimplement"


def verify(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        src = inspect.getsource(fn)
        retries = 0
        t_start = time.time()

        while True:
            # Check if we've exceeded retry limits or timeout
            elapsed = time.time() - t_start
            if retries >= _MAX_REWRITES_PER_FN or elapsed > _HEURISTIC_TIMEOUT_S:
                raise RuntimeError(
                    f"Verifier gave up on {fn.__name__} after {retries} retries and {elapsed:.2f}s"
                )

            before = context.get_snapshot()
            try:
                result = fn(*args, **kwargs)
            except NotImplementedError:
                Verifier.get_reimplement_queue().put(fn)
                retries += 1
                continue

            after = context.get_snapshot()
            verdict = Verifier.check(src, before, after, args, kwargs)

            if verdict == "ok":
                return result
            elif verdict == "reimplement":
                Verifier.get_reimplement_queue().put(fn)
                retries += 1
                continue
            elif verdict == "push_up_stack":
                raise BubbleUp(f"Intent '{fn.__name__}' requires pushing up")

    wrapper._wrapped_fn = fn
    return wrapper
