import functools
import inspect
import queue
import hashlib
import json
import time
import logging
from typing import Any, Optional, Dict, Tuple, Union

from .context import context
from .model import Primitive
from .primitives import last_primitive
from .verifier_utils import _hash_dom, dom_diff_summary
from .unify_client import (
    set_system_message,
    generate_prompt,
    set_stateful,
    generate_user,
)
from . import sys_msg


_HEURISTIC_TIMEOUT_S = 60
_MAX_REWRITES_PER_FN = 3


class BubbleUp(Exception):
    """Raised by a child function to request that its caller be rewritten.

    This exception is used to signal that the current function cannot fulfill
    its intent and the responsibility should be pushed up to its parent in
    the call stack for reimplementation.
    """

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
    def check(*payload) -> str:
        """
        Dispatcher for checking whether the function execution matched the intent.
        Routes to appropriate helper based on input type.

        Args:
            *payload: Variable arguments that will be routed to the appropriate helper

        Returns:
            Verdict string: "ok", "reimplement", or "push_up_stack"
        """
        if isinstance(payload[0], Primitive):
            return Verifier._check_primitive(payload[0], payload[1], payload[2])
        else:
            return Verifier._check_src(
                payload[0], payload[1], payload[2], payload[3], payload[4]
            )

    # TODO: remove this once we've verified that the new verifier is working
    @staticmethod
    def _check_src(
        src: str,
        before: Dict[str, Any],
        after: Dict[str, Any],
        args: Tuple,
        kwargs: Dict,
    ) -> str:
        """
        Legacy check method for source code based verification.

        Args:
            src: Source code of the function being verified
            before: Browser state snapshot before function execution
            after: Browser state snapshot after function execution
            args: Function arguments
            kwargs: Function keyword arguments

        Returns:
            Verdict string: "ok", "reimplement", or "push_up_stack"
        """
        if before is None or after is None:
            return "ok"  # Can't verify without snapshots

        # Tier 1: Zero-cost heuristics for specific primitives
        # Navigation primitives check (URL change)
        if "open_url" in src or "go_back" in src or "go_forward" in src:
            before_url = before.get("url")
            after_url = after.get("url")

            if before_url == after_url:
                return "reimplement"  # URL should change for navigation functions
            return "ok"

        # Default to "ok" for legacy calls
        return "ok"

    @staticmethod
    def _check_primitive(
        primitive: Primitive, before: Dict[str, Any], after: Dict[str, Any]
    ) -> str:
        """
        Check whether the primitive execution matched the intent
        by comparing before and after state snapshots.

        Args:
            primitive: The primitive action that was executed
            before: Browser state snapshot before function execution
            after: Browser state snapshot after function execution

        Returns:
            Verdict string: "ok", "reimplement", or "push_up_stack"
        """
        if before is None or after is None:
            return "ok"  # Can't verify without snapshots

        # Get primitive name and args
        primitive_name = primitive.name
        primitive_args = primitive.args

        # Tier 1: Zero-cost heuristics for specific primitives

        # Navigation primitives
        if primitive_name in ["open_url", "go_back", "go_forward"]:
            before_url = before.get("url")
            after_url = after.get("url")

            if before_url == after_url:
                return "reimplement"  # URL should change for navigation functions
            return "ok"

        # Scroll primitives
        if primitive_name in [
            "scroll_up",
            "scroll_down",
            "start_scrolling_up",
            "start_scrolling_down",
            "stop_scrolling",
            "continue_scrolling",
        ]:
            return "ok"  # Assume scrolling always works

        # Click button primitive
        if primitive_name == "click_button":
            before_dom_hash = before.get("dom_sha")
            after_dom_hash = after.get("dom_sha")

            if before_dom_hash and after_dom_hash and before_dom_hash != after_dom_hash:
                return "ok"  # DOM changed, so click likely worked

        # Tier 2: Fast fallback - DOM hash comparison
        before_dom_hash = before.get("dom_sha")
        after_dom_hash = after.get("dom_sha")

        if before_dom_hash and after_dom_hash and before_dom_hash != after_dom_hash:
            return "ok"  # DOM changed, so action likely worked

        # For interactive functions, if DOM didn't change, likely failed
        if (
            primitive_name.startswith("click")
            or primitive_name.startswith("press")
            or primitive_name.startswith("select")
        ):
            return "reimplement"

        # Tier 3: LLM fallback with DOM diff
        try:
            # Summarize snapshots for LLM
            before_summary = context.summarise_snapshot(before)
            after_summary = context.summarise_snapshot(after)

            # Generate DOM diff if available
            diff_summary = ""
            if "dom" in before and "dom" in after:
                try:
                    diff_summary = dom_diff_summary(
                        before.get("dom", {}), after.get("dom", {})
                    )
                except Exception as e:
                    logging.warning(f"Error generating DOM diff: {e}")

            # Build a payload for the LLM
            payload = {
                "primitive_name": primitive_name,
                "primitive_args": primitive_args,
                "before_url": before.get("url"),
                "after_url": after.get("url"),
                "before_title": before.get("title"),
                "after_title": after.get("title"),
                "dom_changed": (
                    before_dom_hash != after_dom_hash
                    if before_dom_hash and after_dom_hash
                    else "unknown"
                ),
                "diff_summary": (
                    diff_summary[:1000] if diff_summary else "No diff available"
                ),
            }

            # Convert to JSON string for the prompt
            payload_json = json.dumps(payload, default=str, indent=2)

            # Set up the LLM for verification
            set_stateful(False)

            # Generate the verdict using LLM
            prompt = sys_msg.VERIFY_PRIMITIVE_PROMPT.format(payload_json=payload_json)

            verdict = generate_user(prompt).strip().lower()

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
        retries = 0
        t_start = time.time()

        # Push the current function onto the call stack
        context.push_frame(fn.__name__)
        try:
            while True:
                # Check if we've exceeded retry limits or timeout
                elapsed = time.time() - t_start
                if retries >= _MAX_REWRITES_PER_FN or elapsed > _HEURISTIC_TIMEOUT_S:
                    raise RuntimeError(
                        f"Verifier gave up on {fn.__name__} after {retries} retries and {elapsed:.2f}s"
                    )

                before = context.get_snapshot()
                try:
                    # Execute the function and get the primitive
                    src = inspect.getsource(fn)
                    result = fn(*args, **kwargs)
                    primitive = last_primitive()
                except NotImplementedError:
                    Verifier.get_reimplement_queue().put(fn)
                    retries += 1
                    continue
                except BubbleUp as e:
                    # Enqueue this function for rewrite and retry
                    Verifier.get_reimplement_queue().put(fn)
                    retries += 1
                    continue

                after = context.get_snapshot()

                # Use the primitive for verification if available
                verdict = Verifier.check(src, before, after, args, kwargs)

                if verdict == "ok":
                    return result
                elif verdict == "reimplement":
                    Verifier.get_reimplement_queue().put(fn)
                    retries += 1
                    continue
                elif verdict == "push_up_stack":
                    raise BubbleUp(f"Intent '{fn.__name__}' requires pushing up")
        finally:
            # Ensure we pop the frame from the call stack
            context.pop_frame()

    wrapper._wrapped_fn = fn
    return wrapper
