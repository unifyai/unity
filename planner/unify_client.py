"""
Planner unify client wrapper for centralizing LLM calls.
"""

import time
from datetime import datetime, timezone

import unify
from constants import LOGGER

# Single traced Unify client for all planner interactions
client = unify.Unify(traced=True)
client.set_endpoint("o4-mini@openai")
client.set_stateful(True)

# Fallback flag in case SDK doesn't support set_stateful
_is_stateful = True


def set_system_message(prompt: str) -> None:
    """
    Set the system message for the planner's Unify client.
    """
    LOGGER.info(f"🤖 Planner: setting system message.")
    client.set_system_message(prompt)


def set_response_format(schema) -> None:
    """
    Set the response format schema for the planner's Unify client.
    """
    LOGGER.info(f"🤖 Planner: setting response format.")
    client.set_response_format(schema)


def set_stateful(flag: bool) -> None:
    """
    Control whether the client maintains memory across calls.

    Args:
        flag: If True, maintain conversation history. If False, clear history after each call.
    """
    global _is_stateful
    LOGGER.info(f"🤖 Planner: setting stateful mode to {flag}.")

    # Try to use the SDK's method if available
    try:
        client.set_stateful(flag)
    except AttributeError:
        # Fall back to manual management if the SDK doesn't support it
        if not flag:
            # Preserve system messages
            client.messages = [
                m for m in getattr(client, "messages", []) if m.get("role") == "system"
            ]
    finally:
        _is_stateful = flag


def generate_user(content) -> any:
    """
    Send a chat-style user message to Unify and return the response.
    """
    # Save current stateful state
    current_stateful = _is_stateful

    # Temporarily set to stateless for this call
    set_stateful(False)

    t0 = time.perf_counter()
    t = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
    LOGGER.info(f"\n🤖 Planner: sending user message... ⏳ [⏱️ {t}]\n")
    messages = client.messages + [{"role": "user", "content": content}]
    response = client.generate(messages=messages)
    t2 = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
    elapsed = time.perf_counter() - t0
    LOGGER.info(f"\n🤖 Planner: received response ✅ [⏱️ {t2}] [⏩{elapsed:.3g}s]\n")

    # Restore previous stateful state
    set_stateful(current_stateful)

    return response


def generate_prompt(prompt: str) -> any:
    """
    Send a plain prompt to Unify and return the response.
    """
    # Save current stateful state
    current_stateful = _is_stateful

    # Temporarily set to stateless for this call
    set_stateful(False)

    t0 = time.perf_counter()
    t = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
    LOGGER.info(f"\n🤖 Planner: sending prompt... ⏳ [⏱️ {t}]\n")
    response = client.generate(prompt)
    t2 = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
    elapsed = time.perf_counter() - t0
    LOGGER.info(
        f"\n🤖 Planner: received prompt response ✅ [⏱️ {t2}] [⏩{elapsed:.3g}s]\n"
    )

    # Restore previous stateful state
    set_stateful(current_stateful)

    return response
