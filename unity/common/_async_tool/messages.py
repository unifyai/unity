"""
Contains classes and helpers for manipulating and managing messages in an async tool loop.
"""

import copy
import unify
from typing import Callable, Optional
from .utils import maybe_await
from ...constants import LOGGER


# Helper: scan transcript for assistant messages that have tool_calls with
# missing tool replies (before the next assistant message).
def find_unreplied_assistant_entries(client: unify.AsyncUnify) -> list[dict]:
    findings: list[dict] = []
    try:
        for i, m in enumerate(client.messages):
            if m.get("role") != "assistant":
                continue
            tcs = m.get("tool_calls") or []
            if not tcs:
                continue
            ids = [tc.get("id") for tc in tcs if isinstance(tc, dict)]
            if not ids:
                continue
            responded: set[str] = set()
            j = i + 1
            while (
                j < len(client.messages)
                and client.messages[j].get("role") != "assistant"
            ):
                mm = client.messages[j]
                if mm.get("role") == "tool":
                    tcid = mm.get("tool_call_id")
                    if tcid in ids:
                        responded.add(tcid)
                j += 1
            missing = [c for c in ids if c not in responded]
            if missing:
                findings.append(
                    {
                        "assistant_index": i,
                        "assistant_msg": m,
                        "missing": missing,
                    },
                )
    except Exception:
        pass
    return findings


# Helper: call `client.generate` with optional preprocessing
async def generate_with_preprocess(
    client: unify.AsyncUnify,
    preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]],
    **gen_kwargs,
):
    if preprocess_msgs is None:
        return await maybe_await(client.generate(**gen_kwargs))

    original_msgs = client.messages  # reference to canonical log
    msgs_copy = copy.deepcopy(original_msgs)

    try:
        patched = preprocess_msgs(msgs_copy) or msgs_copy
    except Exception as exc:  # resilience – don't fail the loop
        LOGGER.error(
            f"preprocess_msgs raised {exc!r}; using original messages.",
        )
        patched = msgs_copy

    start_len = len(patched)

    # ------------------------------------------------------------------
    # Some ``AsyncUnify`` implementations (the real one) keep their chat
    # transcript in a **private** attribute ``_messages`` which is what
    # ``.generate`` reads from, while lightweight test doubles (e.g.
    # ``SpyAsyncUnify`` in the test-suite) expose only a public
    # ``messages`` list.  To remain compatible with *both* variants we
    # detect the attribute that is actually consumed by the downstream
    # ``generate`` call and patch **that** for the duration of the call.
    # ------------------------------------------------------------------
    target_attr = "_messages" if hasattr(client, "_messages") else "messages"

    original_container = getattr(client, target_attr)
    setattr(client, target_attr, patched)
    try:
        result = await maybe_await(client.generate(**gen_kwargs))

        # Append any new messages the LLM produced back to canonical log
        current_msgs = getattr(client, target_attr)
        if len(current_msgs) > start_len:
            original_msgs.extend(copy.deepcopy(current_msgs[start_len:]))

        return result
    finally:
        # Always restore the canonical chat log so the outer loop remains
        # consistent irrespective of whether we patched `_messages` or
        # `messages`.
        setattr(client, target_attr, original_container)


def chat_context_repr(
    parent_ctx: Optional[list[dict]],
    current_msgs: list[dict],
) -> list[dict]:
    """
    Combine **existing** ``parent_ctx`` with the *current* chat history
    (``current_msgs``) into a depth-aware nested structure:

        root_msg0
        root_msg1
        root_msg2
          └── children:
              ├── child_msg0
              └── child_msg1

    Strategy – keep the original list untouched and attach the new
    messages as ``children`` of the *last* element.
    """
    ctx_block = [
        {"role": m.get("role"), "content": m.get("content")} for m in current_msgs
    ]
    if not parent_ctx:
        return ctx_block

    combined = copy.deepcopy(parent_ctx)
    combined[-1].setdefault("children", []).extend(ctx_block)
    return combined
