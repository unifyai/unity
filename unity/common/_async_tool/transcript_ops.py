from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypedDict

# Central placeholder/non-final detection lives here
from .messages import is_non_final_tool_reply as _is_non_final_tool_reply


class CleanToolCall(TypedDict):
    index: int
    name: str
    arguments: str
    result: Any


def _prune_assistant_msg(
    msg: dict,
    *,
    allowed_tools: Optional[Set[str]] = None,
) -> dict | None:
    """
    Return a shallow-copied assistant message with tool_calls filtered by policy,
    or None when no tool_calls survive.

    Invariants and policy:
    - Drops synthetic helper calls whose names start with "check_status_".
    - When `allowed_tools` is provided, only keeps tools in that set; always
      preserves the built-in clarification helper "request_clarification" even
      if it is not in `allowed_tools`.
    """
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return None
    tool_calls = msg.get("tool_calls") or []
    if not isinstance(tool_calls, list) or not tool_calls:
        return None

    kept: list = []
    for tc in tool_calls:
        try:
            fn = tc.get("function", {})
            name = fn.get("name")
        except Exception:
            name = None
        if not isinstance(name, str) or not name:
            continue
        # Drop synthetic completion/status helpers
        if name.startswith("check_status_"):
            continue
        # Keep clarification helper even if not part of base registry
        if allowed_tools is not None and name not in allowed_tools:
            if name != "request_clarification":
                continue
        kept.append(tc)

    if not kept:
        return None

    pruned = dict(msg)
    pruned["tool_calls"] = kept
    return pruned


def extract_assistant_and_tool_steps(
    msgs: List[dict],
    *,
    allowed_tools: Optional[Set[str]] = None,
    non_final: Optional[Callable[[dict], bool]] = None,
) -> Dict[str, Any]:
    """
    Extract pruned assistant tool-call messages and paired final tool results.

    Returns a dict with keys:
      - assistant_steps: List[dict]
      - assistant_indices: List[int]
      - tool_results: List[dict]
      - tool_results_indices: List[int]
      - callid_to_tool_name: Dict[str, str]
      - final_call_ids: Set[str]
      - pending_call_ids: Set[str]

    Invariants and behaviour:
    - Assistant messages are pruned via `_prune_assistant_msg` respecting
      `allowed_tools` and keeping clarification helpers.
    - Tool results are paired by `tool_call_id` and we keep the LAST occurrence
      per call id to reflect the final state in a streaming transcript.
    - Final vs pending is determined using `non_final` (defaults to
      `messages.is_non_final_tool_reply`): only replies that are not placeholders
      (e.g., progress or clarification wrappers) are considered final.
    - Indices are taken from the original `msgs` list to allow precise
      reconstruction when needed.
    """
    if non_final is None:
        non_final = _is_non_final_tool_reply

    assistant_steps_raw: List[dict] = []
    assistant_indices_raw: List[int] = []
    tool_results_raw: List[dict] = []
    tool_results_raw_indices: List[int] = []

    for i, m in enumerate(msgs or []):
        try:
            role = m.get("role")
        except Exception:
            continue
        if role == "assistant":
            pruned = _prune_assistant_msg(m, allowed_tools=allowed_tools)
            if pruned is not None:
                assistant_steps_raw.append(pruned)
                assistant_indices_raw.append(i)
        elif role == "tool":
            tool_results_raw.append(m)
            tool_results_raw_indices.append(i)

    # Build the set of referenced call_ids from pruned assistant steps
    referenced_call_ids: Set[str] = set()
    callid_to_tool_name: Dict[str, str] = {}
    for am in assistant_steps_raw:
        for tc in am.get("tool_calls", []) or []:
            try:
                _cid = tc.get("id")
                _nm = (tc.get("function") or {}).get("name")
            except Exception:
                _cid, _nm = None, None
            if isinstance(_cid, str) and _cid:
                referenced_call_ids.add(_cid)
                if isinstance(_nm, str) and _nm:
                    callid_to_tool_name[_cid] = _nm

    # Keep only tool replies whose call_id is referenced and whose name passes policy.
    last_index_by_call_id: Dict[str, int] = {}
    for idx, tm in enumerate(tool_results_raw):
        try:
            name = tm.get("name")
            call_id = tm.get("tool_call_id")
        except Exception:
            continue
        if not isinstance(call_id, str) or not call_id:
            continue
        if call_id not in referenced_call_ids:
            continue
        if allowed_tools is not None:
            # Allow clarification wrappers to pass even if not in tool registry
            if not (
                (isinstance(name, str) and name in allowed_tools)
                or (isinstance(name, str) and name.startswith("clarification_request_"))
                or (name == "request_clarification")
            ):
                continue
        last_index_by_call_id[call_id] = idx

    tool_results: List[dict] = []
    tool_results_indices: List[int] = []
    for idx, tm in enumerate(tool_results_raw):
        try:
            call_id = tm.get("tool_call_id")
        except Exception:
            call_id = None
        if (
            isinstance(call_id, str)
            and call_id in last_index_by_call_id
            and last_index_by_call_id[call_id] == idx
        ):
            tool_results.append(tm)
            try:
                tool_results_indices.append(tool_results_raw_indices[idx])
            except Exception:
                tool_results_indices.append(-1)

    # Compute final vs pending call_ids using non-final predicate
    final_call_ids: Set[str] = set()
    try:
        for tm in tool_results:
            try:
                _cid = tm.get("tool_call_id")
            except Exception:
                _cid = None
            if isinstance(_cid, str) and _cid and _cid in referenced_call_ids:
                if not non_final(tm):  # treat only non-placeholder replies as final
                    final_call_ids.add(_cid)
    except Exception:
        final_call_ids = set()

    pending_call_ids: Set[str] = set()
    try:
        pending_call_ids = referenced_call_ids - final_call_ids
    except Exception:
        pending_call_ids = set()

    return {
        "assistant_steps": assistant_steps_raw,
        "assistant_indices": assistant_indices_raw,
        "tool_results": tool_results,
        "tool_results_indices": tool_results_indices,
        "callid_to_tool_name": callid_to_tool_name,
        "final_call_ids": final_call_ids,
        "pending_call_ids": pending_call_ids,
    }


def extract_interjections(msgs: List[dict]) -> Tuple[List[dict], List[int]]:
    """
    Return (interjections, interjections_indices) for user interjection
    messages (user messages that appear after the first user message).

    Invariant: The first user message is the original request; all subsequent
    user messages are treated as interjections. This design uses user messages
    for interjections (not system messages) for broad provider compatibility.

    For backwards compatibility, also captures any non-leading system messages
    that may exist from older transcripts (system messages at positions > 0
    that are not marked as context headers).
    """
    interjections: List[dict] = []
    interjections_indices: List[int] = []
    first_user_seen = False

    for i, m in enumerate(msgs or []):
        try:
            role = m.get("role")

            # New format: user messages after the first user message are interjections
            if role == "user":
                if first_user_seen:
                    interjections.append(m)
                    interjections_indices.append(i)
                else:
                    first_user_seen = True
                continue

            # Backwards compatibility: non-leading system messages without _ctx_header
            # are treated as interjections (from older transcripts)
            if role == "system" and i > 0:
                # Skip context header system messages
                if m.get("_ctx_header"):
                    continue
                interjections.append(m)
                interjections_indices.append(i)
        except Exception:
            continue
    return interjections, interjections_indices


def extract_clarifications(
    assistant_steps: List[dict],
    tool_results: List[dict],
    *,
    callid_to_tool_name: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """
    Build a clarifications summary from tool_results whose name startswith
    "clarification_request_". Uses callid_to_tool_name to attach the base tool
    name.

    Invariants:
    - Each entry contains {call_id, tool, question} where `question` is the
      tool message content verbatim; no parsing or rewriting is applied.
    """
    callid_to_tool_name = callid_to_tool_name or {}
    clarifications: List[dict] = []
    for tm in tool_results:
        try:
            name = str(tm.get("name"))
            cid = str(tm.get("tool_call_id"))
            content = tm.get("content")
        except Exception:
            continue
        if isinstance(name, str) and name.startswith("clarification_request_"):
            clarifications.append(
                {
                    "call_id": cid,
                    "tool": callid_to_tool_name.get(cid, ""),
                    "question": content,
                },
            )
    return clarifications


def initial_user_from_user_visible_history(history: List[dict] | None) -> Any:
    """
    Return the initial user-visible message content when available.

    Invariant: inspects only the first entry and returns its `content` when the
    entry has role == "user"; otherwise returns None.
    """
    try:
        if history:
            first = history[0]
            if isinstance(first, dict) and first.get("role") == "user":
                return first.get("content")
    except Exception:
        pass
    return None


def build_clean_tool_trajectory(
    msgs: List[dict],
    *,
    drop_names: Optional[Set[str]] = None,
    non_final: Optional[Callable[[dict], bool]] = None,
) -> List[CleanToolCall]:
    """
    Flatten a (assistant, tool) transcript into a sequenced list of tool calls
    with arguments/results, preserving the execution order implied by assistant
    tool_calls and pairing against the last corresponding tool result.

    Invariants and behaviour:
    - For each call id, pairs against the LAST tool reply message (final state).
    - Uses `non_final` (defaults to `messages.is_non_final_tool_reply`) to
      exclude placeholders/progress/clarification wrappers from results.
    - Only considers tool replies whose `tool_call_id` conforms to typical
      provider shape (startswith "call_") to avoid mis-pairing arbitrary ids.
    - Respects `drop_names` to omit specific tool names from the trajectory.
    - Produces a stable 0..N-1 `index` in the returned list.
    """
    if non_final is None:
        non_final = _is_non_final_tool_reply
    drop_names = drop_names or set()

    # Map call_id -> last tool reply message
    last_reply_by_call_id: Dict[str, dict] = {}
    for m in msgs or []:
        try:
            if m.get("role") != "tool":
                continue
            tcid = m.get("tool_call_id")
            if not isinstance(tcid, str) or not tcid:
                continue
            # Keep compatibility with typical call id shape used by providers
            if not str(tcid).startswith("call_"):
                continue
            if non_final(m):
                continue
            last_reply_by_call_id[tcid] = m
        except Exception:
            continue

    cleaned: List[CleanToolCall] = []
    for m in msgs or []:
        try:
            if m.get("role") != "assistant":
                continue
            tcs = m.get("tool_calls") or []
            for tc in tcs:
                try:
                    cid = tc.get("id")
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    args = fn.get("arguments", "{}")
                except Exception:
                    cid, name, args = None, None, "{}"
                if not isinstance(cid, str) or cid not in last_reply_by_call_id:
                    continue
                if isinstance(name, str) and name in drop_names:
                    continue
                cleaned.append(
                    {
                        "index": -1,
                        "name": name,
                        "arguments": args,
                        "result": last_reply_by_call_id[cid].get("content"),
                    },
                )
        except Exception:
            continue

    # Re-index
    for i, rec in enumerate(cleaned):
        rec["index"] = i

    return cleaned


__all__ = [
    "CleanToolCall",
    "extract_assistant_and_tool_steps",
    "extract_interjections",
    "extract_clarifications",
    "initial_user_from_user_visible_history",
    "build_clean_tool_trajectory",
]
