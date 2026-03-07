from __future__ import annotations

import json
import re

import unillm
from pydantic import BaseModel
from unity.common.llm_client import new_llm_client
from unity.common.token_utils import count_tokens
from unity.function_manager.execution_env import create_base_globals


def context_over_threshold(
    n_tokens: int,
    threshold: float,
    max_input_tokens: int,
) -> bool:
    return n_tokens >= max_input_tokens * threshold


COMPRESSION_PROMPT = (
    "You are a context compactor with surgical editing tools. You receive a "
    "conversation transcript as a sequence of JSON-serialized messages, each "
    "prefixed with an `[N]` index. You must compress it by selectively editing "
    "the most verbose entries.\n"
    "\n"
    "## Strategy\n"
    "- Focus on the LARGEST and most verbose entries first — large tool results, "
    "full tracebacks, verbose assistant reasoning, extended thinking blocks. "
    "These yield the most savings.\n"
    "- Leave concise messages untouched. They cost nothing.\n"
    "- You may issue multiple `update` calls per turn (they execute in parallel).\n"
    "- Each tool response includes the current token usage as a percentage of "
    "the context window. Use this to gauge progress and stop when usage is low "
    "enough.\n"
    "- When you have compressed enough, stop calling tools and respond with any "
    "short message to finish.\n"
    "\n"
    "## Compression Rules\n"
    '- Error messages and tracebacks: reduce to "error: <one-line cause>". '
    "Keep the traceback only if later messages reference it.\n"
    "- Tool results: keep only data that was actually used or referenced later. "
    "Discard decorative formatting, repeated schema keys, and unreferenced fields.\n"
    '- Image placeholders (e.g. "[2 image(s) provided]"): keep as-is.\n'
    "- Thinking blocks: compress extended reasoning to key conclusions and "
    "decisions. Remove exploratory tangents that didn't affect the outcome.\n"
    "- Assistant messages with tool_calls: compact to tool names and key arguments.\n"
    '- Narration-only assistant text ("Let me look that up"): replace with the '
    "tool call summary. Keep reasoning text only if it informs later steps.\n"
    '- System messages (role "system"): preserve instructions faithfully. '
    "The exceptions are parent chat context sections "
    '("Parent Chat Context" headers) and compressed prior context sections '
    '("Compressed Prior Context" headers) — both can be heavily pruned '
    "as they contain summaries, not instructions.\n"
    "- Every entry must remain non-empty after transformation.\n"
    "- Do NOT invent information that was not in the original message.\n"
    "\n"
    "## `update` Tool — Transformation Code\n"
    "The `transformation` argument is Python code that transforms the message. "
    "The variable `x` holds the full JSON string of the message. The final "
    "value of `x` after execution becomes the new entry.\n"
    "\n"
    "Patterns:\n"
    '- Surgical replace: `x = x.replace("verbose section", "summary")`\n'
    '- Regex: `x = re.sub(r"Traceback[\\s\\S]*?", "traceback omitted", x)`\n'
    "- Truncate: `x = x[:500]`\n"
    "- Parse and rebuild:\n"
    "  ```\n"
    "  msg = json.loads(x)\n"
    '  msg["content"] = "error: connection timeout"\n'
    "  x = json.dumps(msg)\n"
    "  ```\n"
    "- String operations work directly on the JSON string — no need to\n"
    "  parse/serialize for simple replacements.\n"
    "  (`re` and `json` are available in the execution environment)"
)

COMPRESSION_MULTI_PASS_ADDENDUM = (
    "\n\n## Multi-Pass Compression\n"
    "Some entries are from prior compression passes (already compressed once or "
    "more). You have a `get_raw` tool to retrieve the original uncompressed "
    "content of any entry.\n"
    "\n"
    "- Prior entries may still be verbose or contain information now irrelevant "
    "given later context.\n"
    "- Use `get_raw(index)` to inspect the original before deciding whether to "
    "re-compress.\n"
    "- You can `update` any entry (prior or new) using the same transformation "
    "rules.\n"
    "- If an old compression lost important detail, use `get_raw` to recover it "
    "and write a better summary."
)


# ── Sentinel returned by the loop when compression is requested ──────────────

_COMPRESSION_SIGNAL = object()


# ── Marker tool exposed to the loop LLM ─────────────────────────────────────


def compress_context() -> str:
    """Compress the conversation history to free up context window space.

    This tool is always visible. When the context window is nearly full, this
    becomes the **only** available tool and you **must** call it.

    Cannot be called while other tools are in-flight -- finish or stop all
    running tools first.
    """
    return "compression acknowledged"


def prepare_messages_for_compression(messages: list[dict]) -> list[dict]:
    """Strip expensive binary content before sending to the compression LLM.

    Replaces image blocks (``image`` / ``image_url``) with text placeholders.
    All other content (including thinking blocks) is preserved.
    """
    result: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            result.append(msg)
            continue

        image_count = 0
        new_blocks: list[dict] = []
        for block in content:
            btype = block.get("type", "")
            if btype in ("image", "image_url"):
                image_count += 1
            else:
                new_blocks.append(block)

        if image_count:
            label = f"[{image_count} image(s) provided]"
            new_blocks.append({"type": "text", "text": label})

        new_msg = {k: v for k, v in msg.items() if k != "content"}
        new_msg["content"] = new_blocks
        result.append(new_msg)

    return result


class CompressedMessage(BaseModel):
    content: str


class CompressedMessages(BaseModel):
    messages: list[CompressedMessage]


_SANDBOX_GLOBALS = None


def _get_sandbox_globals() -> dict:
    """Lazy-init sandbox globals for transformation exec.

    Extends ``create_base_globals()`` with ``re`` and ``json`` for
    regex and JSON operations. The result is cached at module level
    for reuse.
    """
    global _SANDBOX_GLOBALS
    if _SANDBOX_GLOBALS is None:
        g = create_base_globals()
        g["re"] = re
        g["json"] = json
        _SANDBOX_GLOBALS = g
    return _SANDBOX_GLOBALS


def _eval_transformation(transformation_str: str, content: str) -> str:
    """Execute transformation code against message content.

    The code receives the current content as ``x`` and must leave the
    transformed result in ``x`` after execution.  Single expressions
    (e.g. ``x.replace("old", "new")``) are auto-assigned to ``x``.

    Returns the transformed content string.  Raises on syntax errors
    or runtime errors (caller should catch and report to the LLM).
    """
    sandbox = dict(_get_sandbox_globals())
    sandbox["x"] = content

    code = transformation_str.strip()

    # Support bare expressions: if the code is a single expression with no
    # assignment, wrap it as ``x = <expr>`` so the user doesn't need to
    # write the boilerplate.  Multi-line code or code containing ``=``
    # is executed as-is and expected to mutate ``x`` directly.
    if "\n" not in code and "=" not in code:
        try:
            compile(code, "<transformation>", "eval")
            code = f"x = {code}"
        except SyntaxError:
            pass

    exec(code, sandbox)  # noqa: S102
    result = sandbox["x"]
    if not isinstance(result, str):
        result = str(result)
    return result


def _compute_token_usage(entries: dict[int, str], endpoint: str) -> str:
    """Compute token usage of all entries as a percentage of the context window."""
    total_text = "\n".join(entries.values())
    tokens = count_tokens(total_text)
    max_input = unillm.get_max_input_tokens(endpoint) or 0
    if max_input > 0:
        pct = tokens / max_input * 100
        return f"[{tokens:,} tokens | {pct:.1f}% of context window]"
    return f"[{tokens:,} tokens]"


def _make_update_tool(entries: dict[int, str], endpoint: str) -> callable:
    """Build the ``update`` tool closure over a mutable entries dict."""

    def update(index: int, transformation: str) -> str:
        """Transform a message to compress it in-place.

        Args:
            index: The [N] index of the message to transform.
            transformation: Python code that transforms the message.
                The variable ``x`` holds the full JSON string of the
                message. The final value of ``x`` after execution becomes
                the new entry.

                String operations (work directly on the JSON string):
                  x = x.replace("verbose section", "summary")
                  x = re.sub(r"Traceback[\\s\\S]*", "omitted", x)
                  x = x[:500]

                Structured operations (parse, modify, serialize):
                  msg = json.loads(x)
                  msg["content"] = "error: timeout"
                  x = json.dumps(msg)

                A bare expression (no ``=``) is auto-assigned to ``x``:
                  x.replace("verbose", "short")

                ``re`` and ``json`` are available.

        Returns:
            The new entry after transformation, followed by current
            token usage of the compressed context.
        """
        if index not in entries:
            valid = sorted(entries.keys())
            return f"Error: index {index} not found. Valid indices: {valid}"
        try:
            new_content = _eval_transformation(
                transformation,
                entries[index],
            )
        except Exception as exc:
            return f"Error applying transformation: {type(exc).__name__}: {exc}"
        if not new_content:
            new_content = "(empty)"
        entries[index] = new_content
        usage = _compute_token_usage(entries, endpoint)
        return f"{new_content}\n{usage}"

    return update


def _make_get_raw_tool(raw_archives: list[list[dict]]) -> callable:
    """Build the ``get_raw`` tool closure over raw message archives."""

    def get_raw(index: int, n: int = 1) -> str:
        """Retrieve the original uncompressed content of a message.

        Use this to inspect what a previously-compressed entry looked
        like before compression.  Helpful when re-evaluating whether
        an old compression decision was too aggressive or when the
        conversation focus has shifted.

        Args:
            index: The [N] index of the message.
            n: Number of consecutive messages to retrieve (default 1).
        """
        flat = [msg for archive in raw_archives for msg in archive]
        if index < 0 or index >= len(flat):
            return json.dumps(
                {"error": f"Index {index} out of range (0-{len(flat) - 1})"},
            )
        end = min(index + n, len(flat))
        return json.dumps(flat[index:end], default=str)

    return get_raw


async def compress_messages(
    messages: list[dict],
    endpoint: str,
    *,
    prior_entries: list[tuple[int, str]] | None = None,
    raw_archives: list[list[dict]] | None = None,
    new_indices: list[int] | None = None,
) -> CompressedMessages:
    """Compact a message list using a tool-loop that surgically edits entries.

    1. Strips images via ``prepare_messages_for_compression``
    2. JSON-serializes each message into indexed ``[N] {json}`` format
    3. Starts a short-lived async tool loop where the LLM calls ``update``
       to compress individual entries via Python transformations
    4. Wraps the mutated entries as ``CompressedMessages``

    For multi-pass compression, ``prior_entries`` provides already-compressed
    entries from previous passes (as ``(global_index, content)`` tuples),
    ``raw_archives`` enables the ``get_raw`` tool for peeking at originals,
    and ``new_indices`` maps each message to its global archive index.
    """
    from unity.common.async_tool_loop import start_async_tool_loop

    prepared = prepare_messages_for_compression(messages)

    if new_indices is not None and len(new_indices) != len(prepared):
        raise ValueError(
            f"new_indices length ({len(new_indices)}) must match "
            f"messages length ({len(prepared)})",
        )

    # Build entries dict: prior entries + new messages
    entries: dict[int, str] = {}
    if prior_entries:
        for idx, content in prior_entries:
            entries[idx] = content

    indices = new_indices if new_indices is not None else list(range(len(prepared)))
    for i, msg in enumerate(prepared):
        entries[indices[i]] = json.dumps(msg, default=str)

    serialized = "\n\n".join(f"[{idx}] {e}" for idx, e in sorted(entries.items()))

    update_tool = _make_update_tool(entries, endpoint)
    tools: dict[str, callable] = {"update": update_tool}

    if prior_entries and raw_archives:
        tools["get_raw"] = _make_get_raw_tool(raw_archives)

    initial_usage = _compute_token_usage(entries, endpoint)

    if prior_entries:
        prior_indices = sorted(idx for idx, _ in prior_entries)
        new_msg_indices = sorted(indices)
        user_prompt = (
            f"Compress the following transcript. "
            f"Entries [{prior_indices[0]}]-[{prior_indices[-1]}] are from "
            f"prior passes (already compressed). "
            f"Entries [{new_msg_indices[0]}]-[{new_msg_indices[-1]}] are "
            f"from the current session.\n"
            f"Focus on the largest entries first.\n"
            f"Current usage: {initial_usage}\n\n"
            f"{serialized}"
        )
    else:
        user_prompt = (
            f"Compress the following {len(messages)} message transcript. "
            f"Focus on the largest entries first.\n"
            f"Current usage: {initial_usage}\n\n"
            f"{serialized}"
        )

    sys_prompt = COMPRESSION_PROMPT
    if prior_entries:
        sys_prompt += COMPRESSION_MULTI_PASS_ADDENDUM

    client = new_llm_client(endpoint, origin="compress_messages")
    client.set_system_message(sys_prompt)

    handle = start_async_tool_loop(
        client,
        user_prompt,
        tools,
        loop_id="compress_messages",
        max_steps=50,
        timeout=120,
        log_steps=True,
        enable_compression=False,
    )

    await handle.result()

    return CompressedMessages(
        messages=[
            CompressedMessage(content=entries[idx]) for idx in sorted(entries.keys())
        ],
    )


def render_compressed_context(
    compressed: CompressedMessages,
    index_offset: int = 0,
) -> str:
    """Render compressed messages into compact indexed lines.

    Returns only the message lines (no header).  The caller is responsible
    for prepending ``## Compressed Prior Context`` when assembling the
    final system-message block.

    ``index_offset`` shifts every ``[N]`` label so that multiple renders
    can be concatenated with continuous numbering.
    """
    lines: list[str] = []
    for i, msg in enumerate(compressed.messages):
        lines.append(f"[{i + index_offset}] {msg.content}")
    return "\n".join(lines)
