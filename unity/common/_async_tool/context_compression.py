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
    "The exception is parent chat context sections — look for "
    '"Parent Chat Context" headers — which can be heavily pruned '
    "if they contain information redundant with the conversation.\n"
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


def _compute_token_usage(entries: list[str], endpoint: str) -> str:
    """Compute token usage of all entries as a percentage of the context window."""
    total_text = "\n".join(entries)
    tokens = count_tokens(total_text)
    max_input = unillm.get_max_input_tokens(endpoint) or 0
    if max_input > 0:
        pct = tokens / max_input * 100
        return f"[{tokens:,} tokens | {pct:.1f}% of context window]"
    return f"[{tokens:,} tokens]"


def _make_update_tool(entries: list[str], endpoint: str) -> callable:
    """Build the ``update`` tool closure over a mutable entries list."""

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
        if index < 0 or index >= len(entries):
            return f"Error: index {index} out of range (0-{len(entries) - 1})"
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


async def compress_messages(
    messages: list[dict],
    endpoint: str,
) -> CompressedMessages:
    """Compact a message list using a tool-loop that surgically edits entries.

    1. Strips images via ``prepare_messages_for_compression``
    2. JSON-serializes each message into indexed ``[N] {json}`` format
    3. Starts a short-lived async tool loop where the LLM calls ``update``
       to compress individual entries via Python transformations
    4. Wraps the mutated entries as ``CompressedMessages``
    """
    from unity.common.async_tool_loop import start_async_tool_loop

    prepared = prepare_messages_for_compression(messages)
    entries = [json.dumps(msg, default=str) for msg in prepared]
    serialized = "\n\n".join(f"[{i}] {e}" for i, e in enumerate(entries))

    update_tool = _make_update_tool(entries, endpoint)
    tools = {"update": update_tool}

    initial_usage = _compute_token_usage(entries, endpoint)
    user_prompt = (
        f"Compress the following {len(messages)} message transcript. "
        f"Focus on the largest entries first.\n"
        f"Current usage: {initial_usage}\n\n"
        f"{serialized}"
    )

    client = new_llm_client(endpoint, origin="compress_messages")
    client.set_system_message(COMPRESSION_PROMPT)

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
        messages=[CompressedMessage(content=e) for e in entries],
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
