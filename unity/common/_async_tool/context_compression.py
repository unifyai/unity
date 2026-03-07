from __future__ import annotations

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
    "conversation transcript and must compress it by selectively editing the "
    "most verbose entries.\n"
    "\n"
    "## Strategy\n"
    "- Focus on the LARGEST and most verbose entries first — large tool results, "
    "full tracebacks, verbose assistant reasoning. These yield the most savings.\n"
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
    "- Assistant messages with tool_calls: compact to tool names and key arguments, "
    'e.g. search(name="John"), filter(field="city").\n'
    '- Narration-only assistant text ("Let me look that up"): replace with the '
    "tool call summary. Keep reasoning text only if it informs later steps.\n"
    '- Every entry must remain non-empty after transformation. Use "ok" for '
    'acknowledgements, "error" for failures.\n'
    "- Do NOT invent information that was not in the original message.\n"
    "\n"
    "## `update` Tool — Transformation Code\n"
    "The `transformation` argument is Python code that transforms the message "
    "content. The variable `x` holds the current content string. The final "
    "value of `x` after execution becomes the new content.\n"
    "\n"
    "Patterns:\n"
    '- Overwrite: `x = "error: connection timeout"`\n'
    '- Surgical replace: `x = x.replace("verbose section", "summary")`\n'
    '- Keep first line: `x = x.split("\\n")[0]`\n'
    '- Regex: `x = re.sub(r"Traceback[\\s\\S]*", "traceback omitted", x)`\n'
    "  (`re` is available in the execution environment)\n"
    "- Truncate: `x = x[:200]`\n"
    "- Multi-line logic:\n"
    "  ```\n"
    "  lines = x.split('\\n')\n"
    "  x = '\\n'.join(line for line in lines if 'ERROR' not in line)\n"
    "  ```"
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

    Handles two categories:
    1. Image blocks (``image`` / ``image_url``) -> ``[N image(s) provided]``
    2. Thinking blocks -> removed entirely
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
            elif btype == "thinking":
                continue
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
    role: str
    content: str


class CompressedMessages(BaseModel):
    messages: list[CompressedMessage]


def _serialize_messages_for_prompt(messages: list[dict]) -> str:
    """Serialize messages into a compact text representation for the LLM."""
    lines: list[str] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")

        parts: list[str] = []
        if content is not None:
            if isinstance(content, list):
                text_parts = [
                    b.get("text", "") for b in content if b.get("type") == "text"
                ]
                parts.append(" ".join(text_parts))
            elif isinstance(content, str):
                parts.append(content)

        if tool_calls:
            tc_strs: list[str] = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                short_id = tc.get("id", "")[-6:]
                if short_id:
                    tc_strs.append(f"{name}[{short_id}]({args})")
                else:
                    tc_strs.append(f"{name}({args})")
            parts.append("[tool_calls: " + ", ".join(tc_strs) + "]")

        extra = ""
        if "tool_call_id" in msg:
            short_id = msg["tool_call_id"][-6:]
            extra = f":{short_id}" if short_id else ""

        lines.append(
            f"[{i}] [{role}{extra}]: {' | '.join(parts) if parts else '(empty)'}",
        )

    return "\n".join(lines)


_ENTRY_RE = re.compile(r"^\[(\d+)\]\s+\[([^\]]*)\]:\s?(.*)", re.DOTALL)


def _parse_serialized_entries(serialized: str) -> list[dict]:
    """Parse the output of ``_serialize_messages_for_prompt`` into mutable entries.

    Each entry is ``{"role": str, "content": str}`` matching the
    ``CompressedMessage`` schema so the list can be directly wrapped
    into ``CompressedMessages`` after mutations.
    """
    entries: list[dict] = []
    for line in serialized.split("\n"):
        m = _ENTRY_RE.match(line)
        if m:
            role_raw = m.group(2)
            # Strip tool_call_id suffix (e.g. "tool:abc123" → "tool")
            role = role_raw.split(":")[0] if ":" in role_raw else role_raw
            entries.append({"role": role, "content": m.group(3)})
    return entries


_SANDBOX_GLOBALS = None


def _get_sandbox_globals() -> dict:
    """Lazy-init sandbox globals for transformation exec.

    Extends ``create_base_globals()`` with ``re`` for regex support.
    The result is cached at module level for reuse.
    """
    global _SANDBOX_GLOBALS
    if _SANDBOX_GLOBALS is None:
        g = create_base_globals()
        g["re"] = re
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


def _compute_token_usage(entries: list[dict], endpoint: str) -> str:
    """Compute token usage of all entry content as a percentage of the context window."""
    total_text = "\n".join(e["content"] for e in entries)
    tokens = count_tokens(total_text)
    max_input = unillm.get_max_input_tokens(endpoint) or 0
    if max_input > 0:
        pct = tokens / max_input * 100
        return f"[{tokens:,} tokens | {pct:.1f}% of context window]"
    return f"[{tokens:,} tokens]"


def _make_update_tool(entries: list[dict], endpoint: str) -> callable:
    """Build the ``update`` tool closure over a mutable entries list."""

    def update(index: int, transformation: str) -> str:
        """Transform a message to compress it in-place.

        Args:
            index: The [N] index of the message to transform.
            transformation: Python code that transforms the message content.
                The variable ``x`` holds the current content string. The
                final value of ``x`` after execution becomes the new content.

                Single-line examples:
                  x = "error: timeout"
                  x = x.replace("verbose section", "summary")
                  x = x.split("\\n")[0]
                  x = re.sub(r"Traceback[\\s\\S]*", "omitted", x)
                  x = x[:200]

                A bare expression (no ``=``) is auto-assigned to ``x``:
                  x.replace("verbose", "short")

                Multi-line code (must assign to ``x``):
                  lines = x.split("\\n")
                  x = "\\n".join(l for l in lines if "ERROR" not in l)

                ``re`` is available for regex operations.

        Returns:
            The new content after transformation, followed by current
            token usage of the compressed context.
        """
        if index < 0 or index >= len(entries):
            return f"Error: index {index} out of range (0-{len(entries) - 1})"
        try:
            new_content = _eval_transformation(
                transformation,
                entries[index]["content"],
            )
        except Exception as exc:
            return f"Error applying transformation: {type(exc).__name__}: {exc}"
        if not new_content:
            new_content = "(empty)"
        entries[index]["content"] = new_content
        usage = _compute_token_usage(entries, endpoint)
        return f"{new_content}\n{usage}"

    return update


async def compress_messages(
    messages: list[dict],
    endpoint: str,
) -> CompressedMessages:
    """Compact a message list using a tool-loop that surgically edits entries.

    1. Strips images and thinking blocks via ``prepare_messages_for_compression``
    2. Serializes into indexed ``[N] [role]: content`` format
    3. Starts a short-lived async tool loop where the LLM calls ``update``
       to compress individual entries via lambda transformations
    4. Wraps the mutated entries as ``CompressedMessages``
    """
    from unity.common.async_tool_loop import start_async_tool_loop

    prepared = prepare_messages_for_compression(messages)
    serialized = _serialize_messages_for_prompt(prepared)
    entries = _parse_serialized_entries(serialized)

    if len(entries) != len(messages):
        raise ValueError(
            f"Serialization produced {len(entries)} entries but expected "
            f"{len(messages)} — _serialize_messages_for_prompt / "
            f"_parse_serialized_entries mismatch",
        )

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
        messages=[
            CompressedMessage(role=e["role"], content=e["content"]) for e in entries
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
        lines.append(f"[{i + index_offset}] [{msg.role}]: {msg.content}")
    return "\n".join(lines)
