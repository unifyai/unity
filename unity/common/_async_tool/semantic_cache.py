import unify
import json
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, TypedDict


if TYPE_CHECKING:
    from unity.common._async_tool.tools_data import ToolsData

from .tools_data import create_tool_call_message
from ..semantic_search import escape_single_quotes

_USER_MESSAGE_EMBEDDING_FIELD_NAME = "user_message_emb"
_EMBED_MODEL = "text-embedding-3-small"


@dataclass
class SemanticCacheResult:
    original_user_message: str
    closest_match: str
    tool_trajectory: list[dict]


# Dummy tool placeholder (used by async_tool_loop_inner)
def semantic_search(user_message: str):
    """
    Search a semantic cache for prior solutions relevant to the given user_message.

    Workflow:
    1. Performs a semantic search over cached user messages using embeddings keyed by the
       provided user_message.
    2. Returns the ordered tool trajectory previously used to answer a similar message.
       For each step, the tool is re-executed with the stored arguments to compute
       fresh results (avoiding stale data), and the original order is preserved.

    Usage guidance:
    - Prefer these returned results over issuing new tool calls for the same purpose.
    - The tools may or may not fully resolve the request; use judgment to synthesize an
      answer from these results before deciding to call additional tools.

    Input: user_message (str)
    Output: List[dict] with entries of the form {"name", "arguments", "result"},
            in execution order.
    """


def get_hint():
    return """
    Prefer using tool results from 'semantic_search' tool instead of creating new tool calls, these tool_calls are pre-computed and the results are fresh, you should prefer using them over creating new tool calls to reduce time, do not inform the user that you already have
    the results, respond as you DID call it.
    """


def _simplify_tool_trajectory(tool_trajectory: list[dict]):
    _simplified_trajectory = []
    for tool_call_pair in tool_trajectory:
        name = tool_call_pair["request"]["function"]["name"]
        arguments = tool_call_pair["request"]["function"]["arguments"]
        result = tool_call_pair["response"]["content"]

        _simplified_trajectory.append(
            {
                "name": name,
                "arguments": arguments,
                "result": result,
            },
        )

    return _simplified_trajectory


async def get_dummy_tool(
    semantic_cache_result: SemanticCacheResult,
    tools: "ToolsData",
):
    history = _simplify_tool_trajectory(semantic_cache_result.tool_trajectory)
    for tool_call in history:
        if (tool_name := tool_call.get("name")) in tools.normalized:
            # TODO use ThreadPoolExecutor to run the tool calls in parallel
            try:
                args = json.loads(tool_call.get("arguments"))
                if inspect.iscoroutinefunction(tools.normalized[tool_name].fn):
                    tool_call["result"] = await tools.normalized[tool_name].fn(**args)
                else:
                    tool_call["result"] = tools.normalized[tool_name].fn(**args)
            except Exception:
                continue

    call_id = f"call_MyCall"
    dummy_tool_call = {
        "content": None,
        "refusal": None,
        "role": "assistant",
        "annotations": [],
        "audio": None,
        "function_call": None,
        "tool_calls": [
            {
                "id": call_id,
                "function": {
                    "arguments": f"{semantic_cache_result.original_user_message}",
                    "name": "semantic_search",
                },
                "type": "function",
            },
        ],
    }
    msg = create_tool_call_message(
        name="semantic_search",
        call_id=call_id,
        content=json.dumps(history, indent=4),
    )
    return [
        dummy_tool_call,
        msg,
    ]


async def construct_new_user_message(init_user_message, messages_history):
    if not messages_history:
        return init_user_message

    CLEAN_USER_MESSAGE_PROMPT = """
You are given a list of messages representing a conversation history.  Your only task is:

Reconstruct the final intended user message by applying all interjections and corrections to the initial user message(s).

Output only the final, corrected user message as a plain string (not JSON, not explanation) You should not include
any redundant words or phrases that are not related to the user intended request.

Each message has the format:

"role": "message"
"role" can be either "user" or "assistant".
"message" is the text of the message.

Some conversations may include interjections. An interjection is initiated by a "user" message.

Example:

Input messages:
[
"user: Hi, what is the weather in Tokyo?",
"user: Actually, I meant in Cairo"
]

Output:
Hi, what is the weather in Cairo?
"""

    client = unify.AsyncUnify("gpt-4o@openai")
    client.set_system_message(CLEAN_USER_MESSAGE_PROMPT)
    res = await client.generate(
        user_message=f"Messages: {json.dumps(messages_history)}",
    )
    return res


def clean_tool_trajectory(msgs):

    class ToolRequestPair(TypedDict):
        request: Mapping[str, Any]
        response: Mapping[str, Any]

    cleaned_trajectory = []
    _flatten_tools = {
        msg.get("tool_call_id"): msg for msg in msgs if msg.get("role") == "tool"
    }

    for msg in msgs:
        if msg.get("role") != "assistant":
            continue

        if msg.get("tool_calls") is not None:
            for tool_call in msg.get("tool_calls"):
                if (id := tool_call.get("id")) in _flatten_tools.keys():
                    if _flatten_tools[id].get("name") == "semantic_search":
                        continue
                    pair = ToolRequestPair(
                        request=tool_call,
                        response=_flatten_tools[id],
                    )
                    cleaned_trajectory.append(pair)

    return cleaned_trajectory


def store_tool_trajectory(user_message, tool_trajectory):
    from unity import ASSISTANT_CONTEXT

    store_context = f"{ASSISTANT_CONTEXT}/Cache"

    # Ensure context exists
    context_exist = store_context in unify.get_contexts(prefix=store_context)
    if not context_exist:
        unify.create_context(store_context)

    log_id = unify.log(
        context=store_context,
        user_message=user_message,
        tool_trajectory=json.dumps(tool_trajectory),
    )

    embed_expr = f"embed({{logs:user_message}}, model='{_EMBED_MODEL}')"
    unify.create_derived_logs(
        context=store_context,
        key=_USER_MESSAGE_EMBEDDING_FIELD_NAME,
        equation=embed_expr,
        referenced_logs={"logs": [log_id.id]},
    )

    return log_id


def get_tool_trajectory(user_message):
    from unity import ASSISTANT_CONTEXT

    store_context = f"{ASSISTANT_CONTEXT}/Cache"

    # Ensure context exists
    context_exist = store_context in unify.get_contexts(prefix=store_context)
    if not context_exist:
        unify.create_context(store_context)

    threshold = 0.2
    limit = 1
    logs = unify.get_logs(
        context=store_context,
        exclude_fields=["user_message_emb"],
        filter=f"cosine(user_message, embed('{escape_single_quotes(user_message)}', model='{_EMBED_MODEL}')) < {threshold}",
        sorting={
            f"cosine(user_message, embed('{escape_single_quotes(user_message)}', model='{_EMBED_MODEL}'))": "descending",
        },
        limit=limit,
    )

    if logs:
        return SemanticCacheResult(
            original_user_message=user_message,
            closest_match=logs[0].entries["user_message"],
            tool_trajectory=json.loads(logs[0].entries["tool_trajectory"]),
        )

    return None
