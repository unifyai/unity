import unify
import json

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unity.common._async_tool.tools_data import ToolsData

from .tools_data import create_tool_call_message
from ..semantic_search import escape_single_quotes

_USER_MESSAGE_EMBEDDING_FIELD_NAME = "user_message_emb"
_EMBED_MODEL = "text-embedding-3-small"


# Dummy tool placeholder (used by async_tool_loop_inner)
def semantic_search(user_message):
    """
    Retrieve and return a closest match of the user request if any, this includes tools called and their newly computed results,
    Prefer using tool results instead of creating new tool calls
    """


def get_hint():
    return """
    Prefer using tool results from 'semantic_search' tool instead of creating new tool calls, these tool_calls are pre-computed and the results are fresh, you should prefer using them over creating new tool calls to reduce time, do not inform the user that you already have
    the results, respond as you DID call it.
    """


def get_dummy_tool(user_query, closest_match, tools: "ToolsData"):
    history = json.loads(closest_match.entries["tool_trajectory"])
    for tool_call in history:
        if (tool_name := tool_call.get("name")) in tools.normalized:
            try:
                args = json.loads(tool_call.get("arguments"))
                tool_call["result"] = tools.normalized[tool_name].fn(**args)
            except Exception as e:
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
                    "arguments": f"{user_query}",
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
    cleaned_trajectory = []
    _flatten_tools = {
        msg.get("tool_call_id"): {"content": msg["content"], "name": msg["name"]}
        for msg in msgs
        if msg.get("role") == "tool"
    }

    for msg in msgs:
        if msg.get("role") != "assistant":
            continue

        if msg.get("tool_calls") is not None:
            for tool_call in msg.get("tool_calls"):
                if (id := tool_call.get("id")) in _flatten_tools.keys():
                    tool_call_content = _flatten_tools[id].get("content")
                    new_tool_call = {
                        "name": _flatten_tools[id].get("name"),
                        "arguments": tool_call.get("function", {}).get("arguments"),
                        "result": tool_call_content,
                    }
                    cleaned_trajectory.append(new_tool_call)

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
        return logs[0]

    return None
