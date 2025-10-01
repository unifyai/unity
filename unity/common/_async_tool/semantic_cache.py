import unify
import json
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, TypedDict
from pydantic import BaseModel

if TYPE_CHECKING:
    from unity.common._async_tool.tools_data import ToolsData

from .tools_data import create_tool_call_message
from ..semantic_search import escape_single_quotes

_USER_MESSAGE_EMBEDDING_FIELD_NAME = "user_message_emb"


@dataclass
class SemanticCacheResult:
    original_user_message: str
    closest_user_message: str
    tool_trajectory: list[dict]


class _Config:
    def __init__(
        self,
        context: str = "",
        threshold: float = 0.2,
        top_k: int = 1,
        embedding_model: str = "text-embedding-3-small",
        model: str = "gpt-4o@openai",
    ):
        if not context:
            from unity import ASSISTANT_CONTEXT

            context = f"{ASSISTANT_CONTEXT}/Cache"

        self._context = context
        self._threshold = threshold
        self._top_k = top_k
        self._embedding_model = embedding_model
        self._model = model

    @property
    def context(self):
        return self._context

    @property
    def threshold(self):
        return self._threshold

    @property
    def top_k(self):
        return self._top_k

    @property
    def embedding_model(self):
        return self._embedding_model

    @property
    def model(self):
        return self._model


_CONFIG = _Config()


# Dummy tool placeholder (used by async_tool_loop_inner)
def semantic_search(user_message: str):
    """
    Search a semantic cache for prior solutions relevant to the given user_message.

    Workflow:
    1. Performs a semantic search over cached user messages using embeddings keyed by the
       provided user_message.
    2. Returns the ordered tool trajectory previously used to answer a similar message.
       For each step, the result_status is either "new" or "cached", and the original order is preserved.
    3. If the result_status is "new", the tool is re-executed with the stored arguments to compute
       fresh results (avoiding stale data), and the original order is preserved.
    4. If the result_status is "cached", the tool is not re-executed, and the result is the cached result from previous run.

    Usage guidance:
    - Prefer these returned results over issuing new tool calls for the same purpose.
    - The tools may or may not fully resolve the request; use judgment to synthesize an
      answer from these results before deciding to call additional tools.

    Input: user_message (str)
    Output: List[dict] with entries of the form {"name", "arguments", "result", "result_status"}, in execution order.
    """


def get_hint():
    return """
    You have access to a best-effort semantic cache of prior tool trajectories.
    For the current user message, the 'semantic_search' tool may return the set of tools that were previously called to answer a similar query, along with already-computed results.

    Guidance:
    - Prefer using those returned results directly when the status is "new", or follow a similar sequence of tools with updated arguments if required, before creating new or unrelated tool calls.
    - Treat them as latency-saving precomputed outputs to avoid redundant work and reduce time.
    - If gaps remain or the results do not fully address the request, call only the minimal additional tools needed.

    Rules:
    - Do not call the 'semantic_search' tool again, it is only used to get the tool trajectory.
    - Do not call tools that are in the tool trajectory, they are already executed with the same arguments and has the "result_status":"new".
    - Do NOT state, hint, or imply that you are using a cache, 'semantic_search', prior runs, or precomputed results.
    - Write answers as if you executed the necessary tools now.
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
                tool_call["result_status"] = "new"
            except Exception:
                tool_call["result_status"] = "cached"
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
Task: From the conversation history, return the final intended user message.

Rules:
- Apply all user interjections/corrections; the latest user message overrides earlier ones.
- Ignore assistant messages; they are never part of the output.
- Output exactly one plain string: the final corrected user message. No quotes, JSON, or explanation.
- Do not add new information. Remove redundant or off-topic words.

Example:
Input:
[
"user: Hi, what is the weather in Tokyo?",
"user: Actually, I meant in Cairo"
]
Output:
Hi, what is the weather in Cairo?
"""

    global _CONFIG
    client = unify.AsyncUnify(_CONFIG.model)
    client.set_system_message(CLEAN_USER_MESSAGE_PROMPT)
    res = await client.generate(
        user_message=f"Messages: {json.dumps(messages_history)}",
    )
    return res


async def clean_tool_trajectory(user_message, msgs, previous_tool_trajectory=None):
    class ToolRequestPair(TypedDict):
        request: Mapping[str, Any]
        response: Mapping[str, Any]

    class PruneToolsResponseFormat(BaseModel):
        indices: list[int]

    global _CONFIG

    cleaned_trajectory = []
    if previous_tool_trajectory:
        cleaned_trajectory.extend(previous_tool_trajectory)
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

                    request = tool_call
                    request.pop("id")

                    response = _flatten_tools[id]
                    response.pop("tool_call_id")

                    pair = ToolRequestPair(
                        request=request,
                        response=response,
                    )
                    cleaned_trajectory.append(pair)

    client = unify.AsyncUnify(_CONFIG.model)
    client.set_system_message(
        """
        You are a helpful assistant that cleans redundant tool calls, given a user query and a list of tool calls,
        you should return indicies of the tool calls to prune, that are redundant/duplicate or not relevant to the user query.
        """,
    )
    res = await client.generate(
        user_message=f"User query: {user_message}\nTool trajectory: {json.dumps(cleaned_trajectory, indent=2)}",
        response_format=PruneToolsResponseFormat,
    )

    res = PruneToolsResponseFormat.model_validate_json(res)

    cleaned_trajectory = [
        tool_call_pair
        for idx, tool_call_pair in enumerate(cleaned_trajectory)
        if idx not in res.indices
    ]

    return cleaned_trajectory


def store_tool_trajectory(user_message, tool_trajectory):
    global _CONFIG
    store_context = _CONFIG.context

    # Ensure context exists
    context_exist = store_context in unify.get_contexts(prefix=store_context)
    if not context_exist:
        unify.create_context(store_context)

    log_id = unify.log(
        context=store_context,
        user_message=user_message,
        tool_trajectory=json.dumps(tool_trajectory),
    )

    embed_expr = f"embed({{logs:user_message}}, model='{_CONFIG.embedding_model}')"
    unify.create_derived_logs(
        context=store_context,
        key=_USER_MESSAGE_EMBEDDING_FIELD_NAME,
        equation=embed_expr,
        referenced_logs={"logs": [log_id.id]},
    )

    return log_id


def get_tool_trajectory(user_message):
    global _CONFIG
    store_context = _CONFIG.context

    # Ensure context exists
    context_exist = store_context in unify.get_contexts(prefix=store_context)
    if not context_exist:
        unify.create_context(store_context)

    logs = unify.get_logs(
        context=store_context,
        exclude_fields=["user_message_emb"],
        filter=f"cosine(user_message, embed('{escape_single_quotes(user_message)}', model='{_CONFIG.embedding_model}')) < {_CONFIG.threshold}",
        sorting={
            f"cosine(user_message, embed('{escape_single_quotes(user_message)}', model='{_CONFIG.embedding_model}'))": "descending",
        },
        limit=_CONFIG.top_k,
    )

    if logs:
        return SemanticCacheResult(
            original_user_message=user_message,
            closest_user_message=logs[0].entries["user_message"],
            tool_trajectory=json.loads(logs[0].entries["tool_trajectory"]),
        )

    return None
