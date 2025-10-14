import unify
import json
import inspect
import copy
import asyncio
import functools
import atexit
import logging

from dataclasses import dataclass
import threading
from typing import Any, Mapping, TypedDict, Callable
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, wait

from unity.common.tool_spec import ToolSpec

from .tools_data import create_tool_call_message
from ..semantic_search import escape_single_quotes
from ..llm_helpers import _dumps

_USER_MESSAGE_EMBEDDING_FIELD_NAME = "_user_message_emb"
logger = logging.getLogger(__name__)


class _SemanticCacheSaver:
    """
    A singleton semantic cache task handler that manages saving tasks in a thread pool.
    Automatically cleans up all saving tasks before application exit.
    """

    _instance = None
    _executor = None
    _futures = []
    _future_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_workers: int = 4):
        # Only initialize once
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
            atexit.register(self._cleanup)
            logger.info(
                f"SemanticCacheSaver initialized with {max_workers} workers",
            )

    def _submit(self, fn: Callable, *args, **kwargs) -> Any:
        future = self._executor.submit(fn, *args, **kwargs)
        with self._future_lock:
            self._futures.append(future)
        future.add_done_callback(self._task_done_callback)
        return future

    def _task_done_callback(self, future):
        with self._future_lock:
            self._futures.remove(future)
        try:
            future.result()
        except Exception as e:
            logger.error(f"Saving failed with error: {e}", exc_info=True)

    def _cleanup(self):
        """
        Cleanup method called automatically at exit. Waits for all pending saving tasks to complete.
        """
        logger.info("Shutting down SemanticCacheSaver...")
        self._executor.shutdown(wait=True)

    def _save_to_cache(self, store_context, namespace, user_message, tool_trajectory):
        # store_context is captured at call-time to avoid thread-local context loss
        # Ensure context exists
        context_exist = store_context in unify.get_contexts(prefix=store_context)
        if not context_exist:
            unify.create_context(store_context)

        log_id = unify.log(
            context=store_context,
            user_message=user_message,
            namespace=namespace,
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

    def _construct_new_user_message(
        self,
        init_user_message,
        messages_history,
        full_messages_history,
    ):
        history = None
        if not messages_history:
            history = init_user_message
        else:
            history = messages_history

        _user_clarifications = {}
        for msg in full_messages_history:
            if msg.get("role") == "tool" and msg.get("name").startswith(
                "request_clarification",
            ):
                _user_clarifications[msg["tool_call_id"]] = {
                    "assistant_question": "",
                    "user_answer": msg["content"],
                }

        for msg in full_messages_history:
            if msg.get("role") == "assistant" and msg.get("tool_calls") is not None:
                for tool_call in msg.get("tool_calls"):
                    if (id := tool_call["id"]) in _user_clarifications.keys():
                        args = json.loads(tool_call["function"]["arguments"])
                        _user_clarifications[id]["assistant_question"] = args[
                            "question"
                        ]

        CLEAN_USER_MESSAGE_PROMPT = """
    You are a specialist assistant that extracts the user's final intended message from a conversation.

    Task:
    - From the conversation history, return the final intended user message.

    Rules:
    - Apply all user interjections/corrections; the latest user message overrides earlier ones.
    - Ignore assistant messages; they are never part of the output.
    - Output exactly one plain string: the final corrected user message. No quotes, JSON, or explanation.
    - Do not add new information. Remove redundant or off-topic words.
    - If clarifications are provided, use them to construct the new user message.

    Examples:

    Input:
    Messages:
    [
    "user: Hi, what is the weather in Tokyo?",
    "user: Actually, I meant in Cairo"
    ]
    Clarifications: {}
    Output:
    Hi, what is the weather in Cairo?

    Input:
    Messages:
    [
    "user: Can you find the contact with the name John Doe?",
    "user: Sorry it's actually John Smith"
    ]
    Clarifications: {}
    Output:
    Can you find the contact with the name John Smith?

    Input:
    Messages:
    [
    "user: Book a flight to Paris",
    "user: Actually, make it Berlin"
    ]
    Clarifications:
    [{"assistant_question": "What date should I book it for?", "user_answer": "Next Friday"}]
    Output:
    Book a flight to Berlin next Friday.
    """

        global _CONFIG
        # Fast path: if there were no interjections and no clarifications, keep the
        # original initial message verbatim to preserve exact-match behaviour for cache keys.
        if (not messages_history) and not _user_clarifications:
            result = init_user_message
        else:
            client = _CONFIG.get_client()
            client.set_system_message(CLEAN_USER_MESSAGE_PROMPT)
            result = client.generate(
                user_message=f"Messages: {json.dumps(history)}\nClarifications: {json.dumps([v for _, v in _user_clarifications.items()])}",
            )
        return result

    def _clean_tool_trajectory(self, user_message, msgs, previous_tool_trajectory=None):

        class PruneToolsResponseFormat(BaseModel):
            call_indices: list[int]

        global _CONFIG

        cleaned_trajectory = []
        _flatten_tools = {}

        for msg in msgs:
            if msg.get("role") == "tool":
                # Skip completion status tools or anything not an actual tool call
                if not msg["tool_call_id"].startswith("call_"):
                    continue

                _flatten_tools[msg["tool_call_id"]] = msg

        for msg in msgs:
            if msg.get("role") != "assistant":
                continue

            if msg.get("tool_calls") is not None:
                for tool_call in msg.get("tool_calls"):
                    if (id := tool_call.get("id")) in _flatten_tools.keys():
                        if _flatten_tools[id].get("name") == "semantic_search":
                            continue

                        request = tool_call

                        response = _flatten_tools[id]
                        response.pop("tool_call_id")

                        pair = ToolCallPair(
                            request=request,
                            response=response,
                        )

                        cleaned_trajectory.append(pair)

        cleaned_trajectory = _simplify_tool_trajectory(cleaned_trajectory)
        if previous_tool_trajectory:
            cleaned_trajectory = [*previous_tool_trajectory, *cleaned_trajectory]
            # re-index the tool calls
            for idx, tool_call in enumerate(cleaned_trajectory):
                tool_call["index"] = idx

        client = _CONFIG.get_client()
        client.set_system_message(
            """
            You are a helpful assistant that cleans redundant tool calls, given a user query and a list of tool calls,
            you should return indices of the tool calls to prune, that are redundant/duplicate or not relevant to the user query.
            """,
        )
        res = client.generate(
            user_message=f"User query: {user_message}\nTool trajectory: {json.dumps(cleaned_trajectory, indent=2)}",
            response_format=PruneToolsResponseFormat,
        )

        res = PruneToolsResponseFormat.model_validate_json(res)

        cleaned_trajectory = [
            tool_call_pair
            for tool_call_pair in cleaned_trajectory
            if tool_call_pair["index"] not in res.call_indices
        ]

        # re-index the tool calls
        for idx, tool_call in enumerate(cleaned_trajectory):
            tool_call["index"] = idx

        return cleaned_trajectory

    def _save_semantic_cache(
        self,
        initial_user_message,
        user_message_visible_history,
        messages_history,
        previous_tool_trajectory,
        store_context,
        namespace,
    ):
        new_user_message = self._construct_new_user_message(
            initial_user_message,
            user_message_visible_history,
            messages_history,
        )

        tool_trajectory = self._clean_tool_trajectory(
            new_user_message,
            messages_history,
            previous_tool_trajectory=previous_tool_trajectory,
        )

        self._save_to_cache(store_context, namespace, new_user_message, tool_trajectory)

    def save(
        self,
        initial_user_message,
        user_message_visible_history,
        messages_history,
        previous_tool_trajectory,
        namespace,
    ):
        # Capture the resolved store context at submission time. This avoids
        # losing the context when executing inside a background thread where
        # contextvars are not propagated by default.
        store_context = _CONFIG.context
        self._submit(
            self._save_semantic_cache,
            initial_user_message,
            user_message_visible_history,
            messages_history,
            previous_tool_trajectory,
            store_context,
            namespace,
        )

    def wait(self, timeout: int = 360) -> bool:
        """
        Wait for all pending saving tasks to complete
        Returns True if all tasks completed, False if some timed out.
        """
        with self._future_lock:
            snapshot = set(self._futures)

        if not snapshot:
            return True

        _, pending = wait(snapshot, timeout=timeout)
        if pending:
            logger.warning(f"Some saving tasks timed out: {pending}")
            return False

        return True


_SEMANTIC_CACHE_SAVER = None


class _Config:
    threshold: float = 0.2
    top_k: int = 1
    embedding_model: str = "text-embedding-3-small"
    _model: str = "gpt-5@openai"
    _reasoning_effort = "high"
    _context: str = "Cache"

    @property
    def context(self):
        from unity import ASSISTANT_CONTEXT

        return f"{ASSISTANT_CONTEXT}/{self._context}"

    def get_client(self):
        return unify.Unify(self._model, reasoning_effort=self._reasoning_effort)


class ToolCallPair(TypedDict):
    request: Mapping[str, Any]
    response: Mapping[str, Any]


class ToolTrajectory(TypedDict):
    index: int
    name: str
    arguments: str
    result: str


@dataclass
class SemanticCacheResult:
    original_user_message: str
    closest_user_message: str
    tool_trajectory: list[ToolTrajectory]


_CONFIG = _Config()


def _simplify_tool_trajectory(tool_trajectory: list[ToolCallPair]):
    ret = []
    for idx, tool_call_pair in enumerate(tool_trajectory):
        name = tool_call_pair["request"]["function"]["name"]
        arguments = tool_call_pair["request"]["function"]["arguments"]
        result = tool_call_pair["response"]["content"]

        ret.append(
            {
                "index": idx,
                "name": name,
                "arguments": arguments,
                "result": result,
            },
        )

    return ret


def search_semantic_cache(
    user_message,
    namespace,
) -> SemanticCacheResult | None:
    global _CONFIG
    store_context = _CONFIG.context

    # Ensure context exists
    context_exist = store_context in unify.get_contexts(prefix=store_context)
    if not context_exist:
        unify.create_context(store_context)

    # Build distance/similarity expression once for consistent logging and querying
    _escaped = escape_single_quotes(user_message)
    metric_expr = f"cosine({_USER_MESSAGE_EMBEDDING_FIELD_NAME}, embed('{_escaped}', model='{_CONFIG.embedding_model}'))"
    # NOTE: On this backend, `cosine(a,b)` acts like a distance (lower is better).
    # We keep candidates with distance <= threshold and sort ascending so exact/close matches win.
    filter_expr = f"({metric_expr} <= {_CONFIG.threshold})"
    if namespace:
        filter_expr += f" and (namespace == '{namespace}')"

    logs = unify.get_logs(
        context=store_context,
        exclude_fields=[_USER_MESSAGE_EMBEDDING_FIELD_NAME],
        filter=filter_expr,
        sorting={metric_expr: "ascending"},
        limit=_CONFIG.top_k,
    )

    if logs:
        entries = logs[0].entries
        return SemanticCacheResult(
            original_user_message=user_message,
            closest_user_message=entries["user_message"],
            tool_trajectory=json.loads(entries["tool_trajectory"]),
        )

    return None


# Dummy tool placeholder (passed to async tool use loop)
def semantic_search_placeholder(user_message: str):
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


def get_system_msg_hint() -> str:
    return """
    You have access to a best-effort semantic cache of prior tool trajectories.
    For the current user message, the 'semantic_search' tool may return the set of tools that were previously called to answer a similar query, along with already-computed results.

    Guidance:
    - Prefer using those returned results directly when the status is "new", or follow a similar sequence of tools with updated arguments if required, before creating new or unrelated tool calls.
    - Treat them as latency-saving precomputed outputs to avoid redundant work and reduce time.
    - If gaps remain or the results do not fully address the request, call only the minimal additional tools needed.
    - If tools status is "cached", and you need the result, you must call the tool instead of using the cached result.

    Rules:
    - Do not call the 'semantic_search' tool again, it is only used to get the tool trajectory.
    - Do not call tools that are in the tool trajectory, they are already executed with the same arguments and has the "result_status":"new".
    - Do NOT state, hint, or imply that you are using a cache, 'semantic_search', prior runs, or precomputed results.
    - Write answers as if you executed the necessary tools now.
    """


async def get_dummy_tool(
    semantic_cache_result: SemanticCacheResult,
    tools: Mapping[str, ToolSpec],
):
    history = copy.deepcopy(semantic_cache_result.tool_trajectory)
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(thread_name_prefix="semantic_cache") as executor:
        awaitables_map = {}
        for idx, tool_call in enumerate(history):
            history[idx]["result_status"] = "cached"  # type: ignore

            if (tool_name := tool_call.get("name")) in tools:
                # Only re-call tools that are read-only
                if not tools[tool_name].read_only:
                    continue
                fn = tools[tool_name].fn
                try:
                    args = json.loads(tool_call.get("arguments")) or {}
                except Exception:
                    continue
                if inspect.iscoroutinefunction(fn):
                    task = loop.create_task(fn(**args))
                else:
                    task = loop.run_in_executor(
                        executor,
                        functools.partial(fn, **args),
                    )
                awaitables_map[idx] = task
        ordered_indices = list(awaitables_map.keys())
        results = await asyncio.gather(
            *[awaitables_map[i] for i in ordered_indices],
            return_exceptions=True,
        )
        for idx, result in zip(ordered_indices, results):
            if isinstance(result, Exception):
                continue

            history[idx]["result_status"] = "new"
            history[idx]["result"] = result

    call_id = f"call_SemanticSearchCallIdPlaceholder"
    request = {
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
    response = create_tool_call_message(
        name="semantic_search",
        call_id=call_id,
        content=_dumps(history, indent=2),
    )
    return [
        request,
        response,
    ]


def save_semantic_cache(
    initial_user_message,
    user_message_visible_history,
    messages_history,
    namespace,
    previous_tool_trajectory=None,
):
    global _SEMANTIC_CACHE_SAVER
    if _SEMANTIC_CACHE_SAVER is None:
        _SEMANTIC_CACHE_SAVER = _SemanticCacheSaver()

    _SEMANTIC_CACHE_SAVER.save(
        initial_user_message,
        user_message_visible_history,
        messages_history,
        previous_tool_trajectory,
        namespace,
    )
