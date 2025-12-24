import json
import unify
import inspect
import copy
import asyncio
import functools
import atexit
import logging

from dataclasses import dataclass
import threading
from typing import Any, Mapping, Callable, List
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, wait

from unity.common.log_utils import log as unity_log
from unity.common.tool_spec import ToolSpec, normalise_tools
from unity.common.llm_client import new_llm_client

from .tools_data import create_tool_call_message
from ..semantic_search import escape_single_quotes
from ..llm_helpers import _dumps
from .transcript_ops import (
    extract_assistant_and_tool_steps as _extract_assistant_and_tool_steps,
    extract_clarifications as _extract_clarifications,
)
from .transcript_ops import (
    build_clean_tool_trajectory as _build_clean_tool_trajectory,
    CleanToolCall,
)


_SEMANTIC_CACHE_SAVER: "_SemanticCacheSaver | None" = None
_USER_MESSAGE_EMBEDDING_FIELD_NAME = "_user_message_emb"
logger = logging.getLogger(__name__)


@dataclass
class SemanticCacheResult:
    original_user_message: str
    closest_user_message: str
    tool_trajectory: List[CleanToolCall]


class _Config:
    threshold: float = 0.2
    top_k: int = 1
    embedding_model: str = "text-embedding-3-small"
    _context: str = "Cache"

    @property
    def context(self):
        from unity.session_details import SESSION_DETAILS

        return f"{SESSION_DETAILS.user_context}/{SESSION_DETAILS.assistant_context}/{self._context}"

    def get_client(self):
        return new_llm_client(async_client=False)


_CONFIG = _Config()


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

    def _save_to_cache(
        self,
        store_context,
        namespace,
        user_message,
        tool_trajectory,
    ):
        # store_context is captured at call-time to avoid thread-local context loss
        unify.create_context(store_context)

        log_id = unity_log(
            context=store_context,
            user_message=user_message,
            namespace=namespace,
            tool_trajectory=tool_trajectory,
        )

        embed_expr = f"embed({{logs:user_message}}, model='{_CONFIG.embedding_model}', async_embeddings=False)"
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

        # Build Clarifications using shared transcript helpers
        extracted = _extract_assistant_and_tool_steps(
            full_messages_history or [],
            allowed_tools=None,
        )
        clar_list = []
        try:
            clar_summ = _extract_clarifications(
                extracted.get("assistant_steps") or [],
                extracted.get("tool_results") or [],
                callid_to_tool_name=extracted.get("callid_to_tool_name", {}),
            )
            # Map to the shape expected by the prompt
            for c in clar_summ:
                q = c.get("question")
                if q is not None:
                    clar_list.append(
                        {"assistant_question": q, "user_answer": ""},
                    )
        except Exception:
            clar_list = []

        # Additionally pair explicit request_clarification calls with their answers
        try:
            # Map tool_call_id -> answer for request_clarification results
            answers_by_cid = {}
            for tm in extracted.get("tool_results") or []:
                try:
                    if (
                        tm.get("role") == "tool"
                        and tm.get("name") == "request_clarification"
                    ):
                        cid = tm.get("tool_call_id")
                        if isinstance(cid, str) and cid:
                            answers_by_cid[cid] = tm.get("content")
                except Exception:
                    continue

            # Walk assistant tool_calls to find request_clarification questions
            for am in extracted.get("assistant_steps") or []:
                try:
                    for tc in am.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        if fn.get("name") != "request_clarification":
                            continue
                        cid = tc.get("id")
                        args_json = fn.get("arguments", "{}")
                        try:
                            args = (
                                json.loads(args_json)
                                if isinstance(args_json, str)
                                else (args_json or {})
                            )
                        except Exception:
                            args = {}
                        q = args.get("question")
                        if q is None:
                            continue
                        ans = (
                            answers_by_cid.get(cid, "") if isinstance(cid, str) else ""
                        )
                        clar_list.append(
                            {"assistant_question": q, "user_answer": ans},
                        )
                except Exception:
                    continue
        except Exception:
            pass

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
        if (not messages_history) and not clar_list:
            result = init_user_message
        else:
            client = _CONFIG.get_client()
            client.set_system_message(CLEAN_USER_MESSAGE_PROMPT)
            result = client.generate(
                user_message=(
                    f"Messages: {json.dumps(history)}\n"
                    f"Clarifications: {json.dumps(clar_list)}"
                ),
            )
        return result

    def _prune_tool_trajectory(
        self,
        user_message,
        tool_trajectory,
    ) -> List[CleanToolCall]:
        class PruneToolsResponseFormat(BaseModel):
            call_indices: List[int]

        global _CONFIG
        client = _CONFIG.get_client()
        client.set_system_message(
            """
            You are a helpful assistant that cleans redundant tool calls, given a user query and a list of tool calls,
            you should return indices of the tool calls to prune, that are redundant/duplicate or not relevant to the user query.
            """,
        )
        res = client.generate(
            user_message=f"User query: {user_message}\nTool trajectory: {json.dumps(tool_trajectory, indent=2)}",
            response_format=PruneToolsResponseFormat,
        )

        res = PruneToolsResponseFormat.model_validate_json(res)

        cleaned_trajectory = [
            tool_call
            for tool_call in tool_trajectory
            if tool_call["index"] not in res.call_indices
        ]

        return cleaned_trajectory

    def _clean_tool_trajectory(
        self,
        user_message,
        msgs,
        previous_tool_trajectory=None,
    ) -> List[CleanToolCall]:
        # Build a cleaned trajectory using the shared transcript ops helper,
        # ensuring we drop the semantic_search placeholder results.
        cleaned_trajectory = _build_clean_tool_trajectory(
            msgs,
            drop_names={"semantic_search"},
        )

        # Prepend any prior trajectory steps when provided
        if previous_tool_trajectory:
            cleaned_trajectory = [*previous_tool_trajectory, *cleaned_trajectory]

        # Re-index before pruning (for deterministic prompts)
        for idx, tool_call in enumerate(cleaned_trajectory):
            tool_call["index"] = idx

        # Prune redundant/irrelevant calls via the LLM-based cleaner
        cleaned_trajectory = self._prune_tool_trajectory(
            user_message,
            cleaned_trajectory,
        )

        # Re-index after pruning to keep a compact, stable ordering
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

        self._save_to_cache(
            store_context,
            namespace,
            new_user_message,
            tool_trajectory,
        )

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


def search_semantic_cache(
    user_message,
    namespace,
) -> SemanticCacheResult | None:
    global _CONFIG
    store_context = _CONFIG.context

    unify.create_context(store_context)

    _escaped = escape_single_quotes(user_message)
    metric_expr = f"cosine({_USER_MESSAGE_EMBEDDING_FIELD_NAME}, embed('{_escaped}', model='{_CONFIG.embedding_model}', async_embeddings=False))"
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
            tool_trajectory=entries["tool_trajectory"],
        )

    return None


# Dummy tool placeholder (passed to async tool use loop)
def semantic_search_placeholder(user_message: str):
    """
    Search a semantic cache for prior solutions relevant to the given user_message.

    Returns a tool trajectory from a similar past query. Each tool in the trajectory
    has a `result_status` field:

    - "fresh": The tool was AUTOMATICALLY RE-EXECUTED just now with up-to-date data.
      The result is CURRENT and ACCURATE. Do NOT call this tool again — the fresh
      result is already provided and calling it would be redundant.

    - "stale": The tool was NOT re-executed (e.g., it has side effects). The result
      is from a previous run and may be outdated. You may need to call this tool
      yourself if you need current data.

    IMPORTANT: For tools with result_status="fresh", the data is guaranteed current.
    There is NO reason to call these tools again. Use the provided results directly.

    Input: user_message (str)
    Output: List[dict] with {"name", "arguments", "result", "result_status"}, in order.
    """


def get_system_msg_hint() -> str:
    return """
    You have access to a semantic cache that provides pre-computed tool results.
    The 'semantic_search' tool returns a trajectory of tools from a similar past query.

    CRITICAL: Each tool result has a `result_status`:
    - "fresh" = Tool was AUTOMATICALLY RE-EXECUTED with current data. Use this result
      directly. Do NOT call the tool again — it would be redundant and wasteful.
    - "stale" = Tool was not re-executed. Result may be outdated. Call if needed.

    Rules:
    1. For result_status="fresh": TRUST the result. It is current. Do NOT re-call.
    2. For result_status="stale": Call the tool yourself if you need fresh data.
    3. Only call additional tools if the cached trajectory doesn't cover your needs.
    4. Never mention the cache, semantic_search, or prior runs to the user.
    5. Write your answer as if you executed the tools yourself just now.
    """


def _is_manager_tool(tool: ToolSpec) -> bool:
    return tool.manager_tool


async def _handle_manager_tool(tool: ToolSpec, args):
    base = tool.fn.__self__.__class__
    # Best-effort to get the namespace from the tool
    namespace = f"{base.__name__}.{tool.fn.__name__}"
    usermessage = args["text"]
    result = search_semantic_cache(usermessage, namespace)
    if not result:
        raise Exception("No result found in semantic cache")
    tools = {}

    manager = base()
    tools = manager.get_tools(tool.fn.__name__, include_sub_tools=True)
    tools = normalise_tools(tools)
    history = await _rexecute_tools(result.tool_trajectory, tools)
    return history


async def _rexecute_tools(tool_trajectory, tools):
    history = copy.deepcopy(tool_trajectory)
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(thread_name_prefix="semantic_cache") as executor:
        awaitables_map = {}
        for idx, tool_call in enumerate(history):
            history[idx]["result_status"] = "stale"  # type: ignore

            if (tool_name := tool_call.get("name")) in tools:
                if _is_manager_tool(tools[tool_name]):
                    args = json.loads(tool_call.get("arguments")) or {}
                    task = loop.create_task(
                        _handle_manager_tool(tools[tool_name], args),
                    )
                    awaitables_map[idx] = task
                    continue

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

            history[idx]["result_status"] = "fresh"
            history[idx]["result"] = result

    return history


async def get_dummy_tool(
    semantic_cache_result: SemanticCacheResult,
    tools: Mapping[str, ToolSpec],
):
    history = await _rexecute_tools(semantic_cache_result.tool_trajectory, tools)
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
